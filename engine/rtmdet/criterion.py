"""RTMDet's training loss: DynamicSoftLabelAssigner + QualityFocalLoss + GIoU, as MMDetection 3.3.0 has them
(`task_modules/assigners/dynamic_soft_label_assigner.py`, `losses/gfocal_loss.py`, `dense_heads/rtmdet_head.py`,
Apache-2.0), on DEIM's targets: per image `boxes` cxcywh normalised to the input and `labels`.

Normalisation is mmdet's: the class loss is summed over every prior and divided by the sum of the positives'
IoU (their soft labels), the box loss is IoU-weighted and divided by the same sum, both at least 1.

With a distribution box head (`RTMDet(reg_max > 0)`, `pred_corners` in the outputs) a third term, Distribution Focal
Loss (GFL, mmdet's `losses/gfocal_loss.py::distribution_focal_loss`), pulls each positive's edge distribution onto the
two bins either side of the true distance; it is weighted and normalised like the box loss, averaged over the four
edges, and needs `loss_dfl` in weight_dict (GFL uses 0.25).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from ..core import register

__all__ = ['RTMDetCriterion']

EPS = 1.0e-7
INF = 100000000.0


def pairwise_iou(a, b, eps=1e-6):
    """mmdet's bbox_overlaps: the union is clamped, so an inverted predicted box (its distances are not
    constrained positive for tiny/s) scores 0 rather than torchvision's negative or infinite IoU."""
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    overlap = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return overlap / (area_a[:, None] + area_b[None] - overlap).clamp(min=eps)


def dynamic_soft_label_assign(priors, scores, boxes, gt_boxes, gt_labels, soft_center_radius=3.0, topk=13,
                              iou_weight=3.0):
    """One image. priors [N, 4] (x, y, stride, stride), scores [N, C] logits, boxes [N, 4] xyxy, both detached.
    Returns per prior: the matched gt index (-1 for background) and the matched IoU.

    mmdet's assignment, written without its per-gt loop or data-dependent indexing so a step never waits on
    the GPU: costs are computed for every prior and the priors whose centre lies in no gt box are masked out
    (cost INF, IoU 0), which selects exactly what mmdet's valid-prior subset selects."""
    num_priors, num_gt = priors.shape[0], gt_boxes.shape[0]
    if num_gt == 0:
        return priors.new_full((num_priors,), -1, dtype=torch.long), priors.new_zeros((num_priors,))

    centers = priors[:, :2]
    deltas = torch.cat([centers[:, None] - gt_boxes[None, :, :2], gt_boxes[None, :, 2:] - centers[:, None]], -1)
    valid = (deltas.min(dim=-1).values > 0).any(dim=1)

    gt_center = (gt_boxes[:, :2] + gt_boxes[:, 2:]) / 2.0
    distance = (centers[:, None] - gt_center[None]).pow(2).sum(-1).sqrt() / priors[:, 2:3]
    soft_center_prior = torch.pow(10, distance - soft_center_radius)

    ious = pairwise_iou(boxes, gt_boxes)
    iou_cost = -torch.log(ious + EPS) * iou_weight

    soft_label = F.one_hot(gt_labels.long(), scores.shape[-1]).float()[None] * ious[..., None]
    pred = scores[:, None, :].expand(-1, num_gt, -1)
    scale = soft_label - pred.sigmoid()
    soft_cls_cost = (F.binary_cross_entropy_with_logits(pred, soft_label, reduction='none')
                     * scale.abs().pow(2.0)).sum(-1)
    cost = torch.where(valid[:, None], soft_cls_cost + iou_cost + soft_center_prior, INF)
    ious = ious * valid[:, None]

    # dynamic k: each gt takes as many priors as the sum of its top-13 IoUs, at least one; a prior claimed
    # by several gts goes to the cheapest.
    k = min(topk, num_priors)
    dynamic_ks = torch.clamp(torch.topk(ious, k, dim=0).values.sum(0).int(), min=1)
    _, candidates = torch.topk(cost, k, dim=0, largest=False)
    take = torch.arange(k, device=priors.device)[:, None] < dynamic_ks[None]
    matching = torch.zeros_like(cost, dtype=torch.bool).scatter_(0, candidates, take) & valid[:, None]
    multiple = matching.sum(1) > 1
    cheapest = F.one_hot(cost.argmin(dim=1), num_gt).bool()
    matching = torch.where(multiple[:, None], cheapest, matching)

    fg = matching.any(dim=1)
    matched_gt = torch.where(fg, matching.float().argmax(dim=1), -1)
    matched_iou = (matching.float() * ious).sum(1)
    return matched_gt, matched_iou


def quality_focal_loss(logits, labels, quality, beta=2.0):
    """logits [N, C]; labels [N] in [0, C) for positives, C for background; quality [N] (IoU). Per-prior sum.

    mmdet treats positive and negative entries with two formulas; with the soft target t (the IoU at the
    positive's class, 0 elsewhere) both are BCE(x, t) * |t - sigmoid(x)|^beta, so one expression covers both."""
    target = F.one_hot(labels, logits.shape[1] + 1)[:, :-1].float() * quality[:, None]
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction='none') \
        * (target - logits.sigmoid()).abs().pow(beta)
    return loss.sum(dim=1)


def giou_loss(pred, target, eps=1e-6):
    """mmdet's GIoULoss per pair, xyxy."""
    lt = torch.max(pred[:, :2], target[:, :2])
    rb = torch.min(pred[:, 2:], target[:, 2:])
    wh = (rb - lt).clamp(min=0)
    overlap = wh[:, 0] * wh[:, 1]
    area_p = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    area_t = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
    union = (area_p + area_t - overlap).clamp(min=eps)
    ious = overlap / union
    enclose = (torch.max(pred[:, 2:], target[:, 2:]) - torch.min(pred[:, :2], target[:, :2])).clamp(min=0)
    enclose_area = (enclose[:, 0] * enclose[:, 1]).clamp(min=eps)
    return 1 - (ious - (enclose_area - union) / enclose_area)


def distribution_focal_loss(corners, target):
    """corners [P, 4, R+1] logits; target [P, 4] distances in stride units, already clamped to [0, R). Per positive,
    the mean over edges of CE to the lower bin weighted by its closeness plus CE to the upper bin likewise."""
    left = target.long()
    right = left + 1
    weight_left = right.float() - target
    weight_right = target - left.float()
    logp = corners.log_softmax(-1)
    loss = -(logp.gather(-1, left[..., None])[..., 0] * weight_left
             + logp.gather(-1, right[..., None])[..., 0] * weight_right)
    return loss.mean(-1)


def bbox2distance(points, boxes, stride, reg_max):
    """Prior centres [P, 2], xyxy boxes [P, 4], strides [P] -> [P, 4] (left, top, right, bottom) in stride units,
    clamped just under reg_max so the upper bin exists (mmdet's bbox2distance with max_dis = reg_max - 0.01)."""
    distance = torch.stack([points[:, 0] - boxes[:, 0], points[:, 1] - boxes[:, 1],
                            boxes[:, 2] - points[:, 0], boxes[:, 3] - points[:, 1]], -1) / stride[:, None]
    return distance.clamp(min=0, max=reg_max - 0.01)


def distill_loss(pairs):
    """Mean over levels of the per-location cosine distance between the student's projected neck feature and the
    teacher's encoder feature (RT-DETRv4's form, engine/deim/deim_criterion.py), both [B, C, H, W]."""
    total = 0.0
    for student, teacher in pairs:
        if student.shape != teacher.shape:
            raise ValueError(f'distillation shape mismatch: student {tuple(student.shape)}, teacher {tuple(teacher.shape)}')
        total = total + (1 - F.cosine_similarity(student.float(), teacher.float(), dim=1)).mean()
    return total / len(pairs)


@register()
class RTMDetCriterion(nn.Module):
    __share__ = ['num_classes']

    def __init__(self, num_classes=80, weight_dict=None, topk=13, soft_center_radius=3.0, iou_weight=3.0,
                 beta=2.0):
        super().__init__()
        self.num_classes = num_classes
        self.weight_dict = weight_dict or {'loss_cls': 1.0, 'loss_bbox': 2.0}
        self.topk, self.soft_center_radius, self.iou_weight, self.beta = topk, soft_center_radius, iou_weight, beta

    def forward(self, outputs, targets, **kwargs):
        logits = outputs['pred_logits'].float()
        boxes = outputs['pred_boxes'].float()
        priors = outputs['priors'].float()
        h, w = outputs['input_hw']
        scale = boxes.new_tensor([w, h, w, h])

        all_labels, all_quality, all_targets = [], [], []
        for b, target in enumerate(targets):
            gt = torchvision.ops.box_convert(target['boxes'].float(), 'cxcywh', 'xyxy') * scale
            gt_labels = target['labels']
            with torch.no_grad():
                matched_gt, matched_iou = dynamic_soft_label_assign(
                    priors, logits[b].detach(), boxes[b].detach(), gt, gt_labels,
                    self.soft_center_radius, self.topk, self.iou_weight)
            pos = matched_gt >= 0
            safe = matched_gt.clamp(min=0)
            if gt.shape[0]:
                labels = torch.where(pos, gt_labels[safe].long(), self.num_classes)
                box_targets = gt[safe] * pos[:, None]
            else:
                labels = torch.full_like(matched_gt, self.num_classes)
                box_targets = torch.zeros_like(boxes[b])
            all_labels.append(labels)
            all_quality.append(matched_iou)
            all_targets.append(box_targets)

        labels = torch.cat(all_labels)
        quality = torch.cat(all_quality)
        box_targets = torch.cat(all_targets)
        flat_logits = logits.reshape(-1, logits.shape[-1])
        flat_boxes = boxes.reshape(-1, 4)
        corners = outputs.get('pred_corners')
        if corners is not None and 'loss_dfl' not in self.weight_dict:
            raise ValueError('the model has a distribution box head (pred_corners) but weight_dict has no loss_dfl')

        avg_factor = quality.sum().clamp(min=1.0)
        loss_cls = quality_focal_loss(flat_logits, labels, quality, self.beta).sum() / avg_factor

        pos = labels < self.num_classes
        loss_dfl = None
        if pos.any():
            weight = quality[pos]
            loss_bbox = (giou_loss(flat_boxes[pos], box_targets[pos]) * weight).sum() / weight.sum().clamp(min=1.0)
            if corners is not None:
                reg_max = corners.shape[-1] - 1
                flat_priors = priors[None].expand(logits.shape[0], -1, -1).reshape(-1, 4)[pos]
                target = bbox2distance(flat_priors[:, :2], box_targets[pos], flat_priors[:, 2], reg_max)
                loss_dfl = (distribution_focal_loss(corners.reshape(-1, 4, reg_max + 1)[pos], target)
                            * weight).sum() / weight.sum().clamp(min=1.0)
        else:
            loss_bbox = flat_boxes.sum() * 0
        if corners is not None and loss_dfl is None:
            loss_dfl = corners.sum() * 0

        losses = {
            'loss_cls': loss_cls * self.weight_dict['loss_cls'],
            'loss_bbox': loss_bbox * self.weight_dict['loss_bbox'],
        }
        if loss_dfl is not None:
            losses['loss_dfl'] = loss_dfl * self.weight_dict['loss_dfl']
        if 'distill_pairs' in outputs:
            if 'loss_distill' not in self.weight_dict:
                raise ValueError('the model returned distillation features but weight_dict has no loss_distill')
            losses['loss_distill'] = self.weight_dict['loss_distill'] * distill_loss(outputs['distill_pairs'])
        return losses
