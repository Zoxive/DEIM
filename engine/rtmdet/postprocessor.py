"""RTMDet's test-time decode, as MMDetection 3.3.0's `BaseDenseHead._predict_by_feat_single` runs it with
RTMDet's `test_cfg`: sigmoid scores, per-level score floor and top-`nms_pre` over (prior, class) pairs, boxes
clamped to the input, rescaled to the original image, empty boxes dropped, class-aware NMS at 0.65, at most
`max_per_img` kept. One prior can survive as several classes.

The deploy output is DEIM's: `labels, boxes, scores` as [B, K] / [B, K, 4] / [B, K] tensors in original-image
pixels, K = `max_per_img`, so every instrument that reads a DEIM postprocessor reads this one. Rows past an
image's last detection are padding: score 0 and label `num_classes`.

Exporting to ONNX (or with `export_mode` set, for the PyTorch side of an export's parity check) the deploy output
is DEIM's PostProcessor exactly instead: the top `max_per_img` (prior, class) pairs by score, no NMS, a graph with
a dynamic batch and no data-dependent shapes, for an exported graph whose caller runs NMS on the host.
"""

import torch
import torch.nn as nn
import torchvision

from ..core import register

__all__ = ['RTMDetPostProcessor']


@register()
class RTMDetPostProcessor(nn.Module):
    __share__ = ['num_classes']

    def __init__(self, num_classes=80, score_thr=0.001, nms_pre=30000, iou_threshold=0.65, max_per_img=300):
        super().__init__()
        self.num_classes = num_classes
        self.score_thr, self.nms_pre, self.iou_threshold, self.max_per_img = score_thr, nms_pre, iou_threshold, max_per_img
        self.deploy_mode = False
        self.export_mode = False

    def extra_repr(self):
        return (f'num_classes={self.num_classes}, score_thr={self.score_thr}, nms_pre={self.nms_pre}, '
                f'iou_threshold={self.iou_threshold}, max_per_img={self.max_per_img}')

    def single(self, logits, boxes, priors, scale):
        """One image: logits [N, C], boxes [N, 4] in input pixels, scale [4] input -> original."""
        scores = logits.sigmoid()
        kept_scores, kept_labels, kept_boxes = [], [], []
        for stride in priors[:, 2].unique(sorted=True):
            level = (priors[:, 2] == stride).nonzero().squeeze(1)
            level_scores = scores[level]
            valid = level_scores > self.score_thr
            values = level_scores[valid]
            pairs = valid.nonzero()
            values, order = values.sort(descending=True)
            order = order[:self.nms_pre]
            pairs = pairs[order]
            kept_scores.append(values[:self.nms_pre])
            kept_labels.append(pairs[:, 1])
            kept_boxes.append(boxes[level[pairs[:, 0]]])
        scores, labels, boxes = torch.cat(kept_scores), torch.cat(kept_labels), torch.cat(kept_boxes) * scale
        wh = boxes[:, 2:] - boxes[:, :2]
        nonempty = (wh > 0).all(dim=1)
        scores, labels, boxes = scores[nonempty], labels[nonempty], boxes[nonempty]
        keep = torchvision.ops.batched_nms(boxes, scores, labels, self.iou_threshold)[:self.max_per_img]
        return labels[keep], boxes[keep], scores[keep]

    def forward(self, outputs, orig_target_sizes: torch.Tensor):
        logits = outputs['pred_logits'].float()
        boxes = outputs['pred_boxes'].float()
        priors = outputs['priors']
        h, w = outputs['input_hw']
        boxes = torch.stack([boxes[..., 0].clamp(0, w), boxes[..., 1].clamp(0, h),
                             boxes[..., 2].clamp(0, w), boxes[..., 3].clamp(0, h)], -1)
        sizes = orig_target_sizes.to(boxes.dtype)
        scales = torch.stack([sizes[:, 0] / w, sizes[:, 1] / h, sizes[:, 0] / w, sizes[:, 1] / h], 1)

        if self.deploy_mode and (self.export_mode or torch.onnx.is_in_onnx_export()):
            scores, index = torch.topk(logits.sigmoid().flatten(1), self.max_per_img, dim=-1)
            labels = index % self.num_classes
            index = index // self.num_classes
            boxes = boxes.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, 4)) * scales[:, None, :]
            return labels, boxes, scores

        # One size row for the whole batch broadcasts, as DEIM's PostProcessor does (its exporter warms up that way).
        scales = scales.expand(logits.shape[0], -1)
        results = [self.single(logits[b], boxes[b], priors, scales[b]) for b in range(logits.shape[0])]
        if not self.deploy_mode:
            return [dict(labels=lab, boxes=box, scores=sco) for lab, box, sco in results]

        k = self.max_per_img
        batch = len(results)
        labels = torch.full((batch, k), self.num_classes, dtype=torch.long, device=logits.device)
        out_boxes = torch.zeros((batch, k, 4), device=logits.device)
        scores = torch.zeros((batch, k), device=logits.device)
        for b, (lab, box, sco) in enumerate(results):
            n = lab.shape[0]
            labels[b, :n], out_boxes[b, :n], scores[b, :n] = lab, box, sco
        return labels, out_boxes, scores

    def deploy(self):
        self.eval()
        self.deploy_mode = True
        return self
