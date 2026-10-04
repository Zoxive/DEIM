"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DETR (https://github.com/facebookresearch/detr/blob/main/engine.py)
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""


import sys
import math
from typing import Iterable

import torch
import torch.amp
from torch.utils.tensorboard import SummaryWriter
from torch.amp.grad_scaler import GradScaler

from ..optim import ModelEMA, Warmup
from ..data import CocoEvaluator, device_batch
from ..misc import MetricLogger, SmoothedValue, dist_utils


def _encoder_transformer_grad_percentage(model: torch.nn.Module):
    """The AIFI encoder transformer's share of the whole model's gradient L1, in percent (RT-DETRv4's
    signal for retuning the distillation weight; det_solver.py). The loss scale cancels in the ratio.
    None on a step whose gradients overflowed, which the scaler skips anyway."""
    total, encoder = 0.0, 0.0
    for name, param in dist_utils.de_parallel(model).named_parameters():
        if param.grad is None:
            continue
        value = param.grad.detach().abs().sum().item()
        total += value
        if name.startswith('encoder.encoder'):
            encoder += value
    if total <= 0.0 or not math.isfinite(total):
        return None
    return 100.0 * encoder / total


def train_one_epoch(self_lr_scheduler, lr_scheduler, model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, **kwargs):
    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    print_freq = kwargs.get('print_freq', 10)
    writer :SummaryWriter = kwargs.get('writer', None)

    ema :ModelEMA = kwargs.get('ema', None)
    scaler :GradScaler = kwargs.get('scaler', None)
    # torch dtype when mixed precision is on, else None. A scaler without a dtype was the original
    # code path (fp16 by CUDA default); keep that reading so nothing that passes only `scaler` changes.
    amp_dtype :torch.dtype = kwargs.get('amp_dtype', torch.float16 if scaler is not None else None)
    lr_warmup_scheduler :Warmup = kwargs.get('lr_warmup_scheduler', None)

    # engine/solver/nonfinite_guard.py - opt-in with `--nonfinite-guard`; None leaves the loop as it was.
    guard = kwargs.get('nonfinite_guard', None)
    channels_last = kwargs.get('channels_last', False)

    # Feature distillation (engine/deim/dinov3_teacher.py): None leaves the loop as it was. With a teacher,
    # the encoder's gradient share is collected per step when the criterion retunes its weight.
    teacher_model = kwargs.get('teacher_model', None)
    adaptive = bool((getattr(criterion, 'distill_adaptive_params', None) or {}).get('enabled', False))
    encoder_grad_percentages = []
    model_kwargs = {}

    cur_iters = epoch * len(data_loader)

    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples = samples.to(device, non_blocking=True)
        targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
        # A uint8 batch (ConvertPILImage dtype uint8) is scaled, mixed up and resized here, on the GPU.
        samples, targets = device_batch(samples, targets)
        if channels_last:
            samples = samples.contiguous(memory_format=torch.channels_last)
        global_step = epoch * len(data_loader) + i
        metas = dict(epoch=epoch, step=i, global_step=global_step, epoch_step=len(data_loader))

        if teacher_model is not None:
            # fp32, outside autocast, as upstream RT-DETRv4 runs it.
            model_kwargs = {'teacher_encoder_output': teacher_model(samples)}

        if guard is not None and amp_dtype is not None:
            guard.snapshot()
            with torch.autocast(device_type=str(device), dtype=amp_dtype, cache_enabled=True):
                outputs = model(samples, targets=targets, **model_kwargs)
            boxes_finite = bool(torch.isfinite(outputs['pred_boxes']).all())
            guard.log_step(epoch, i, targets, boxes_finite)
            if not boxes_finite:
                del outputs
                optimizer.zero_grad(set_to_none=True)
                guard.handle(samples, targets, epoch, i, amp_dtype, dataset=getattr(data_loader, 'dataset', None))
                # The step is skipped: no backward, no optimizer or EMA update. The LR schedule is indexed
                # by iteration, so it still advances and the run stays on the same schedule.
                if self_lr_scheduler:
                    optimizer = lr_scheduler.step(cur_iters + i, optimizer)
                elif lr_warmup_scheduler is not None:
                    lr_warmup_scheduler.step()
                continue

        elif amp_dtype is not None:
            with torch.autocast(device_type=str(device), dtype=amp_dtype, cache_enabled=True):
                outputs = model(samples, targets=targets, **model_kwargs)

        if amp_dtype is not None:
            if torch.isnan(outputs['pred_boxes']).any() or torch.isinf(outputs['pred_boxes']).any():
                print(outputs['pred_boxes'])
                state = model.state_dict()
                new_state = {}
                for key, value in model.state_dict().items():
                    # Replace 'module' with 'model' in each key
                    new_key = key.replace('module.', '')
                    # Add the updated key-value pair to the state dictionary
                    state[new_key] = value
                new_state['model'] = state
                dist_utils.save_on_master(new_state, "./NaN.pth")

            with torch.autocast(device_type=str(device), enabled=False):
                loss_dict = criterion(outputs, targets, **metas)

            loss = sum(loss_dict.values())
            if scaler is not None:
                scaler.scale(loss).backward()

                if max_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

                if adaptive:
                    pct = _encoder_transformer_grad_percentage(model)
                    if pct is not None:
                        encoder_grad_percentages.append(pct)

                scaler.step(optimizer)
                scaler.update()
            else:
                # bfloat16: fp32's exponent range, so no loss scaling and no skipped steps.
                loss.backward()
                if max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                if adaptive:
                    pct = _encoder_transformer_grad_percentage(model)
                    if pct is not None:
                        encoder_grad_percentages.append(pct)
                optimizer.step()
            optimizer.zero_grad()

        else:
            outputs = model(samples, targets=targets, **model_kwargs)
            loss_dict = criterion(outputs, targets, **metas)

            loss : torch.Tensor = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()

            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            if adaptive:
                pct = _encoder_transformer_grad_percentage(model)
                if pct is not None:
                    encoder_grad_percentages.append(pct)

            optimizer.step()

        # ema
        if ema is not None:
            ema.update(model)

        if self_lr_scheduler:
            optimizer = lr_scheduler.step(cur_iters + i, optimizer)
        else:
            if lr_warmup_scheduler is not None:
                lr_warmup_scheduler.step()

        loss_dict_reduced = dist_utils.reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if writer and dist_utils.is_main_process() and global_step % 10 == 0:
            writer.add_scalar('Loss/total', loss_value.item(), global_step)
            for j, pg in enumerate(optimizer.param_groups):
                writer.add_scalar(f'Lr/pg_{j}', pg['lr'], global_step)
            for k, v in loss_dict_reduced.items():
                writer.add_scalar(f'Loss/{k}', v.item(), global_step)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if adaptive:
        return stats, encoder_grad_percentages
    return stats


@torch.no_grad()
def evaluate(model: torch.nn.Module, criterion: torch.nn.Module, postprocessor, data_loader, coco_evaluator: CocoEvaluator, device):
    model.eval()
    criterion.eval()
    coco_evaluator.cleanup()

    metric_logger = MetricLogger(delimiter="  ")
    # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    # iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessor.keys())
    iou_types = coco_evaluator.iou_types
    # coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

        results = postprocessor(outputs, orig_target_sizes)

        # if 'segm' in postprocessor.keys():
        #     target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        #     results = postprocessor['segm'](results, outputs, orig_target_sizes, target_sizes)

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    extended_stats = None

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        extended_stats = coco_evaluator.summarize()

    stats = {}
    # stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in iou_types:
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in iou_types:
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()

    return stats, coco_evaluator, extended_stats
