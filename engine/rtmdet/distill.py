"""A frozen DEIM detector as the teacher for RTMDet feature distillation (training only).

Meant for a dataset where RTMDet's boxes come out looser than a DEIM model's. The published
localisation-distillation methods (LD, CrossKD) need a dense teacher head that shares the student's priors, and DEIM is
a query-based DETR, so the transfer here is feature distillation instead. RTMDet's neck outputs, projected to 256
channels by a 1x1 conv per level (`RTMDet(distill_dim=256)`, trained and then unused at inference), are pulled toward
DEIM's encoder outputs at the same strides (its PAN outputs, what DEIM's decoder reads boxes from) by a per-location
cosine loss, as RT-DETRv4 does with its DINOv3 teacher (`engine/deim/dinov3_teacher.py`).

The teacher sees the student's batch (RGB 0-1, any multi-scale size); DEIM takes the same input contract. It is built
from its own config with `eval_spatial_size` unset, so its AIFI position embedding follows the batch's size, and runs in
fp16 under autocast: the fp32attn checkpoint holds its attention in fp32, which is what kept its training finite.
"""

import os

import torch
import torch.nn as nn

from ..core import register

__all__ = ['DEIMTeacher']

DEIM_SRC = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@register()
class DEIMTeacher(nn.Module):
    def __init__(self, config='configs/deim_dfine/bakeoff_d-nohand-rot90-nc-fp32attn.yml',
                 checkpoint='outputs/bakeoff_d-nohand-rot90-nc-fp32attn/best_stg1.pth', fp16=True):
        super().__init__()
        from ..core import YAMLConfig

        config = config if os.path.isabs(config) else os.path.join(DEIM_SRC, config)
        checkpoint = checkpoint if os.path.isabs(checkpoint) else os.path.join(DEIM_SRC, checkpoint)
        cfg = YAMLConfig(config, eval_spatial_size=None)
        if 'HGNetv2' in cfg.yaml_cfg:
            cfg.yaml_cfg['HGNetv2']['pretrained'] = False
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        state = state['ema']['module'] if 'ema' in state else state['model']
        # The decoder's anchors / valid_mask are derived from eval_spatial_size, not trained; unset, it has none.
        state = {k: v for k, v in state.items() if k not in ('decoder.anchors', 'decoder.valid_mask')}
        model = cfg.model
        model.load_state_dict(state)
        self.backbone, self.encoder = model.backbone, model.encoder
        for p in self.parameters():
            p.requires_grad = False
        self.fp16 = fp16
        self.strides = (8, 16, 32)
        self.teacher_feature_dim = self.encoder.hidden_dim
        print(f'[DEIMTeacher] {os.path.relpath(checkpoint, DEIM_SRC)}: encoder outputs {self.teacher_feature_dim} '
              f'channels at strides {self.strides}, fp16 {fp16}')

    def train(self, mode: bool = True):
        # Frozen: stays in eval (BN statistics included) whatever the solver asks of it.
        return super().train(False)

    @torch.no_grad()
    def forward(self, images: torch.Tensor):
        with torch.autocast(device_type=images.device.type, dtype=torch.float16, enabled=self.fp16):
            feats = self.encoder(self.backbone(images))
        feats = [f.float() for f in feats]
        for stride, f in zip(self.strides, feats):
            if not torch.isfinite(f).all():
                raise FloatingPointError(f'[DEIMTeacher] non-finite stride-{stride} features for a '
                                         f'{tuple(images.shape)} batch')
        return dict(zip(self.strides, feats))
