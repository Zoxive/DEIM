"""
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
"""

import torch.nn as nn
from ..core import register


__all__ = ['DEIM', ]


@register()
class DEIM(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', ]

    def __init__(self, \
        backbone: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
    ):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder

    def forward(self, x, targets=None, teacher_encoder_output=None):
        x = self.backbone(x)
        x = self.encoder(x)
        # With distillation on (HybridEncoder.distill_teacher_dim > 0) the encoder returns
        # (features, projected F5) in training; the criterion's `distill` loss reads both maps.
        student_distill_output = None
        if isinstance(x, tuple):
            x, student_distill_output = x
        x = self.decoder(x, targets)

        if self.training and student_distill_output is not None and teacher_encoder_output is not None:
            x['student_distill_output'] = student_distill_output
            x['teacher_encoder_output'] = teacher_encoder_output

        return x

    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self
