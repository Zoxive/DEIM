"""
A frozen DINOv3 teacher for RT-DETRv4-style feature distillation (https://arxiv.org/abs/2510.25257).

Ported from RT-DETRv4's `engine/rtv4/dinov3_teacher.py` (Apache-2.0), with one change: the weights load
through Hugging Face `transformers` from the local HF cache, not through a `torch.hub` checkout of the
DINOv3 repo. The `p3_12` env's transformers (4.55.2) predates DINOv3, and calflops imports transformers
as soon as `engine` loads, so the newer one has to be on PYTHONPATH for the whole process:
`pip install --no-deps --target <dir> transformers==4.56.2 tokenizers==0.22.2`, with that directory on
PYTHONPATH for any config that names a teacher_model.

Used in training only. The teacher sees the student's batch - 0-1 RGB at 640 - normalises it with the
ImageNet mean/std and 2x2 average-pools it to 320, so a patch-16 ViT gives 20 x 20 tokens, the size of
the AIFI output (F5, stride 32) the student's projector is pulled toward.
"""

import torch
import torch.nn as nn

from ..core import register


__all__ = ['DINOv3Teacher', ]


@register()
class DINOv3Teacher(nn.Module):
    def __init__(self,
                 model_id: str = 'facebook/dinov3-vitb16-pretrain-lvd1689m',
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225)):
        super().__init__()
        import transformers
        from transformers import AutoModel
        print(f'[DINOv3Teacher] transformers {transformers.__version__} from {transformers.__file__}')
        if not hasattr(transformers, 'DINOv3ViTModel'):
            raise ImportError(f'transformers {transformers.__version__} has no DINOv3 (needs >= 4.56): put '
                              f'a transformers>=4.56 install on PYTHONPATH (see this module\'s docstring)')

        self.model = AutoModel.from_pretrained(model_id)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        # The CLS token and the register tokens come before the patch tokens in last_hidden_state.
        self.num_prefix_tokens = 1 + self.model.config.num_register_tokens
        self.teacher_feature_dim = self.model.config.hidden_size
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1), persistent=False)
        self.avgpool_2x2 = nn.AvgPool2d(kernel_size=2, stride=2)
        print(f'[DINOv3Teacher] {model_id}: dim {self.teacher_feature_dim}, '
              f'{self.num_prefix_tokens} prefix tokens dropped')

    def train(self, mode: bool = True):
        # Frozen: stays in eval whatever the solver asks of it.
        return super().train(False)

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.avgpool_2x2((images - self.mean) / self.std)
        tokens = self.model(pixel_values=x).last_hidden_state[:, self.num_prefix_tokens:]
        B, N, C = tokens.shape
        h, w = x.shape[2] // self.model.config.patch_size, x.shape[3] // self.model.config.patch_size
        if h * w != N:
            raise ValueError(f'[DINOv3Teacher] {N} patch tokens for a {x.shape[2]}x{x.shape[3]} input '
                             f'is not {h}x{w}')
        return tokens.permute(0, 2, 1).reshape(B, C, h, w).detach()
