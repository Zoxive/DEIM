"""RTMDet (tiny / s / m) in plain PyTorch: CSPNeXt backbone, CSPNeXt-PAFPN neck, SepBN head.

A port of MMDetection 3.3.0's RTMDet (Apache-2.0, https://github.com/open-mmlab/mmdetection,
`configs/rtmdet/`, `mmdet/models/{backbones/cspnext.py, necks/cspnext_pafpn.py,
dense_heads/rtmdet_head.py, layers/csp_layer.py}`), with no mmdet / mmcv / mmengine dependency.
Module names mirror mmdet's so its released COCO checkpoints load key for key
(converted to a state dict, a released checkpoint serves as a DEIM `-t` init).

The input contract is DEIM's, so every instrument and export path takes either model: RGB in [0, 1],
[N, 3, H, W], a squash to 640 at deploy. mmdet's RTMDet consumes BGR 0-255 normalised by BGR mean/std;
the conversion permutes the stem's input channels to RGB, and the model normalises with the RGB-ordered
constants below, which is exact (a channel permutation commutes with per-channel normalisation).

Every op is a convolution, BN, SiLU, max-pool, global average pool, hard-sigmoid, nearest upsample or
concat: nothing the LiteRT GPU delegate refuses. Decoding is `prior +- distance`, elementwise.

Two options exist for a mobile GPU: `act` (relu / hswish in
place of SiLU) and `strides=(16, 32)` (no 80x80 level; the neck becomes two-level). With `strides=(16, 32)` the
COCO checkpoint's two-level-specific tensors (bottom-up path, out convs, the head's per-level BN) do not line up and
are left at their initialisation by `train.py -t`.

`reg_max` > 0 makes the box head a distribution head (Generalized Focal Loss, Li et al. 2020; the dense form of
D-FINE's edge distributions): each edge is `reg_max + 1` logits over distances 0, 1, ..., reg_max in units of the
level's stride, and the distance is their softmax expectation. `heads()` still returns pixel distances, computed
in-graph with GPU-friendly ops only - per edge a slice of `reg_max + 1` channels, a softmax over channels (the last
axis once the graph is NHWC) and a fixed 1x1 convolution with weights 0..reg_max - so an exported graph's outputs and
its decode do not change. `forward()` also returns the logits (`pred_corners`) for the DFL term.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core import register

__all__ = ['RTMDet']

# (deepen_factor, widen_factor, neck channels in, neck/head channels, neck CSP blocks, exp_on_reg)
ARCHS = {
    'tiny': (0.167, 0.375, [96, 192, 384], 96, 1, False),
    's': (0.33, 0.5, [128, 256, 512], 128, 1, False),
    'm': (0.67, 0.75, [192, 384, 768], 192, 2, True),
}

ACTIVATIONS = {'silu': None, 'relu': nn.ReLU, 'hswish': nn.Hardswish}

# mmdet's data_preprocessor constants, reordered BGR -> RGB.
PIXEL_MEAN = (123.675, 116.28, 103.53)
PIXEL_STD = (58.395, 57.12, 57.375)


class ConvModule(nn.Module):
    """mmcv ConvModule as RTMDet uses it: conv (no bias) -> BN -> SiLU."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activate = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.activate(self.bn(self.conv(x)))


class DepthwiseSeparableConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.depthwise_conv = ConvModule(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.pointwise_conv = ConvModule(in_channels, out_channels, 1)

    def forward(self, x):
        return self.pointwise_conv(self.depthwise_conv(x))


class ChannelAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Hardsigmoid(inplace=True)

    def forward(self, x):
        # mmdet pools in fp32 under autocast; a 640-wide mean of SiLU activations is safe in fp16
        # only by luck, so keep it.
        with torch.autocast(device_type=x.device.type, enabled=False):
            out = self.global_avgpool(x.float())
        out = self.act(self.fc(out.to(x.dtype)))
        return x * out


class CSPNeXtBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion=0.5, add_identity=True, kernel_size=5):
        super().__init__()
        hidden = int(out_channels * expansion)
        self.conv1 = ConvModule(in_channels, hidden, 3, 1, 1)
        self.conv2 = DepthwiseSeparableConvModule(hidden, out_channels, kernel_size, 1, kernel_size // 2)
        self.add_identity = add_identity and in_channels == out_channels

    def forward(self, x):
        out = self.conv2(self.conv1(x))
        return out + x if self.add_identity else out


class CSPLayer(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio=0.5, num_blocks=1, add_identity=True,
                 channel_attention=False):
        super().__init__()
        mid = int(out_channels * expand_ratio)
        self.channel_attention = channel_attention
        self.main_conv = ConvModule(in_channels, mid, 1)
        self.short_conv = ConvModule(in_channels, mid, 1)
        self.final_conv = ConvModule(2 * mid, out_channels, 1)
        self.blocks = nn.Sequential(*[CSPNeXtBlock(mid, mid, 1.0, add_identity) for _ in range(num_blocks)])
        if channel_attention:
            self.attention = ChannelAttention(2 * mid)

    def forward(self, x):
        x_short = self.short_conv(x)
        x_main = self.blocks(self.main_conv(x))
        out = torch.cat((x_main, x_short), dim=1)
        if self.channel_attention:
            out = self.attention(out)
        return self.final_conv(out)


class SPPBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=(5, 9, 13)):
        super().__init__()
        mid = in_channels // 2
        self.conv1 = ConvModule(in_channels, mid, 1)
        self.poolings = nn.ModuleList([nn.MaxPool2d(k, 1, k // 2) for k in kernel_sizes])
        self.conv2 = ConvModule(mid * (len(kernel_sizes) + 1), out_channels, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [pool(x) for pool in self.poolings], dim=1)
        return self.conv2(x)


class CSPNeXt(nn.Module):
    """P5 arch: in, out, blocks, add_identity, use_spp."""
    arch_settings = [[64, 128, 3, True, False], [128, 256, 6, True, False],
                     [256, 512, 6, True, False], [512, 1024, 3, False, True]]

    def __init__(self, deepen_factor, widen_factor, expand_ratio=0.5, out_indices=(2, 3, 4)):
        super().__init__()
        self.out_indices = out_indices
        stem = int(self.arch_settings[0][0] * widen_factor)
        self.stem = nn.Sequential(
            ConvModule(3, stem // 2, 3, 2, 1),
            ConvModule(stem // 2, stem // 2, 3, 1, 1),
            ConvModule(stem // 2, stem, 3, 1, 1))
        self.layers = ['stem']
        for i, (cin, cout, blocks, add_identity, use_spp) in enumerate(self.arch_settings):
            cin, cout = int(cin * widen_factor), int(cout * widen_factor)
            blocks = max(round(blocks * deepen_factor), 1)
            stage = [ConvModule(cin, cout, 3, 2, 1)]
            if use_spp:
                stage.append(SPPBottleneck(cout, cout))
            stage.append(CSPLayer(cout, cout, expand_ratio, blocks, add_identity, channel_attention=True))
            self.add_module(f'stage{i + 1}', nn.Sequential(*stage))
            self.layers.append(f'stage{i + 1}')

    def forward(self, x):
        outs = []
        for i, name in enumerate(self.layers):
            x = getattr(self, name)(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)


class CSPNeXtPAFPN(nn.Module):
    def __init__(self, in_channels, out_channels, num_csp_blocks, expand_ratio=0.5):
        super().__init__()
        self.in_channels = in_channels
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.reduce_layers = nn.ModuleList()
        self.top_down_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1, 0, -1):
            self.reduce_layers.append(ConvModule(in_channels[idx], in_channels[idx - 1], 1))
            self.top_down_blocks.append(CSPLayer(in_channels[idx - 1] * 2, in_channels[idx - 1], expand_ratio,
                                                 num_csp_blocks, add_identity=False))
        self.downsamples = nn.ModuleList()
        self.bottom_up_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1):
            self.downsamples.append(ConvModule(in_channels[idx], in_channels[idx], 3, 2, 1))
            self.bottom_up_blocks.append(CSPLayer(in_channels[idx] * 2, in_channels[idx + 1], expand_ratio,
                                                  num_csp_blocks, add_identity=False))
        self.out_convs = nn.ModuleList([ConvModule(c, out_channels, 3, 1, 1) for c in in_channels])

    def forward(self, inputs):
        n = len(self.in_channels)
        inner_outs = [inputs[-1]]
        for idx in range(n - 1, 0, -1):
            feat_high = self.reduce_layers[n - 1 - idx](inner_outs[0])
            inner_outs[0] = feat_high
            inner_outs.insert(0, self.top_down_blocks[n - 1 - idx](
                torch.cat([self.upsample(feat_high), inputs[idx - 1]], 1)))
        outs = [inner_outs[0]]
        for idx in range(n - 1):
            down = self.downsamples[idx](outs[-1])
            outs.append(self.bottom_up_blocks[idx](torch.cat([down, inner_outs[idx + 1]], 1)))
        return tuple(conv(out) for conv, out in zip(self.out_convs, outs))


class RTMDetSepBNHead(nn.Module):
    """Per-level BN, convolutions shared across levels (`share_conv=True`), 1x1 predictors."""

    def __init__(self, num_classes, in_channels, feat_channels, strides=(8, 16, 32), stacked_convs=2,
                 exp_on_reg=False, reg_max=0):
        super().__init__()
        if reg_max and exp_on_reg:
            raise ValueError('a distribution box head (reg_max > 0) has no exp_on_reg: distances are expectations')
        self.num_classes = num_classes
        self.strides = strides
        self.exp_on_reg = exp_on_reg
        self.reg_max = reg_max
        reg_channels = 4 * (reg_max + 1) if reg_max else 4
        if reg_max:
            self.register_buffer('project', torch.arange(reg_max + 1, dtype=torch.float32).view(1, -1, 1, 1),
                                 persistent=False)
        self.cls_convs, self.reg_convs = nn.ModuleList(), nn.ModuleList()
        self.rtm_cls, self.rtm_reg = nn.ModuleList(), nn.ModuleList()
        for _ in strides:
            self.cls_convs.append(nn.ModuleList(
                [ConvModule(in_channels if i == 0 else feat_channels, feat_channels, 3, 1, 1)
                 for i in range(stacked_convs)]))
            self.reg_convs.append(nn.ModuleList(
                [ConvModule(in_channels if i == 0 else feat_channels, feat_channels, 3, 1, 1)
                 for i in range(stacked_convs)]))
            self.rtm_cls.append(nn.Conv2d(feat_channels, num_classes, 1))
            self.rtm_reg.append(nn.Conv2d(feat_channels, reg_channels, 1))
        for n in range(len(strides)):
            for i in range(stacked_convs):
                self.cls_convs[n][i].conv = self.cls_convs[0][i].conv
                self.reg_convs[n][i].conv = self.reg_convs[0][i].conv
        self._init_weights()

    def _init_weights(self):
        """mmdet's init: N(0, 0.01) convolutions, BN at 1/0, class bias at prior 0.01."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        bias_cls = -torch.log(torch.tensor((1 - 0.01) / 0.01)).item()
        for cls in self.rtm_cls:
            nn.init.constant_(cls.bias, bias_cls)

    def integral(self, corners):
        """[B, 4 * (reg_max + 1), H, W] edge logits -> [B, 4, H, W] distances in stride units, phone-safe: a channel
        slice per edge, a softmax over channels, a fixed 1x1 convolution (no 5-D reshape, no reduction op)."""
        return torch.cat([F.conv2d(edge.softmax(dim=1), self.project.to(edge.dtype))
                          for edge in torch.split(corners, self.reg_max + 1, dim=1)], 1)

    def forward(self, feats, return_corners=False):
        cls_scores, bbox_dists, corners = [], [], []
        for idx, (x, stride) in enumerate(zip(feats, self.strides)):
            cls_feat, reg_feat = x, x
            for layer in self.cls_convs[idx]:
                cls_feat = layer(cls_feat)
            for layer in self.reg_convs[idx]:
                reg_feat = layer(reg_feat)
            reg = self.rtm_reg[idx](reg_feat)
            if self.reg_max:
                corners.append(reg)
                reg = self.integral(reg)
            elif self.exp_on_reg:
                reg = reg.exp()
            cls_scores.append(self.rtm_cls[idx](cls_feat))
            bbox_dists.append(reg * stride)
        return (cls_scores, bbox_dists, corners) if return_corners else (cls_scores, bbox_dists)


def grid_priors(featmap_sizes, strides, device, dtype=torch.float32):
    """mmdet MlvlPointGenerator(offset=0): (x, y, stride, stride) per location, row-major, levels concatenated."""
    priors = []
    for (h, w), stride in zip(featmap_sizes, strides):
        ys, xs = torch.meshgrid(torch.arange(h, device=device, dtype=dtype) * stride,
                                torch.arange(w, device=device, dtype=dtype) * stride, indexing='ij')
        s = torch.full_like(xs, stride)
        priors.append(torch.stack([xs, ys, s, s], -1).reshape(-1, 4))
    return torch.cat(priors)


def flatten_levels(maps):
    """[B, C, H, W] per level -> [B, sum(HW), C]."""
    return torch.cat([m.permute(0, 2, 3, 1).reshape(m.shape[0], -1, m.shape[1]) for m in maps], 1)


def distance2bbox(points, distance):
    return torch.stack([points[..., 0] - distance[..., 0], points[..., 1] - distance[..., 1],
                        points[..., 0] + distance[..., 2], points[..., 1] + distance[..., 3]], -1)


@register()
class RTMDet(nn.Module):
    """RGB [0, 1] in; per-level logits and pixel distances out, plus the flat decode the trainer and
    postprocessor share: `pred_logits` [B, N, C], `pred_boxes` [B, N, 4] xyxy in input pixels."""
    __share__ = ['num_classes']

    def __init__(self, arch='s', num_classes=80, act='silu', strides=(8, 16, 32), distill_dim=0, reg_max=0):
        super().__init__()
        if arch not in ARCHS:
            raise ValueError(f'RTMDet arch {arch!r} is not one of {sorted(ARCHS)}')
        if act not in ACTIVATIONS:
            raise ValueError(f'RTMDet act {act!r} is not one of {sorted(ACTIVATIONS)}')
        strides = tuple(strides)
        if strides not in ((8, 16, 32), (16, 32)):
            raise ValueError(f'RTMDet strides {strides} must be (8, 16, 32) or (16, 32)')
        deepen, widen, neck_in, channels, neck_blocks, exp_on_reg = ARCHS[arch]
        self.arch, self.act, self.strides = arch, act, strides
        self.num_classes = num_classes
        levels = [(8, 16, 32).index(s) for s in strides]
        self.backbone = CSPNeXt(deepen, widen, out_indices=tuple(2 + i for i in levels))
        self.neck = CSPNeXtPAFPN([neck_in[i] for i in levels], channels, neck_blocks)
        if reg_max < 0:
            raise ValueError(f'RTMDet reg_max {reg_max} must be 0 (distance head) or positive (distribution head)')
        self.reg_max = reg_max
        self.bbox_head = RTMDetSepBNHead(num_classes, channels, channels, strides=strides,
                                         exp_on_reg=exp_on_reg and not reg_max, reg_max=reg_max)
        # Every SiLU is a LOGISTIC + MUL pair on the phone's GPU; ReLU folds into the convolution and HardSwish is
        # one kernel. Swapped after construction so the module names, and so COCO checkpoint keys, stay the same.
        if ACTIVATIONS[act] is not None:
            for parent in self.modules():
                for name, child in parent.named_children():
                    if isinstance(child, nn.SiLU):
                        setattr(parent, name, ACTIVATIONS[act](inplace=True))
        # Feature distillation (engine/rtmdet/distill.py): a 1x1 projection per level onto the teacher's width,
        # used only when training is handed teacher features. Inference and every export skip it.
        self.distill_proj = (nn.ModuleList([nn.Conv2d(channels, distill_dim, 1) for _ in strides])
                             if distill_dim else None)
        self.register_buffer('pixel_mean', torch.tensor(PIXEL_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer('pixel_std', torch.tensor(PIXEL_STD).view(1, 3, 1, 1), persistent=False)

    def heads(self, images, return_feats=False, return_corners=False):
        """The raw graph a phone runs: per-level class logits and box distances in input pixels (plus, on request,
        a distribution head's per-level edge logits)."""
        x = (images * 255.0 - self.pixel_mean) / self.pixel_std
        feats = self.neck(self.backbone(x))
        out = self.bbox_head(feats, return_corners=return_corners)
        return (out, feats) if return_feats else out

    def forward(self, images, targets=None, teacher_encoder_output=None, **kwargs):
        distill = self.training and teacher_encoder_output is not None
        if distill and self.distill_proj is None:
            raise ValueError('a teacher_model is configured but RTMDet has no distill_dim to project onto it')
        (cls_scores, bbox_dists, corners), feats = self.heads(images, return_feats=True, return_corners=True)
        sizes = [tuple(c.shape[-2:]) for c in cls_scores]
        priors = grid_priors(sizes, self.bbox_head.strides, images.device)
        logits = flatten_levels(cls_scores)
        if self.reg_max:
            # The expectation again in fp32 off the same logits (the in-graph one ran under autocast), so the box
            # the GIoU term sees and the distribution the DFL term sees are one prediction.
            corners = flatten_levels(corners).float().view(logits.shape[0], -1, 4, self.reg_max + 1)
            bins = torch.arange(self.reg_max + 1, device=corners.device, dtype=corners.dtype)
            dists = (corners.softmax(-1) * bins).sum(-1) * priors[None, :, 2:3]
        else:
            dists = flatten_levels(bbox_dists).float()
        out = {
            'pred_logits': logits,
            'pred_boxes': distance2bbox(priors[None, :, :2], dists),
            'priors': priors,
            'input_hw': images.shape[-2:],
        }
        if self.reg_max:
            out['pred_corners'] = corners
        if distill:
            out['distill_pairs'] = [(proj(f), teacher_encoder_output[s])
                                    for proj, f, s in zip(self.distill_proj, feats, self.bbox_head.strides)]
        return out

    def deploy(self):
        self.eval()
        return self
