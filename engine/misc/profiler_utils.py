"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import copy
from calflops import calculate_flops
from typing import Tuple

def stats(
    cfg,
    input_shape: Tuple=(1, 3, 640, 640), ) -> Tuple[str, str, int]:

    # The deploy model's encoder position embedding and decoder anchors are built for eval_spatial_size, so the probe
    # has to be that shape: a square probe crashes a landscape (H != W) frame before the first epoch.
    height, width = cfg.yaml_cfg['eval_spatial_size']
    input_shape = (1, 3, height, width)

    model_for_info = copy.deepcopy(cfg.model).deploy()

    flops, macs, _ = calculate_flops(model=model_for_info,
                                        input_shape=input_shape,
                                        output_as_string=True,
                                        output_precision=4,
                                        print_detailed=False)
    params = sum(p.numel() for p in model_for_info.parameters())
    del model_for_info

    return flops, macs, params
