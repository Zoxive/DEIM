"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import argparse

from engine.misc import dist_utils
from engine.core import YAMLConfig, yaml_utils
from engine.solver import TASKS

debug=False

if debug:
    import torch
    def custom_repr(self):
        return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'
    original_repr = torch.Tensor.__repr__
    torch.Tensor.__repr__ = custom_repr

def main(args, ) -> None:
    """main
    """
    #dist_utils.setup_distributed(args.print_rank, args.print_method, seed=args.seed)
    # setup_distributed was the only caller of setup_seed, so with it commented out `--seed` seeded
    # nothing: runs before this fix were unseeded whatever seed they were given.
    if args.seed is not None:
        dist_utils.setup_seed(args.seed)

    assert not all([args.tuning, args.resume]), \
        'Only support from_scrach or resume or tuning at one time'


    update_dict = yaml_utils.parse_cli(args.update)
    update_dict.update({k: v for k, v in args.__dict__.items() \
        if k not in ['update', ] and v is not None})

    cfg = YAMLConfig(args.config, **update_dict)

    if args.resume or args.tuning:
        if 'HGNetv2' in cfg.yaml_cfg:
            cfg.yaml_cfg['HGNetv2']['pretrained'] = False

    print('cfg: ', cfg.__dict__)

    solver = TASKS[cfg.yaml_cfg['task']](cfg)

    if args.test_only:
        solver.val()
    else:
        solver.fit()

    dist_utils.cleanup()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # priority 0
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('-r', '--resume', type=str, help='resume from checkpoint')
    parser.add_argument('-t', '--tuning', type=str, help='tuning from checkpoint')
    parser.add_argument('-d', '--device', type=str, help='device',)
    parser.add_argument('--seed', type=int, help='exp reproducibility')
    # default=None on the flags below: main() merges every non-None arg over the YAML, so a
    # store_true default of False would silently turn `use_amp: True` in base/optimizer.yml back off.
    parser.add_argument('--use-amp', action='store_true', default=None, help='auto mixed precision training')
    parser.add_argument('--amp-dtype', type=str, choices=['float16', 'bfloat16'], help='autocast dtype (default float16)')
    parser.add_argument('--tf32', action='store_true', default=None, help='TF32 for the fp32 matmuls autocast leaves alone')
    parser.add_argument('--compile', action='store_true', default=None, help='torch.compile the model')
    parser.add_argument('--channels-last', action='store_true', default=None, help='channels-last weights and inputs')
    parser.add_argument('--cudnn-benchmark', action='store_true', default=None, help='cuDNN autotuning per input shape')
    parser.add_argument('--nonfinite-guard', action='store_true', default=None,
                        help='record, diagnose and skip a batch with non-finite predictions (engine/solver/nonfinite_guard.py)')
    parser.add_argument('--nonfinite-max-skips', type=int, help='raise after this many guarded batches (default 10)')
    parser.add_argument('--fake-quant', choices=['w4', 'w6', 'w8'],
                        help='weight-only QAT: conv and linear weights train on a per-output-channel grid of this many bits')
    parser.add_argument('--output-dir', type=str, help='output directoy')
    parser.add_argument('--summary-dir', type=str, help='tensorboard summry')
    parser.add_argument('--test-only', action='store_true', default=False,)

    # priority 1
    parser.add_argument('-u', '--update', nargs='+', help='update yaml config')

    # env
    parser.add_argument('--print-method', type=str, default='builtin', help='print method')
    parser.add_argument('--print-rank', type=int, default=0, help='print rank id')

    parser.add_argument('--local-rank', type=int, help='local rank id')
    args = parser.parse_args()

    main(args)
