"""
D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))

import torch
import torch.nn as nn
import coremltools as ct

from engine.core import YAMLConfig


def main(args):
    """main
    """
    cfg = YAMLConfig(args.config, resume=args.resume)

    if 'HGNetv2' in cfg.yaml_cfg:
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']

        # NOTE load train mode state -> convert to deploy mode
        cfg.model.load_state_dict(state)

    else:
        # raise AttributeError('Only support resume to load model.state_dict by now.')
        print('not load model.state_dict, use default init state dict...')

    class Model(nn.Module):
        def __init__(self, ) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    model = Model()
    model.eval()

    # for name, module in model.named_modules():
    #     if isinstance(module, torch.nn.Linear):
    #         print(name, module.weight.shape)

    # Create example inputs for tracing
    data = torch.rand(1, 3, 640, 640)
    size = torch.tensor([[640, 640], [640, 640]])

    # Trace the model
    traced_model = torch.jit.trace(model, (data, size), optimize=True)
    #traced_model = torch.jit.script(model)

    # print('Traced model:')
    #print(traced_model.graph)
    #exit()

    # Convert to CoreML
    output_file = args.resume.replace('.pth', '.mlpackage') if args.resume else 'model.mlpackage'
    
    # Define input specifications
    input_specs = [
        ct.TensorType(name="images", shape=data.shape, dtype=np.float32),
        ct.TensorType(name="orig_target_sizes", shape=size.shape, dtype=np.float32)
    ]
    
    #Convert to CoreML
    coreml_model = ct.convert(
        traced_model,
        inputs=input_specs,
        minimum_deployment_target=ct.target.iOS16,
        compute_units=ct.ComputeUnit.ALL,
        convert_to="mlprogram"
    )

    # dynamic_axes = {
    #     'images': {0: 'N', },
    #     'orig_target_sizes': {0: 'N'}
    # }

    # torch.onnx.export(
    #     model,
    #     (data, size),
    #     output_file,
    #     input_names=['images', 'orig_target_sizes'],
    #     output_names=['labels', 'boxes', 'scores'],
    #     dynamic_axes=dynamic_axes,
    #     opset_version=16,
    #     verbose=False,
    #     do_constant_folding=True,
    # )
    # coreml_model = ct.converters.onnx.convert("model.onnx")
    
    # Save the CoreML model
    coreml_model.save(output_file)
    print(f'CoreML model exported to: {output_file}')

    if args.check:
        # Load and validate the CoreML model
        loaded_model = ct.models.MLModel(output_file)
        print('CoreML model validation completed successfully...')

if __name__ == '__main__':

    import argparse
    import numpy as np
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', default='configs/dfine/dfine_hgnetv2_l_coco.yml', type=str, )
    parser.add_argument('--resume', '-r', type=str, )
    parser.add_argument('--check',  action='store_true', default=True,)
    args = parser.parse_args()
    main(args)
