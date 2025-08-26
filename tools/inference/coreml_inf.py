"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import torch
import torchvision.transforms as T
import numpy as np
import coremltools as ct
from PIL import Image, ImageDraw
import cv2


def resize_with_aspect_ratio(image, size, interpolation=Image.BILINEAR):
    """Resizes an image while maintaining aspect ratio and pads it."""
    original_width, original_height = image.size
    ratio = min(size / original_width, size / original_height)
    new_width = int(original_width * ratio)
    new_height = int(original_height * ratio)
    image = image.resize((new_width, new_height), interpolation)

    # Create a new image with the desired size and paste the resized image onto it
    new_image = Image.new("RGB", (size, size))
    new_image.paste(image, ((size - new_width) // 2, (size - new_height) // 2))
    return new_image, ratio, (size - new_width) // 2, (size - new_height) // 2


def draw(images, labels, boxes, scores, ratios, paddings, thrh=0.4):
    result_images = []
    for i, im in enumerate(images):
        draw = ImageDraw.Draw(im)
        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]
        scr = scr[scr > thrh]

        ratio = ratios[i]
        pad_w, pad_h = paddings[i]

        for lbl, bb in zip(lab, box):
            # Adjust bounding boxes according to the resizing and padding
            bb = [
                (bb[0] - pad_w) / ratio,
                (bb[1] - pad_h) / ratio,
                (bb[2] - pad_w) / ratio,
                (bb[3] - pad_h) / ratio,
            ]
            draw.rectangle(bb, outline='red')
            draw.text((bb[0], bb[1]), text=str(lbl), fill='blue')

        result_images.append(im)
    return result_images


def process_image(model, im_pil):
    # Resize image while preserving aspect ratio
    resized_im_pil, ratio, pad_w, pad_h = resize_with_aspect_ratio(im_pil, 640)
    # Prepare original image size as a tuple of ints, not a numpy array
    #org_size = [im_pil.size[1], im_pil.size[0]]  # (height, width)

    # Prepare input for CoreML model
    input_dict = {}
    input_dict['images'] = resized_im_pil

    # NP ARRAY of shape [1, 2]
    w, h = im_pil.size
    input_dict['orig_target_sizes'] = np.array([[float(h), float(w)]])

    # Run inference with CoreML
    output = model.predict(input_dict)

    print('Got output')

    # Extract outputs - CoreML returns a dictionary
    # We need to determine the output format based on the model
    if isinstance(output, dict):
        # Try to extract labels, boxes, scores from the output dictionary
        if 'labels' in output:
            labels = output['labels']
        elif 'output0' in output:
            labels = output['output0']
        else:
            # Use the first output for labels
            labels = list(output.values())[0]
            
        if 'boxes' in output:
            boxes = output['boxes']
        elif 'output1' in output:
            boxes = output['output1']
        else:
            # Use the second output for boxes
            boxes = list(output.values())[1]
            
        if 'scores' in output:
            scores = output['scores']
        elif 'output2' in output:
            scores = output['output2']
        else:
            # Use the third output for scores
            scores = list(output.values())[2]
    else:
        # If output is not a dict, assume it's a list/tuple
        labels, boxes, scores = output

    print(f'labels: {labels}')
    print(f'boxes: {boxes}')
    print(f'scores: {scores}')

    result_images = draw(
        [im_pil], labels, boxes, scores,
        [ratio], [(pad_w, pad_h)]
    )
    result_images[0].save('coreml_result.jpg')
    print("Image processing complete. Result saved as 'coreml_result.jpg'.")



def main(args):
    """Main function."""
    # Load the CoreML model
    model = ct.models.MLModel(args.coreml)
    print(f"Loaded CoreML model: {args.coreml}")
    print(f"Model input description: {model.input_description}")
    print(f"Model output description: {model.output_description}")

    input_path = args.input

    try:
        # Try to open the input as an image
        im_pil = Image.open(input_path).convert('RGB')
        process_image(model, im_pil)
    except IOError:
        # Not an image, process as video
        raise ValueError("Input must be an image.")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--coreml', type=str, required=True, help='Path to the CoreML model file.')
    parser.add_argument('--input', type=str, required=True, help='Path to the input image or video file.')
    args = parser.parse_args()
    main(args)
