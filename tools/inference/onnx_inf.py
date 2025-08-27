"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import torch
import torchvision.transforms as T
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont
import cv2

try:
    import coremltools as ct
except ImportError:
    print('coremltools is not installed')


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
    font = ImageFont.load_default(size=20)
    for i, im in enumerate(images):
        draw = ImageDraw.Draw(im)
        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]
        scr = scr[scr > thrh]

        ratio = ratios[i]
        pad_w, pad_h = paddings[i]

        for lbl, bb, score in zip(lab, box, scr):
            # Adjust bounding boxes according to the resizing and padding
            bb = [
                (bb[0] - pad_w) / ratio,
                (bb[1] - pad_h) / ratio,
                (bb[2] - pad_w) / ratio,
                (bb[3] - pad_h) / ratio,
            ]
            draw.rectangle(bb, outline='red', width=3)
            draw.text((bb[0], bb[1]), text=str(f"{int(lbl)} | {score * 100:.2f}"), fill='blue', font=font)
            print(f'lbl: {lbl.item()}, bb: {bb}, score: {score}')

        result_images.append(im)
    return result_images


def process_image(im_pil):
    # Resize image while preserving aspect ratio
    resized_im_pil, ratio, pad_w, pad_h = resize_with_aspect_ratio(im_pil, 640)
    orig_size = [[resized_im_pil.size[1], resized_im_pil.size[0]]]

    print('Image original size: ', resized_im_pil.size[1], resized_im_pil.size[0])
    print('Ratio: ', ratio)
    print('Pad width: ', pad_w)
    print('Pad height: ', pad_h)

    return resized_im_pil, orig_size, ratio, pad_w, pad_h

    # transforms = T.Compose([
    #     T.ToTensor(),
    # ])
    # im_data = transforms(resized_im_pil).unsqueeze(0)


    # output = sess.run(
    #     output_names=None,
    #     input_feed={'images': im_data.numpy(), "orig_target_sizes": orig_size.numpy()}
    # )

    # labels, boxes, scores = output

    # result_images = draw(
    #     [im_pil], labels, boxes, scores,
    #     [ratio], [(pad_w, pad_h)]
    # )
    # 

def main(args):
    """Main function."""
    # Load the ONNX model
    model_path = args.model
    if model_path.endswith('.onnx'):
        model = OnnxModel(model_path)
    else:
        model = CoremlModel(model_path)

    model_type = model_path.split('.')[-1]

    im_pil = Image.open(args.input).convert('RGB')
    
    result_images = model.process_image(im_pil)
    img_name = f'result_{model_type}.jpg'
    result_images[0].save(img_name)
    print(f"Image processing complete. Result saved as '{img_name}'.")


class OnnxModel:
    def __init__(self, model_path):
        self.sess = ort.InferenceSession(model_path)

    def process_image(self, im_pil: Image.Image):
        resized_im_pil, orig_size, ratio, pad_w, pad_h = process_image(im_pil)

        transforms = T.Compose([
            T.ToTensor(),
        ])
        im_data = transforms(resized_im_pil).unsqueeze(0)

        output = self.sess.run(
            output_names=None,
            input_feed={'images': im_data.numpy(), "orig_target_sizes": np.array(orig_size)}
        )

        labels, boxes, scores = output

        result_images = draw(
            [im_pil], labels, boxes, scores,
            [ratio], [(pad_w, pad_h)]
        )
        return result_images

class CoremlModel:
    def __init__(self, model_path):
        self.model = ct.models.MLModel(model_path)

    def process_image(self, im_pil: Image.Image):
        resized_im_pil, orig_size, ratio, pad_w, pad_h = process_image(im_pil)

        input_dict = {}
        input_dict['images'] = resized_im_pil

        # value type not convertible hax
        orig_size = np.array([[float(orig_size[0][0]), float(orig_size[0][1])]])
        input_dict['orig_target_sizes'] = orig_size

        output = self.model.predict(input_dict)

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

        # print(f'labels: {labels}')
        # print(f'boxes: {boxes}')
        # print(f'scores: {scores}')

        result_images = draw(
            [im_pil], labels, boxes, scores,
            [ratio], [(pad_w, pad_h)]
        )
        return result_images


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to the ONNX or CoreML model file.')
    parser.add_argument('--input', type=str, required=True, help='Path to the input image or video file.')
    args = parser.parse_args()
    main(args)
