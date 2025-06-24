import argparse
import os

import cv2
import numpy as np

from estimator.qreader import QReader
import glob
from PIL import Image
import yaml
import re

def get_arguments():
    parser = argparse.ArgumentParser(description="detect qrcode position")
    parser.add_argument("rgb", type=str)
    parser.add_argument("--display", action="store_true", help="Display image (default: False)")
    return parser.parse_args()


def read_png_images(directory, display=False):
    png_files = glob.glob(os.path.join(directory, '*.png'))
    all_offset = {}
    for png_file in png_files:
        print(f'png file path: {png_file}')
        if '_code_pose' not in png_file or 'depth' in png_file:
            continue
        if display:
            image = Image.open(png_file)
            image.show()
            input("press to next")
            image.close()

        match = re.search(r'_code_pose_(left|right)_', png_file)

        if match:
            side = match.group(1)
            print("Extracted side:", side)
        else:
            print("No match found.")

        rgb = cv2.imread(png_file)
        result = qreader.detect_and_decode(image=rgb, return_bboxes=True)
        num_offset = calc_code_offset(result)
        if all_offset.get(num_offset[0]) is not None:
            all_offset[num_offset[0]][side] = {'x': float(num_offset[1]), 'y': float(num_offset[2])}
        else:
            all_offset[num_offset[0]] = {side: {'x': float(num_offset[1]), 'y': float(num_offset[2])}}
        print(all_offset)
    save_dict_to_yaml(all_offset, rgb_path + '/offset_result.yaml')

def calc_code_offset(result):
    '''this is for calc code offset'''
    pixel_model = 0.02 / 60
    standard_code_center = np.array([357, 210])
    two_points = result[0][0]
    left_up = np.array([two_points[0], two_points[1]])
    right_down = np.array([two_points[2], two_points[3]])
    code_center = (left_up + right_down) / 2
    offset_center = code_center - standard_code_center
    offset_x = round(offset_center[0] * pixel_model, 3)
    offset_y = round(offset_center[1] * pixel_model, 3)
    code_num = result[0][1][0]
    num_offset = (code_num, offset_x, offset_y)
    print(num_offset)
    return num_offset


def save_dict_to_yaml(dictionary, file_path):
    with open(file_path, 'w') as yaml_file:
        yaml.dump(dictionary, yaml_file)


if __name__ == "__main__":
    # args = get_arguments()
    # qreader = QReader()
    # rgb_path = args.rgb
    # assert os.path.exists(rgb_path)
    # rgb = cv2.imread(rgb_path)
    # result = qreader.detect_and_decode(image=rgb, return_bboxes=True)
    '''this is for calc code offset'''
    # pixel_model = 0.02 / 60
    # standard_code_center = np.array([357, 210])
    # two_points = result[0][0]
    # left_up = np.array([two_points[0], two_points[1]])
    # right_down = np.array([two_points[2], two_points[3]])
    # code_center = (left_up + right_down) / 2
    # offset_center = code_center - standard_code_center
    # offset_x = round(offset_center[0] * pixel_model, 3)
    # offset_y = round(offset_center[1] * pixel_model, 3)
    # code_num = result[0][1][0]
    # num_offset = (code_num, offset_x, offset_y)
    # print(result)
    # print(two_points)
    # print(code_num)
    # print(left_up)
    # print(right_down)
    # print(code_center)
    # print(offset_center)
    # print(offset_x)
    # print(offset_y)
    # print(num_offset)

    '''read image from file'''
    args = get_arguments()
    qreader = QReader()
    rgb_path = args.rgb
    assert os.path.exists(rgb_path)
    read_png_images(rgb_path, display=args.display)
