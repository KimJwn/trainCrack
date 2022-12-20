import cv2
import pdb
import os
import argparse as arg
import time
import numpy as np
import params as pm
import table
from PIL import Image

from dis import dis
from tqdm import tqdm
from importlib.resources import Package
from skimage.morphology import medial_axis
from function import *

# 디렉토리를 로컬 폴더 상황에 맞게 변경 요망
FPATH = __file__
PATH = FPATH[:-3]
IMG_PATH = PATH+'/org_img/'
MASK_PATH = PATH+'/deep_mask/'
SAVE_DIR = PATH+'/results/'

'''API 확인 후 조치 예정'''
parser = arg.ArgumentParser()
parser.add_argument('--img_dir', default=IMG_PATH, help="Directory containing the data")
parser.add_argument('--mask_dir', default=MASK_PATH, help="Directory containing the data")
parser.add_argument('--save_dir', default=SAVE_DIR, help="Directory in which to store results")
parser.add_argument('--mode', default='normal', help="write debug for debug mode")
parser.add_argument('--img_name', default='GQ2_05_2.jpg', help="One image that we want")
parser.add_argument('--width_func', default='adaptive', help="which function to use to calculate crack width")

# 카메라 렌즈 왜곡 보정 파라미터
# parser.add_argument('--FL', default=4.8)
# parser.add_argument('--WD', default=0.45)
# parser.add_argument('--SIAH', default=14.0)
# parser.add_argument('--SRPH', default=9248)

# Galaxy quantum2 => yj
# parser.add_argument('--FL', default=5.2)
# parser.add_argument('--WD', default=1.0)
# parser.add_argument('--SIAH', default=7.4)
# parser.add_argument('--SRPH', default=4624)

#sony 파라미터
parser.add_argument('--FL', default=12.0)
parser.add_argument('--WD', default=1.0)
parser.add_argument('--SIAH', default=6.287)
parser.add_argument('--SRPH', default=4050)

# 입력받은 인자값을 args에 저장 (type: namespace)
args = parser.parse_args()

args.FL = float(args.FL)
args.WD = float(args.WD)
args.SIAH = float(args.SIAH)
args.SRPH = float(args.SRPH)

def main4():
    table.make_table()
    ...
    

if __name__ == '__main__':
    main4()
