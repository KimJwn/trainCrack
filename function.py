import csv
import math
import time
import os.path
import cv2
import numpy as np
from skimage.morphology import medial_axis
from tqdm import tqdm

PIXEL_MAX_VALUE = 255

# 디렉토리 수정
PATH = '/home/jovyan/DragonBall/vision'
IMG_PATH = PATH+'/data'
SAVE_DIR = PATH+'/results'

MIN_CIRCULAR_MASK_RADIUS_RANGE = 7
MAX_CIRCULAR_MASK_RADIUS_RANGE = 50 #원래 : 15

direction_set   = ['N', 'NW', 'W', 'SW', 'S', 'SE', 'E', 'NE']
positive_table  = ...
p_c_table  = ...
p_f_table  = ...
                # ['W', 'SW', 'S', 'SE', 'E', 'NE', 'N', 'NW']
negative_table  = ...
n_c_table  = ... # 진행 방향의 가까이의 음의 수직
n_f_table  = ... # 진행 방향의 멀리의  음의 수직
                # ['E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE']
neighbor_key = ...

import params as pm
color = pm.Color()


def find_max(list):
    block_max = []
    for block in list:
        block_max.append(max(block))
    return (block_max)


def direction_dictionary(row, col):
    return {
        'NW': [row - 1, col - 1 ],
        'N' : [row - 1, col     ],
        'NE': [row - 1, col + 1 ],
        'E' : [row,     col + 1 ],
        'SE': [row + 1, col + 1 ],
        'S' : [row + 1, col     ],
        'SW': [row + 1, col - 1 ],
        'W' : [row,     col - 1 ]
    }

#중심픽셀을 제외한 반지름이 7부터 15인 꽉찬 흰 원 만드는 함수
def circular_mask(radius):
    ...

    return mask, mask_area_pixel_num


def masking_circular_area(img, row, col, radius, mask):
    crack_area_pixel_num = np.sum(np.multiply(mask, img[row - radius: row + radius + 1, col - radius: col + radius + 1]))
    crack_area_pixel_num /= PIXEL_MAX_VALUE

    return crack_area_pixel_num


def write_csv(rxw_item):
    file = SAVE_DIR+'find_width_table.csv'

    if os.path.isfile(file):
        print("function : There is already a table to find crack width.", len(rxw_item))
    else :
        with open(SAVE_DIR+'find_width_table.csv', 'w', newline='') as f:
            makewrite = csv.writer(f)
            for value in rxw_item:
                makewrite.writerow(value)
    

def read_csv():
    total_rxw_item_list = []

    with open(SAVE_DIR + 'find_width_table.csv', 'r') as f:
        reader = csv.reader(f)
        for raw_item_list in reader:
            int_item_list = [int(raw_item) for raw_item in raw_item_list]
            total_rxw_item_list.append(int_item_list)
    return total_rxw_item_list


def search_width_in_table(total_item_list, radius, crack_area_pixel_num):

    radius_index = radius - MIN_CIRCULAR_MASK_RADIUS_RANGE
    # print(radius_index, len(total_item_list))
    if radius_index >= len(total_item_list) :
        return 0
    c = total_item_list[radius_index].copy()
    c.append(int(crack_area_pixel_num))
    d = sorted(c)
    crack_width = d.index(int(crack_area_pixel_num))

    return crack_width


# 
def closing_func(img):
    kernel = np.ones((19, 19), np.uint8)
    img_closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    result = cv2.subtract(img_closing, img)
    return result

def erode_func(img):
    kernel = np.ones((3, 3), np.uint8)
    img_closing = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
    result = cv2.subtract(img_closing, img)
    return result

def sharpening_func(img):
    kernel = np.array([[0, -1, 0],
                     [-1, 5, -1],
                     [0, -1, 0]])
    img = cv2.filter2D(img, -1, kernel)
    return img

def impacting_func(img, low = 200, high = 255):
    #int(input("최댓값 : "))
    height, width = img.shape
    ...
    return img

def opening_func(img):
    kernel = np.ones((15, 15), np.uint8)
    img_opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    result = cv2.subtract(img_opening, img)
    return result

def opening_closing_func(img):
    #kernel = np.ones((3, 3), np.uint8)
    kernel, _ = circular_mask(1)
    img_opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    img_opening_closing = cv2.morphologyEx(img_opening, cv2.MORPH_CLOSE, kernel)
    return img_opening_closing

def closing_opening_func(img):
    #kernel = np.ones((3, 3), np.uint8)
    kernel, _ = circular_mask(1)
    img_closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    img_closing_opening = cv2.morphologyEx(img_closing, cv2.MORPH_OPEN, kernel)
    return img_closing_opening


def otsu_func(img):
    ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    print(ret2)
    return ret2, th2


def threshold_otsu_func(img, th):
    # ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_OTSU)
    # th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, blk_size, C)
    ret2,th2 = cv2.threshold(img,th,255,cv2.THRESH_BINARY)
    print(ret2)
    return ret2, th2


def adaptive_gaussian_otsu_func(img):
    blk_size = 9        # 블럭 사이즈
    C = 5               # 차감 상수 
    th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, blk_size, C)
    return th3


def noise_reduction_func(img, kernel_r = 3, threshold=220):
    ...
    return averaged_img


def circle_noise_removal_using_packing_density_func(img):
    from math import pi
    threshold = 0.1 #0.12
    nlabels, img_labeled, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)

    for i in range(nlabels):
        ...
    return img


def combine_mask(img, mask):
    if img.shape != mask.shape:
        print(img.shape, mask.shape)
    if type(img) != type(mask) :
        print(type(img) , type(mask))
    return cv2.bitwise_and(img, mask)


def thinning_func(img):
    # Compute the medial axis (skeleton) and the distance transform
    ...
    return dist_on_skel


def boundary_func(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    #blank_image = np.zeros((734, 980, 1), np.uint8)
    blank_image = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)

    for i in range(len(contours)):
        ...
    return blank_image


def search_start_interval_point_direction_key(img):
    start_interval_point_direction_key_list = []
    # 몇번째 위치를 접근하고 있는 지를 알려주는 표시
            
    # 시작점을 구하려면 주변의 색과 비교해야 해서, 테두리 1픽셀을 검게 칠해진 이미지 불러옴
    for row in range(1, img.shape[0] - 1):
        for col in range(1, img.shape[1] - 1):
            if img[row, col] == 255:
                ...
                for i in range(-8, 0): #'NW'부터 다음 픽셀과 색 비교
                    if int(pixel_dict[neighbor_key[i]]) - int(pixel_dict[neighbor_key[i + 1]]) == 255:
                        flag += 1
                
                # 분기점(특징점)일때 점과 주변 흰색 점의 방향들을 리스트 start_interval_point_direction_key_list에 추가
                if flag == 1 or flag >= 3:
                    ...
    return start_interval_point_direction_key_list


def display_start_interval_point(img, start_interval_point_direction_key_list):
    ...

    return img


def search_edge_segment(img, start_interval_point_direction_key_list):
    total_segment_list = []
    total_chain_list = []

    # 특징점의 좌표[row,col]만 뽑아냄 (중복되는 좌표가 연속으로 존재할 수 있음)
    start_interval_point_list = list(map(lambda x: x[:2], start_interval_point_direction_key_list))
    #pdb.set_trace()

    #visited_map : [0]를 원소로 가지는 2차원 매트릭스(x*y)
    visited_map = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
    
    # tqdm: 진행상황 표시
    for row, col, key in tqdm(start_interval_point_direction_key_list):
        #pdb.set_trace()
        direction_dict = direction_dictionary(row, col)
        
        # 원소: (특징점 좌표, 이웃 좌표)
        segment_code = []
        
        #원소: (특징점 좌표, 이웃 방향)
        chain_code = []
        
        # 방문한 좌표라면, 다음 (특징점,방향)원소로 접근
        if visited_map[tuple(direction_dict[key])] == 1:
            continue
            
        # 처음 방문한 좌표라면
        else:
           ...
           ...
    return total_segment_list, total_chain_list


def display_edge_segment(img, total_segment_list):
    cv2.imshow('hi', img)
    cv2.waitKey()

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for i, block in enumerate(total_segment_list, 1):
        for [row, col] in block:
            ...

    cv2.imshow('hi', img)
    cv2.waitKey()

    return img


def crack_length_func(total_chain_list):
    total_length_list = []
    for chain_block in total_chain_list:
        crack_length = 0
        for code in chain_block:
            ...
        total_length_list.append(crack_length)
    return total_length_list
    
    

def direction_func(direction, LorR=-1):
    # print(direction)
    d = direction_set.index(direction)
    positive_d = positive_table[d]
    negative_d = negative_table[d]
    p_c = p_c_table[d]  # type: ignore
    p_f = p_f_table[d]
    n_c = n_c_table[d]
    n_f = n_f_table[d]

    if LorR == 0:   return positive_d, p_c, p_f
    elif LorR == 1: return negative_d, n_c, n_f
    else : return positive_d, negative_d, p_c, p_f, n_c, n_f

# 한 방향으로 연속되는 흰색 픽셀 수 구하기
def fill_color_until_black(img_BGR, img, img_th, start, direction, LorR=0, clr=[0,0,0]):
    row, col = (start[0],start[1])
    d_next, d_next_cl, d_next_fr = direction_func(direction, LorR=LorR)  # type: ignore
    
    while True :
        img_BGR[row, col] = clr
        direction_dict = direction_dictionary(row, col)
        next = direction_dict[d_next]       # 정중앙의 수직방향 다음 픽셀
        # next_c = direction_dict[d_next_cl]  # 가까이의 수직방향 다음 픽셀
        # next_f = direction_dict[d_next_fr]  # 더멀리의 수직방향 다음 픽셀

        # 0: 선분 진행 방향의 수직 균열 밖이면! : 검은색 픽셀이면
        ...

        # 1-0: 선분 진행 방향의 수직 균열 내부라면! (흰색 픽셀) : 다른 균열의 중심부일 때! - 너비 끝
        ...

        row, col = (next[0],next[1])
    # while 끝!!
    return img_BGR

# 한 방향으로 연속되는 흰색 픽셀 수 구하기
def finding_white_until_black(img, img_th, start, direction, LorR=0):
    width = 0
    row, col = (start[0],start[1])
    d_next, d_next_cl, d_next_fr = direction_func(direction, LorR=LorR)  # type: ignore
    
    while True :
        direction_dict = direction_dictionary(row, col)
        next = direction_dict[d_next]       # 정중앙의 수직방향 다음 픽셀
        next_c = direction_dict[d_next_cl]  # 가까이의 수직방향 다음 픽셀
        next_f = direction_dict[d_next_fr]  # 더멀리의 수직방향 다음 픽셀

        # 0: 선분 진행 방향의 수직 균열 밖이면! : 검은색 픽셀이면
        ...

        # 1-0: 선분 진행 방향의 수직 균열 내부라면! (흰색 픽셀) : 다른 균열의 중심부일 때! - 너비 끝
        ...

        width += 1
        row = next[0]
        col = next[1]
    # while 끝!!
    return width


# 너비 색상 표시 기법
def fill_crack_width_func(img, img_th, total_segment_list, total_chain_list, total_width_list):
    img_BGR = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    num_1st_lowest, num_2nd, num_3th, num_4th_highest = (0, 4, 5, 60)

    for j in range(len(total_width_list)):
        ...
    #
    return img_BGR

# 추가된 너비 측정 방식 re
def renewal_profiling_crack_width_func(img, img_th, total_segment_list, total_chain_list):
    max_width = 23  #20>1.3194938271604937, 23>1.5174179012345677
    total_width_list = []
    for j in range(len(total_chain_list)):
        segment_block = total_chain_list[j]
        center_pixel_block = total_segment_list[j]
        segment_width_block = []
        for i in range(0, len(segment_block)):
            ...
        #
        total_width_list.append(segment_width_block)
    #
    return total_width_list, img

# 추가된 너비 측정 방식 0 - new one
def profiling_crack_width_func_new(img, img_th, total_chain_list):
    total_width_list = []

    for segment_block in total_chain_list:
        ...
            
        #
        total_width_list.append(segment_width_block)
    #
    return total_width_list, img

# 추가된 너비 측정 방식 1 - old one
def profiling_crack_width_func(img, img_th, total_chain_list):
    ...
                
    return p_list, img_bgr


def adaptive_crack_width_func(img, total_segment_list):
    ...
    return total_width_list



def normal_crack_width_func(img, total_segment_list, radius=20):
   ...
    return total_width_list

