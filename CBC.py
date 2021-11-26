import os
import shutil
import glob

import cv2
import numpy as np
import time
import sys
import copy
import argparse
from sklearn.cluster import KMeans
from sklearn import preprocessing

import labelimg_xml
import get_roi

debug_switch = 0
scope_switch = 0
current_filename = ''
morph_kernel = 11
morph_iteration = 1
magnification = 0

# parameters
parser = argparse.ArgumentParser(description='Complete Blood Count with OpenCV')
parser.add_argument('-si', '--save-images', action='store_true', help='save output images')
parser.add_argument('-t', '--show-running-time', action='store_true', help='show process running time')
parser.add_argument('-rs', '--remove-scope', action='store_true', help='image has scope')
parser.add_argument('-m', '--magnification', default=1000, action='store', help='set manual magnification')
args = parser.parse_args()

specific_magnification = False
if args.magnification is not None:
    specific_magnification = True
    magnification = args.magnification
    
    
# paths
INPUT_PATH = os.path.join(os.getcwd(), 'input')
OUTPUT_PATH = os.path.join(os.getcwd(), 'output')
XML_PATH = os.path.join(OUTPUT_PATH, 'xml')
THRESHOLD_PATH = os.path.join(OUTPUT_PATH, 'threshold')
LABEL_PATH = os.path.join(OUTPUT_PATH, 'labeled')
BBOX_PATH = os.path.join(OUTPUT_PATH, 'bbox')
CLASSES_PATH = os.path.join(OUTPUT_PATH, 'bbox', 'classes')

if os.path.exists(OUTPUT_PATH):
    shutil.rmtree(OUTPUT_PATH)
os.makedirs(XML_PATH)
if args.save_images:
    os.makedirs(THRESHOLD_PATH)
    os.makedirs(LABEL_PATH)
    os.makedirs(CLASSES_PATH)
    os.makedirs(BBOX_PATH)


def error(msg):
    print(msg)
    sys.exit(0)


class Clock:
    def __init__(self, title):
        self.timestamp = []
        self.sum = 0
        self.avg = 0
        self.max = 0
        self.min = 0
        self.title = title

    def start(self):
        self.timestamp.append(time.time())

    def stop(self):
        self.timestamp[-1] = time.time() - self.timestamp[-1]

    def calulate_runningtime(self):
        self.max = max(self.timestamp)
        self.min = min(self.timestamp)

        for item in self.timestamp:
            self.sum += item
        self.avg = self.sum / len(self.timestamp)

        return [self.min, self.max, self.sum, self.avg]


clk_total = Clock('Whole Process')
clk_post_process = Clock('Post-process')
clk_labeling = Clock('Labeling')
clk_organize_bbox = Clock('Organizing bbox')


def threshold(src):
    l,a,b = cv2.split(cv2.cvtColor(src, cv2.COLOR_BGR2LAB))
    threshed = cv2.threshold(a, 0, 255, cv2.THRESH_OTSU)
    cnts = cv2.findContours(threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        flood_filled = cv2.drawContours(threshed, [c], 0, (255, 255, 255), -1)
    morph_open = cv2.morphologyEx(flood_filled, cv2.MORPH_OPEN, np.ones((5, 5)), iterations=2)
    return morph_open


def remove_boarding_bbox(labels, stats, edge_error):
    height, width = labels.shape()
    for idx, box in enumerate(stats[:]):
        lboard = box[0] < edge_error
        rboard = box[0] + box[2] > height - edge_error
        tboard = box[1] < edge_error
        bboard = box[1] + box[3] > width - edge_error
        if lboard or rboard or tboard or bboard:
            labels[labels == idx] = 0
            stats.remove(box)
    return labels, stats


def set_class_norm(stats):
    sizes = stats[:,4]
    dimension = 1
    num_cluster = 3
    min_max = [min(sizes), max(sizes)]
    quantile = (min_max[1]-min_max[0])/num_cluster/2
    clst_mu = np.array([quantile*(2*i+1) + min_max[0] for i in range(num_cluster)])
    distance = (sizes - clst_mu.reshape(num_cluster, 1, dimension))
    distance = distance * distance
    distance = distance.sum(axis=2).T
    cluster = np.where(distance==distance.min(axis=1).reshape((len(sizes),1)))[1]
    clst_mu_old = clst_mu.copy()
    for i in range(num_cluster):
        clst_mu[i] = np.mean(sizes[np.where(cluster==i)[0]], axis=0)
    clst_mu = clst_mu_old*np.isnan(clst_mu) + np.nan_to_num(clst_mu)

    maxEpoch = 10

    for epoch in range(maxEpoch):
        distance = (sizes - clst_mu.reshape(num_cluster, 1, dimension))
        distance = distance * distance
        distance = distance.sum(axis=2).T
        cluster = np.where(distance == distance.min(axis=1).reshape((len(sizes), 1)))[1]
        clst_mu_old = clst_mu.copy()
        for i in range(num_cluster):
            clst_mu[i] = np.mean(sizes[np.where(cluster == i)[0]], axis=0)
        clst_mu = clst_mu_old * np.isnan(clst_mu) + np.nan_to_num(clst_mu)


def classify_boxes(img, stats):
    for box in stats:
        mini_img = img[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
        box_size = box[4]
        if box_size >



file_list = glob.glob(INPUT_PATH + '/*.jpg')
if len(file_list) == 0:
    error("Error: No input files found")
for idx, file in enumerate(file_list):
    file_id = file.split(".jpg", 1)[0]
    img = cv2.imread(os.path.join(INPUT_PATH, file))
    threshed = threshold(img)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(img)
    labels, stats = remove_boarding_bbox(labels, stats, 1)
    stats = classify_boxes(img, stats)
    plt_bboxes = find_plt(img, labels)


def post_process(img):
    """
    이미지 전처리
    1. hue, value로 adaptive theshold하여, 둘다 만족하는 픽셀만 취함
    2. flood fill로 안쪽 채움
    3. morphology operation(open)
    """
    global clk_post_process
    print('post_process', end='', flush=True)
    clk_post_process.start()

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    threshold_block_size = len(v)//3 + (len(v)//3+1)%2
    threshed_value = cv2.adaptiveThreshold(v, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, threshold_block_size, 0)
    threshed_hue = cv2.adaptiveThreshold(h, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, threshold_block_size, 0)
    threshed = threshed_value
    # threshed = cv2.bitwise_and(threshed_hue, threshed_value)

    if scope_switch == 1:
        scope = get_roi.find_roi(img, 'circle')
        threshed = get_roi.process_roi(cv2.cvtColor(threshed, cv2.COLOR_GRAY2BGR), scope)
        threshed = cv2.cvtColor(threshed, cv2.COLOR_BGR2GRAY)

    # Fill Flood
    cnts = cv2.findContours(threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        flood_filled = cv2.drawContours(threshed, [c], 0, (255, 255, 255), -1)

    # Morphology Operation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph_open = cv2.morphologyEx(flood_filled, cv2.MORPH_OPEN, kernel, iterations=2)


    clk_post_process.stop()
    print(', ', end='', flush=True)
    return morph_open


def cvt_label_to_image(labels):
    """
    레이블링된 전처리 이미지를 시각화
    0은 검은색, 나머지는 Hue값 돌아가면서 사용
    """
    image = np.zeros((len(labels), len(labels[0]), 3), 'uint8')
    for r, line in enumerate(labels):
        for c, item in enumerate(line):
            if item == 0:
                image[r][c] = [0, 0, 0]
            else:
                image[r][c] = [(item*5) % 180, 255, 255]
    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    return image


def cvt_stats_to_bboxes(stats):
    bboxes = []
    for line in stats:
        bboxes.append(['Undefined', line[0], line[1], line[0]+line[2], line[1]+line[3]])
    return bboxes


def label_img(img):
    """
    전처리된 이미지를 레이블링하고, 각 레이블마다 픽셀 정보를 담은 리스트를 반환
    :param img: 전처리된 이미지
    :return: 레이블 리스트
    """
    global clk_labeling
    print('ccl', end='', flush=True)
    clk_labeling.start()
    _, labels, stats, _ = cv2.connectedComponentsWithStatsWithAlgorithm(img, 4, cv2.CV_16U, cv2.CCL_WU)
    bboxes = cvt_stats_to_bboxes(stats)
    if debug_switch == 1:
        label_hue = cvt_label_to_image(labels)
        cv2.imwrite(LABEL_PATH + current_filename, label_hue)
    clk_labeling.stop()
    print(', ', end='', flush=True)
    return labels, bboxes


def remove_bbox_outside_scope(img, table):
    """
    스코프가 있는 이미지일 경우 사용. 스코프 밖으로 잡힌 bounding box 제거
    """
    print('remove_bbox_outside_scope', end='', flush=True)
    scope = get_roi.find_roi(img, 'circle')
    radius = scope[2]
    boxes = copy.deepcopy(table)
    boxes_remove = []

    for box in boxes:
        distance = ((scope[0]-(box[4]+box[2])/2)**2 + (scope[1]-(box[1]+box[3])/2)**2)**(1/2)

        if distance > radius:
            boxes_remove.append(box)

    for box in boxes_remove:
        boxes.remove(box)

    print(', ', end='', flush=True)
    return boxes


def remove_bbox_edge(img, bboxes, edge_error):
    print('remove_bbox_edge', end='', flush=True)
    boxes = copy.deepcopy(bboxes)
    boxes_remove = []
    for box in boxes:
        if box[1] < edge_error:
            boxes_remove.append(box)
        elif box[2] < edge_error:
            boxes_remove.append(box)
        elif box[3] > len(img[0])-edge_error:
            boxes_remove.append(box)
        elif box[4] > len(img)-edge_error:
            boxes_remove.append(box)

    for box in boxes_remove:
        boxes.remove(box)
    print(', ', end='', flush=True)
    return boxes


def set_norm_with_magnification():
    norm_plt = magnification * 50 / 1000
    norm_rbc = magnification * 100 / 1000
    return norm_plt, norm_rbc


def set_norm_with_histogram(table):
    sizes = []
    for cell in table:
        size = cell[2] + cell[4] - cell[1] - cell[3]
        sizes.append(size)
    hist = np.histogram(sizes, 100)
    norm_plt = 0
    temp_plt = 0
    sum_hist = 0
    for count in range(len(hist[1])):
        if sum_hist > 0.7*len(hist[1]):
            break
        if sum_hist > 0.05*len(hist[1]):
            norm_plt = temp_plt
        norm_rbc = hist[1][count]
        temp_plt = hist[1][count]
        sum_hist += hist[0][count]
    return norm_plt, norm_rbc


def set_categorize_norm(table):
    """
    카테고리를 나누기 전 rbc, plt를 구분할 기준 설정
    혈소판을 검색하는 방식 특성상 혈소판이 없고 rbc가 모두 붙어있는 경우 자동설정이 불가능함
    :param img: 원본 이미지
    :param table: bbox 리스트
    :return: plt 크기 범위, wbc or connected rbc 크기 범위
    """
    global magnification
    if magnification == 0:
        norm_plt, norm_rbc = set_norm_with_histogram(table)
    else:
        norm_plt, norm_rbc = set_norm_with_magnification()
    return norm_plt, norm_rbc


def categorize_bbox(img, table):
    """
    색상, 크기로 카테고리 분류
    :param img: 색상 비교를 위한 원본 이미지
    :param table: bbox 리스트
    :return: 카테고리 태그가 수정된 bbox 리스트
    """
    print('categorize', end='', flush=True)
    h, s, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    boxes = table.copy()
    norm_plt, norm_rbc = set_categorize_norm(table)

    for box in boxes:
        length = box[3] + box[4] - box[2] - box[1]
        if length < norm_plt:
            box[0] = 'Platelets'
        elif length < norm_rbc:
            box[0] = 'RBC'
        else:
            cropped_box = s[box[2]:box[4], box[1]:box[3]]
            _, threshed = cv2.threshold(cropped_box, 150, 255, cv2.THRESH_BINARY)
            count = 0
            for r, row in enumerate(threshed):
                for c, pixel in enumerate(row):
                    if pixel == 255:
                        count += 1
            if count > 100:
                box[0] = 'WBC'
            else:
                box[0] = 'RBC'
    print(', ', end='', flush=True)
    return boxes


def print_box(boxes, img):
    """
    bounding box 이미지에 출력해서 저장
    """
    print('print_box', end='', flush=True)
    names = ['WBC', 'RBC', 'Platelets']
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    for i, item in enumerate(boxes):
        color = colors[names.index(item[0])]
        xmin = item[1]
        ymin = item[2]
        xmax = item[3]
        ymax = item[4]
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
    cv2.imwrite(BBOX_PATH + current_filename, img)
    print(', ', end='', flush=True)


def organize_bbox(img, bboxes):
    """
    ccl로 레이블링된 bbox 후처리
    remove_bbox_outside_scope : 스코프 바깥 bbox 제거
    remove_bbox_edge : 이미지 가장자리에 붙은 bbox 제거
    categorize_bbox : RBC, WBC, PLT categorize 및 붙어있는 셀 houghcircle 처리
    :param img: 원본 이미지
    :param bboxes: bbox 리스트
    :return: 후처리된 bbox 리스트
    """
    global clk_organize_bbox
    clk_organize_bbox.start()

    if scope_switch == 1:
        bboxes = remove_bbox_outside_scope(img, bboxes)
    else:
        bboxes = remove_bbox_edge(img, bboxes, 2)

    bboxes = categorize_bbox(img, bboxes)
    if debug_switch == 1:
        print_box(bboxes, img)
    clk_organize_bbox.stop()
    return bboxes


def find_plt(img, labels):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    param1, param2, minRadius, maxRadius = [100, 9, 2, 6]
    '''if debug_switch == 1:
        windowName = 'circles'
        cv2.namedWindow(windowName)
        cv2.createTrackbar('param1', windowName, param1, 100, empty)
        cv2.createTrackbar('param2', windowName, param2, 100, empty)
        cv2.createTrackbar('minRadius', windowName, minRadius, 100, empty)
        cv2.createTrackbar('maxRadius', windowName, maxRadius, 100, empty)

        while True:
            if cv2.waitKey(10) == 27:
                break
            param1 = cv2.getTrackbarPos('param1', windowName) + 1
            param2 = cv2.getTrackbarPos('param2', windowName) + 1
            minRadius = cv2.getTrackbarPos('minRadius', windowName) + 1
            maxRadius = cv2.getTrackbarPos('maxRadius', windowName) + 1
            temp = copy.deepcopy(img)
            circles = cv2.HoughCircles(gray, method=cv2.HOUGH_GRADIENT, param1=param1, param2=param2, dp=1, minDist=20, minRadius=minRadius,
                               maxRadius=maxRadius)
            if circles is not None:
                for circle in circles[0, :]:
                    if labels[int(circle[1])][int(circle[0])] == 0:
                        cv2.circle(temp, (circle[0], circle[1]), circle[2], (255, 0, 0), 2)
                    else:
                        cv2.circle(temp, (circle[0], circle[1]), circle[2], (0, 0, 255), 2)
            cv2.imshow('circles', temp)
        print('hough circle parameters : ', param1, param2, minRadius, maxRadius)'''
    circles = cv2.HoughCircles(gray, method=cv2.HOUGH_GRADIENT, param1=param1, param2=param2, dp=1, minDist=10, minRadius=minRadius,
                               maxRadius=maxRadius)
    bboxes = []
    if circles is not None:
        for circle in circles[0, :]:
            if labels[int(circle[1])][int(circle[0])] == 0:
                bbox = ['Platelets', int(circle[0] - circle[2]), int(circle[1] - circle[2]), int(circle[0] + circle[2]),
                        int(circle[1] + circle[2])]
                bboxes.append(bbox)
    return bboxes


def cbc():
    global clk_total
    file_list = [_ for _ in os.listdir(INPUT_PATH) if _.endswith('.jpg')]

    for page, file in enumerate(file_list):
        global current_filename
        current_filename = file

        clk_total.start()
        print('%s (%s/%d) : ' % (current_filename, str(page + 1).zfill(2), len(file_list)), end='', flush=True)
        img_origin = cv2.imread(INPUT_PATH + current_filename)

        img_threshed = post_process(img_origin)
        labels, bboxes = label_img(img_threshed)
        plt_bboxes = find_plt(img_origin, labels)
        bboxes.extend(plt_bboxes)
        bboxes = organize_bbox(img_origin, bboxes)

        title = ['result', current_filename, OUTPUT_PATH + current_filename, '0',
                 str(len(img_origin[0])), str(len(img_origin)), '3', '0']
        labelimg_xml.write_xml(title, bboxes, OUTPUT_PATH, current_filename.split('.')[0] + '.xml')
        print('done.')
        clk_total.stop()


def calculate_elapsed_time():
    index_label = ['post-process', 'ccl', 'organizing bbox', 'whole process']

    print('processed images : %d' % len(clk_total.timestamp), end='\n\n')
    print('─' * 60)
    title_text = '{0:^15} │ {1:^6} │ {2:^6} │ {3:^6} │ {4:^9}'.format('process', 'min', 'max', 'avg', 'total')
    len_text = len(title_text)
    print(title_text)
    for index, item in enumerate([clk_post_process, clk_labeling, clk_organize_bbox, clk_total]):
        item.calulate_runningtime()
        text_title = '{0:<15}'.format(index_label[index])
        min_text = '{:>6}'.format('%.3f' % item.min)
        max_text = '{:>6}'.format('%.3f' % item.max)
        avg_text = '{:>6}'.format('%.3f' % item.avg)
        sum_text = '{:>9}'.format('%.3f' % item.sum)
        if index == 0 or index == -1:
            print('─' * len_text)
        print('%s │ %s │ %s │ %s │ %s' % (text_title, min_text, max_text, avg_text, sum_text))
    print('─' * 60)


def main(args):
    global debug_switch, scope_switch, magnification
    debug_switch = 1 if '-d' in args else 0
    scope_switch = 1 if '-s' in args else 0
    magnification = int(args[args.index('-m') + 1]) if '-m' in args else 0

    set_directories()
    cbc()
    print('\n\nAll process done.')
    calculate_elapsed_time()


if __name__ == '__main__':
    main(sys.argv[:])