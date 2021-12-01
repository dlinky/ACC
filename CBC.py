import os
import shutil
import glob
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import time
import sys
import argparse

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
SAVE_IMAGE = False
if args.magnification is not None:
    specific_magnification = True
    magnification = args.magnification

if args.save_images is not None:
    SAVE_IMAGE = True
    
    
# paths
INPUT_PATH = os.path.join(os.getcwd(), 'input')
OUTPUT_PATH = os.path.join(os.getcwd(), 'output')
XML_PATH = os.path.join(OUTPUT_PATH, 'xml')
THRESHOLD_PATH = os.path.join(OUTPUT_PATH, 'threshold')
LABEL_PATH = os.path.join(OUTPUT_PATH, 'labeled')
BBOX_PATH = os.path.join(OUTPUT_PATH, 'bbox')
CLASSES_PATH = os.path.join(OUTPUT_PATH, 'bbox', 'classes')


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
    l, a, b = cv2.split(cv2.cvtColor(src, cv2.COLOR_BGR2LAB))
    _, threshed = cv2.threshold(a, 0, 255, cv2.THRESH_OTSU)
    cnts = cv2.findContours(threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        threshed = cv2.drawContours(threshed, [c], 0, (255, 255, 255), -1)
    morph_open = cv2.morphologyEx(threshed, cv2.MORPH_OPEN, np.ones((5, 5)), iterations=2)
    if SAVE_IMAGE:
        temp = src[:]
        temp[morph_open == 0] = 0
        cv2.imwrite(os.path.join(THRESHOLD_PATH, current_filename), temp)
    return morph_open


def label_cells(threshed):
    _, labels, stats, _ = cv2.connectedComponentsWithStats(threshed)
    labeled_stats = []
    for box in stats:
        labeled_stats.append([box[0], box[1], box[2], box[3], box[4], 'Undefined'])

    return labels, labeled_stats[1:]


def cvt_label_to_image(labels):
    image = np.zeros((len(labels), len(labels[0]), 3), 'uint8')
    for r, line in enumerate(labels):
        for c, item in enumerate(line):
            if item == 0:
                image[r][c] = [0, 0, 0]
            else:
                image[r][c] = [(item*5) % 180, 255, 255]
    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    return image


def remove_border_touching_bbox(labels, stats, edge_error):
    height = len(labels)
    width = len(labels[0])
    new_stats = []
    for idx, box in enumerate(stats[:]):
        lboard = box[0] < edge_error
        rboard = box[0] + box[2] > height - edge_error
        tboard = box[1] < edge_error
        bboard = box[1] + box[3] > width - edge_error
        if lboard or rboard or tboard or bboard:
            labels[labels == idx] = 0
        else:
            new_stats.append(box)
    return labels, new_stats


def double_threshold(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    ret_cell, th1_a = cv2.threshold(a, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_TOZERO)
    th1_a[th1_a == 0] = ret_cell
    ret_nucleus, th2_a = cv2.threshold(th1_a, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    return ret_cell, ret_nucleus


def find_wbc(img, stats):
    wbcs = 0
    for box in stats:
        ret_cell, ret_nucleus = double_threshold(img[box[1]:box[1]+box[3], box[0]:box[0]+box[2]])
        if ret_nucleus - ret_cell > 10:
            box[-1] = 'WBC'
            wbcs = 1
    return wbcs, stats


def classify_boxes(num_clst, stats):
    # Step 1 : 그룹의 갯수를 정하고, 각 그룹의 중심 값을 정함

    data = np.array(stats).T[4].astype('int16')
    data = np.array(data)
    # num_clst = 3

    mxmn = [np.min(data), np.max(data)]
    quan = (mxmn[1] - mxmn[0]) / num_clst / 2

    # 클러스터의 갯수만큼 중심점을 만듬
    clst_mu = np.array([quan * (2*i+1) + mxmn[0] for i in range(num_clst)])
    # print(f'* 각 클러스터별 평균(설정) : {clst_mu}')

    # Step 2 : 각 데이터에서 가장 가까운 그룹을 찾아서 그룹을 할당

    # 데이터별 중심점과의 거리를 계산
    diff = np.abs(data - clst_mu.reshape((num_clst, 1))).T

    # 그룹을 할당
    clst = np.where(diff == diff.min(axis=1).reshape((len(data), 1)))[1]
    if len(clst) > len(data):
        for count in range(len(clst) - len(data)):
            clst = np.delete(clst, len(clst) - 1)

    # Step 3 : 할당 결과를 가지고 그룹의 중심점을 재계산
    for i in range(num_clst):
        clst_mu[i] = np.mean(data[np.where(clst == i)[0]])
        # print(data[np.where(clst==i)[0]])
    print(f'* 각 클러스터 평균(초기) : {clst_mu}')

    # Step 4 : 2,3 스텝을 계속 반복

    maxEpoch = 10
    for epoch in range(maxEpoch):
        print(f'#{epoch+1}')
        diff = np.abs(data - clst_mu.reshape((num_clst, 1))).T
        clst = np.where(diff == diff.min(axis=1).reshape((len(data), 1)))[1]
        print(clst)

        for i in range(num_clst):
            clst_mu[i] = np.mean(data[np.where(clst == i)[0]])

        print(f'* 각 클러스터 평균(#{epoch}) : {clst_mu}')

    label_list = ['Platelets', 'RBC', 'RBC', 'WBC'] # 일반 RBC, 덩어리진 RBC
    for idx, label_idx in enumerate(clst):
        if idx == 3:
            if stats[idx][-1] == 'WBC':
                pass
            else:
                stats[idx][-1] = 'RBC'
        else:
            stats[idx][-1] = label_list[label_idx]
    return stats


def cvt_stats_to_bboxes(stats):
    bboxes = []
    for line in stats:
        bboxes.append([line[5], line[0], line[1], line[0]+line[2], line[1]+line[3]])
    return bboxes


def print_box(boxes, img):
    names = ['WBC', 'RBC', 'Platelets', 'Undefined']
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255)]
    for i, item in enumerate(boxes):
        color = colors[names.index(item[0])]
        xmin = item[1]
        ymin = item[2]
        xmax = item[3]
        ymax = item[4]
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
    cv2.imwrite(os.path.join(BBOX_PATH, current_filename), img)


def write_xml(img, table):
    root = ET.Element('annotation')
    ET.SubElement(root, 'folder').text = os.path.dirname(OUTPUT_PATH)
    ET.SubElement(root, 'filename').text = current_filename
    ET.SubElement(root, 'path').text = os.path.join(XML_PATH, current_filename)

    source = ET.SubElement(root, 'source')
    ET.SubElement(source, 'database').text = '0'

    size = ET.SubElement(root, 'size')
    ET.SubElement(size, 'width').text = str(len(img[0]))
    ET.SubElement(size, 'height').text = str(len(img))
    ET.SubElement(size, 'depth').text = '3'

    ET.SubElement(root, 'segmented').text = '0'

    for line in table:
        obj = ET.SubElement(root, 'object')
        ET.SubElement(obj, 'name').text = line[0]
        ET.SubElement(obj, 'pose').text = 'Unspecified'
        ET.SubElement(obj, 'truncated').text = '0'
        ET.SubElement(obj, 'difficult').text = '0'

        bndbox = ET.SubElement(obj, 'bndbox')
        ET.SubElement(bndbox, 'xmin').text = str(line[1])
        ET.SubElement(bndbox, 'ymin').text = str(line[2])
        ET.SubElement(bndbox, 'xmax').text = str(line[3])
        ET.SubElement(bndbox, 'ymax').text = str(line[4])

    tree = ET.ElementTree(root)
    tree.write(os.path.join(XML_PATH, current_filename))


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


def main():
    if os.path.exists(OUTPUT_PATH):
        shutil.rmtree(OUTPUT_PATH)
    os.makedirs(XML_PATH)
    if args.save_images:
        os.makedirs(THRESHOLD_PATH)
        os.makedirs(LABEL_PATH)
        os.makedirs(BBOX_PATH)

    file_list = glob.glob(INPUT_PATH + '/*.jpg')
    if len(file_list) == 0:
        error("Error: No input files found")
    for idx, filename in enumerate(file_list):
        global current_filename
        current_filename = os.path.basename(filename)
        img = cv2.imread(os.path.join(INPUT_PATH, filename))
        print('threshold')
        threshed = threshold(img)
        print('labeling')
        labels, stats = label_cells(threshed)
        labels, stats = remove_border_touching_bbox(labels, stats, 1)

        print('find wbc')
        wbcs, stats = find_wbc(img, stats)
        for line in stats:
            print(line)
        if wbcs > 0:
            num_clst = 4
        else:
            num_clst = 3
        print('classify')
        stats = classify_boxes(num_clst, stats)
        bboxes = cvt_stats_to_bboxes(stats)

        if SAVE_IMAGE:
            labels_color = cvt_label_to_image(labels)
            cv2.imwrite(os.path.join(LABEL_PATH, current_filename), labels_color)
            print_box(bboxes, img)

        write_xml(img, bboxes)


if __name__ == '__main__':
    main()