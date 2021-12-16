import copy
import os
import shutil
import glob
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import time
import sys
import argparse
import remove_scope
from matplotlib import pyplot as plt

current_filename = ''

# parameters
parser = argparse.ArgumentParser(description='Complete Blood Count with OpenCV')
parser.add_argument('-si', '--save-images', action='store_true', default=False, help='save output images')
parser.add_argument('-t', '--run-time', action='store_true', default=False, help='show process run time')
parser.add_argument('-rs', '--remove-scope', action='store_true', default=False, help='remove scope')
args = parser.parse_args()

# arguments
SAVE_IMAGE = args.save_images
RUN_TIME = args.run_time
REMOVE_SCOPE = args.remove_scope

# constants
PLATELET = 0
RBC = 1
WBC = 2
UNKNOWN = 3

# paths
INPUT_PATH = os.path.join(os.getcwd(), 'input')
OUTPUT_PATH = os.path.join(os.getcwd(), 'output')
XML_PATH = os.path.join(OUTPUT_PATH, 'xml')
THRESHOLD_PATH = os.path.join(OUTPUT_PATH, 'threshold')
LABEL_PATH = os.path.join(OUTPUT_PATH, 'labeled')
BBOX_PATH = os.path.join(OUTPUT_PATH, 'bbox')
CLASSES_PATH = os.path.join(OUTPUT_PATH, 'bbox', 'classes')
WATERSHED_PATH = os.path.join(OUTPUT_PATH, 'labeled', 'watershed')


def error(msg):
    print(msg)
    sys.exit(0)


def empty(self):
    pass


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

    def get_run_time(self):
        self.max = max(self.timestamp)
        self.min = min(self.timestamp)

        for item in self.timestamp:
            self.sum += item
        self.avg = self.sum / len(self.timestamp)

        return [self.min, self.max, self.sum, self.avg]


def print_progress(iteration, total):
    decimals = 1
    barLength = 20
    prefix = ''
    suffix = ''
    formatStr = "{0:." + str(decimals) + "f}"
    percent = formatStr.format(100 * (iteration / float(total)))
    filledLength = int(round(barLength * iteration / float(total)))
    bar = '#' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s (%s%s) %s' % (prefix, bar, current_filename, percent, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()


clk_total = Clock('Whole Process')
clk_detect = Clock('Detect cells')
clk_classify = Clock('Classify cells')


def threshold(src):
    """
    get pre-processed image
    - otsu threshold to get foreground
    - flood filling to fill inside
    - open operation to remove noises

    :param src: BGR image
    :return: pre-processed binary image
    """
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
    label = np.full((len(stats), 1), UNKNOWN, np.uint8)
    idx = np.array(range(len(stats)), np.uint8).reshape(len(stats), 1)
    stats = np.concatenate((stats, idx, label), axis=1)
    return labels, stats[1:]


def cvt_label_to_image(labels):
    """
    convert from label image to color images
    make empty np.array and filling HSV image, and convert to BGR
    :param labels:
    :return:
    """
    image = np.zeros((len(labels), len(labels[0]), 3), 'uint8')
    for r, line in enumerate(labels):
        for c, item in enumerate(line):
            if item == 0:
                image[r][c] = [0, 0, 0]
            else:
                image[r][c] = [(item * 5) % 180, 255, 255]
    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    return image


def remove_border_touching_bbox(labels, stats, edge_error):
    height = len(labels)
    width = len(labels[0])
    remove_stats = []
    for idx, box in enumerate(stats):
        lboard = box[0] < edge_error
        rboard = box[0] + box[2] > height - edge_error
        tboard = box[1] < edge_error
        bboard = box[1] + box[3] > width - edge_error
        if lboard or rboard or tboard or bboard:
            labels[labels == idx + 1] = 0
            remove_stats.append(idx)
    stats = np.delete(stats, remove_stats, axis=0)
    cv2.waitKey(0)
    return labels, stats[1:]


def detect_by_size(img, normal_size):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    wbcMinRadius = int(normal_size * 1.5)
    wbcMaxRadius = int(normal_size * 3)
    rbcMinRadius = int(normal_size * 0.75)
    rbcMaxRadius = int(normal_size * 1.49)
    pltMinRadius = int(normal_size * 0.1)
    pltMaxRadius = int(normal_size * 0.74)
    print(wbcMinRadius, wbcMaxRadius, rbcMinRadius, rbcMaxRadius, pltMinRadius, pltMaxRadius)

    # WBC
    windowName = 'detect by size'
    cv2.namedWindow(windowName)
    cv2.createTrackbar('minDist', windowName, 3, 100, empty)
    cv2.createTrackbar('param1', windowName, 3, 255, empty)
    cv2.createTrackbar('param2', windowName, 3, 255, empty)
    cv2.createTrackbar('minRadius', windowName, 3, 200, empty)
    cv2.createTrackbar('maxRadius', windowName, 3, 200, empty)
    while True:
        if cv2.waitKey(10) == 27:
            break
        temp = img.copy()

        minDist = cv2.getTrackbarPos('minDist', windowName)
        param1 = cv2.getTrackbarPos('param1', windowName)
        param2 = cv2.getTrackbarPos('param2', windowName)
        minRadius = cv2.getTrackbarPos('minRadius', windowName)
        maxRadius = cv2.getTrackbarPos('maxRadius', windowName)
        circles = cv2.HoughCircles(a, cv2.HOUGH_GRADIENT, 0.3, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
        #circles = np.uint16(np.around(circles))

        if circles is not None:
            for i in circles[0,:]:
                cv2.circle(temp, (i[0], i[1]), i[2], (0, 255, 0), 2)

        cv2.imshow(windowName, temp)


def cvt_stat(stats):
    bboxes = []
    for line in stats:
        bboxes.append([line[-1], line[0], line[1], line[0]+line[2], line[1]+line[3]])
    return bboxes


def print_box(boxes, img):
    names = [WBC, RBC, PLATELET, UNKNOWN]
    colors = [(255, 255, 255), (0, 0, 255), (255, 0, 0), (50, 50, 50)]
    for i, item in enumerate(boxes):
        color = colors[names.index(item[0])]
        xmin = item[1]
        ymin = item[2]
        xmax = item[3]
        ymax = item[4]
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
    cv2.imwrite(os.path.join(BBOX_PATH, current_filename), img)


def write_xml(img, table):
    names = ['Platelets', 'RBC', 'WBC', 'Unknown']
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
        ET.SubElement(obj, 'name').text = names[line[0]]
        ET.SubElement(obj, 'pose').text = 'Unspecified'
        ET.SubElement(obj, 'truncated').text = '0'
        ET.SubElement(obj, 'difficult').text = '0'

        bndbox = ET.SubElement(obj, 'bndbox')
        ET.SubElement(bndbox, 'xmin').text = str(line[1])
        ET.SubElement(bndbox, 'ymin').text = str(line[2])
        ET.SubElement(bndbox, 'xmax').text = str(line[3])
        ET.SubElement(bndbox, 'ymax').text = str(line[4])

    tree = ET.ElementTree(root)
    tree.write(os.path.join(XML_PATH, current_filename.replace('.jpg', '.xml')))


def print_run_time():
    clocks = [clk_detect, clk_classify, clk_total]
    index_label = [_.title for _ in clocks]

    print('\n\nprocessed images : %d' % len(clk_total.timestamp), end='\n\n')
    print('─' * 60)
    title_text = f'{"process":^15} │ {"min":^6} │ {"max":^6} │ {"avg":^6} │ {"total":^9}'
    len_text = len(title_text)
    print(title_text)
    for index, item in enumerate(clocks):
        item.get_run_time()
        text_title = f'{index_label[index]:<15}'
        min_text = f'{item.min:>6.3f}'
        max_text = f'{item.max:>6.3f}'
        avg_text = f'{item.avg:>6.3f}'
        sum_text = f'{item.sum:>9.3f}'
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
        os.makedirs(CLASSES_PATH)
        os.makedirs(WATERSHED_PATH)

    file_list = glob.glob(INPUT_PATH + '/*.jpg')
    if len(file_list) == 0:
        error("Error: No input files found")
    for page, filename in enumerate(file_list):
        clk_total.start()
        print_progress(page, len(file_list))
        global current_filename
        current_filename = os.path.basename(filename)

        img = cv2.imread(os.path.join(INPUT_PATH, filename))
        if REMOVE_SCOPE:
            cropped_img = remove_scope.get_cropped_img(img, find_manually=False)
            if cropped_img is None:
                continue
            else:
                img = cropped_img

        clk_detect.start()
        threshed = threshold(img)
        labels, stats = label_cells(threshed)
        labels, stats = remove_border_touching_bbox(labels, stats, 1)

        normal_area = np.mean(stats.T[4])
        normal_size = normal_area**(0.5)

        detect_by_size(img, normal_size)


if __name__ == "__main__":
    main()