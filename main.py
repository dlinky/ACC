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
parser.add_argument('-m', '--magnification', default=1000, action='store', help='set manual magnification')
parser.add_argument('-rs', '--remove-scope', action='store_true', default=False, help='remove scope')
args = parser.parse_args()

# arguments
magnification = args.magnification
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
    idx = np.array(range(len(stats)), np.uint8).reshape(len(stats),1)
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
                image[r][c] = [(item*5) % 180, 255, 255]
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
            labels[labels == idx+1] = 0
            remove_stats.append(idx)
    stats = np.delete(stats, remove_stats, axis=0)
    cv2.waitKey(0)
    return labels, stats[1:]


def double_threshold(roi):
    """
    process otsu threshold twice with a
    :param roi:
    :return: 
    """
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    ret_cell, th1_a = cv2.threshold(a, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_TOZERO)
    th1_a[th1_a == 0] = ret_cell
    ret_nucleus, th2_a = cv2.threshold(th1_a, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)

    return ret_cell, ret_nucleus


def find_wbc(img, stats):
    wbcs = 0
    for box in stats:
        roi = img[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
        ret_cell, ret_nucleus = double_threshold(roi)
        if ret_nucleus - ret_cell > 10:
            box[-1] = WBC
            wbcs += 1
            if SAVE_IMAGE:
                a = cv2.split(cv2.cvtColor(roi, cv2.COLOR_BGR2LAB))[1]
                _, mask_cell = cv2.threshold(a, ret_cell, 255, cv2.THRESH_TOZERO)
                masked_cell = cv2.bitwise_and(roi, roi, mask=mask_cell)
                _, mask_nucleus = cv2.threshold(a, ret_nucleus, 255, cv2.THRESH_TOZERO)
                masked_nucleus = cv2.bitwise_and(roi, roi, mask=mask_nucleus)
                masked_img = cv2.addWeighted(masked_cell, 0.5, masked_nucleus, 0.5, 0)
                cv2.imwrite(os.path.join(CLASSES_PATH, current_filename.replace('.jpg', f'-wbc{wbcs}.jpg')), masked_img)
    return wbcs, stats


def watershed(labels, box, img):
    rmin = int(max(box[1] - 0.5 * box[3], 0))
    rmax = int(min(box[1] + 1.5 * box[3], len(labels)))
    cmin = int(max(box[0] - 0.5 * box[2], 0))
    cmax = int(min(box[0] + 1.5 * box[2], len(labels[0])))
    box_label = box[-2]
    roi = img.copy()[rmin:rmax, cmin:cmax]
    a = cv2.split(cv2.cvtColor(roi, cv2.COLOR_BGR2LAB))[1]

    mask = np.uint8(labels[rmin:rmax, cmin:cmax])
    mask[mask != box_label] = 0
    mask[mask == box_label] = 255
    masked = cv2.bitwise_and(a, a, mask=mask)

    _, threshed = cv2.threshold(masked, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    kernel = np.ones((3,3))
    opening = cv2.morphologyEx(threshed, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.5*dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(roi, markers)
    markers[markers == -1] = 0
    markers[markers == 1] = 0
    markers = cv2.morphologyEx(np.uint8(markers), cv2.MORPH_OPEN, kernel)
    _, _, stats, _ = cv2.connectedComponentsWithStats(np.uint8(markers), connectivity=8)

    label = np.full((len(stats), 1), UNKNOWN, np.uint8)
    idx = np.array(range(len(stats+box[5])), np.uint8).reshape(len(stats), 1)
    stats = np.concatenate((stats, idx, label), axis=1)
    for idx, stat in enumerate(stats[:]):
        if stat[0] == 0:
            stats = np.delete(stats, idx, axis=0)
    print(stats)
    for stat in stats:
        cv2.rectangle(roi, (stat[0], stat[1]), (stat[0] + stat[2], stat[1] + stat[3]), (0, 0, 255), 2)
    stats += np.array([box[0], box[1], 0, 0, 0, box[5], 0])

    images = [masked, threshed, sure_bg, dist_transform, sure_fg, unknown, markers, roi]
    titles = ['Gray', 'Threshed', 'Sure BG', 'Distance', 'Sure FG', 'Unknown', 'Markers', 'Result']

    for i in range(len(images)):
        plt.subplot(2, 4, i + 1), plt.imshow(images[i]), plt.title(titles[i]), plt.xticks([]), plt.yticks([])
    plt.show()
    return stats


def separate_with_circle(img, box):
    roi = img[box[1]-2:box[1] + box[3]+2, box[0]-2:box[0] + box[2]+2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    windowName = 'separate'
    cv2.namedWindow(windowName)
    cv2.createTrackbar('mindist', windowName, 1, 100, empty)
    cv2.createTrackbar('param1', windowName, 1, 255, empty)
    cv2.createTrackbar('param2', windowName, 1, 255, empty)
    cv2.createTrackbar('minradius', windowName, 1, 255, empty)
    cv2.createTrackbar('maxradius', windowName, 1, 255, empty)

    while True:
        temp = copy.deepcopy(roi)
        if cv2.waitKey(10) == 27:
            break
        minDist = cv2.getTrackbarPos('mindist', windowName)
        param1 = cv2.getTrackbarPos('param1', windowName)
        param2 = cv2.getTrackbarPos('param2', windowName)
        minRadius = cv2.getTrackbarPos('minradius', windowName)
        maxRadius = cv2.getTrackbarPos('maxradius', windowName)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, minDist=minDist+1, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
        if circles is not None:
            for i in circles[0,:]:
                cv2.circle(temp, (i[0], i[1]), i[2], (0, 255, 0), 2)

        cv2.resizeWindow(windowName, 500, 1000)
        cv2.imshow(windowName, temp)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, minDist=minDist, param1=param1, param2=param2,
                               minRadius=minRadius, maxRadius=maxRadius)
    if circles is not None:
        new_box = np.zeros((1, 6))
        for i in circles[0,:]:
            x_min = int(i[0]-i[2])
            y_min = int(i[1]-i[2])
            new_box = np.append(new_box, [x_min, y_min, int(2*i[2]), int(2*i[2]), int(2*3.14*i[2]**2)])
        return new_box
    else:
        return box


def cluster_boxes(num_clst, stats):
    data = np.array(stats).T[4].astype('int16')
    data = np.array(data)

    mxmn = [np.min(data), np.max(data)]
    quan = (mxmn[1] - mxmn[0]) / num_clst / 2
    clst_mu = np.array([quan * (2 * i + 1) + mxmn[0] for i in range(num_clst)])

    diff = np.abs(data - clst_mu.reshape((num_clst, 1))).T
    clst = np.where(diff == diff.min(axis=1).reshape((len(data), 1)))[1]
    if len(clst) > len(data):
        for count in range(len(clst) - len(data)):
            clst = np.delete(clst, len(clst) - 1)

    for i in range(num_clst):
        clst_mu[i] = np.mean(data[np.where(clst == i)[0]])

    maxEpoch = 5
    for epoch in range(maxEpoch):
        diff = np.abs(data - clst_mu.reshape((num_clst, 1))).T
        clst = np.where(diff == diff.min(axis=1).reshape((len(data), 1)))[1]
        for i in range(num_clst):
            clst_mu[i] = np.mean(data[np.where(clst == i)[0]])

    label_list = [PLATELET, RBC, WBC, UNKNOWN]  # 일반 RBC, 덩어리진 RBC
    for idx, label_idx in enumerate(clst):
        if idx == 3:
            if stats[idx][-1] == WBC:
                pass
            else:
                stats[idx][-1] = RBC
        else:
            stats[idx][-1] = label_list[label_idx]
    return stats


def classify_boxes(num_clst, stats):
    stats = cluster_boxes(num_clst, stats)
    return stats


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

        for idx, stat in enumerate(stats):
            new_stats = watershed(labels, stat, img)
            # new_stats = separate_with_circle(img, stat)
            if new_stats.ndim == 2 and len(new_stats) > 1:
                if SAVE_IMAGE:
                    temp = copy.deepcopy(img)
                    for new_stat in new_stats:
                        cv2.rectangle(temp, (new_stat[0], new_stat[1]), (new_stat[0]+new_stat[2], new_stat[1]+new_stat[3]), (0, 255, 0), 2)
                    roi = temp[stat[1]:stat[1] + stat[3], stat[0]:stat[0] + stat[2]]
                    cv2.imwrite(os.path.join(WATERSHED_PATH, current_filename.replace('.jpg', f'-{idx+1}({len(new_stats)}).jpg')), roi)
                stats = np.delete(stats, idx, axis=0)
                stats = np.append(stats, tuple(new_stats), axis=0)
        clk_detect.stop()
        
        clk_classify.start()
        wbcs, stats = find_wbc(img, stats)
        if wbcs > 0:
            num_clst = 3
        else:
            num_clst = 2
        stats = classify_boxes(num_clst, stats)
        bboxes = cvt_stat(stats)
        clk_classify.stop()

        if SAVE_IMAGE:
            labels_color = cvt_label_to_image(labels)
            cv2.imwrite(os.path.join(LABEL_PATH, current_filename), labels_color)
            print_box(bboxes, img)

        write_xml(img, bboxes)
        clk_total.stop()

    if RUN_TIME:
        print_run_time()
        

if __name__ == '__main__':
    main()