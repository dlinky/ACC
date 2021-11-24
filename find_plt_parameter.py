import cv2
import os
import numpy as np
import copy

import labelimg_xml


def post_process(img):
    """
    이미지 전처리
    1. hue, value로 adaptive theshold하여, 둘다 만족하는 픽셀만 취함
    2. flood fill로 안쪽 채움
    3. morphology operation(open)
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    threshold_block_size = len(v)//3 + (len(v)//3+1)%2
    threshed_value = cv2.adaptiveThreshold(v, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, threshold_block_size, 0)
    threshed_hue = cv2.adaptiveThreshold(h, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, threshold_block_size, 0)
    threshed = threshed_value
    # threshed = cv2.bitwise_and(threshed_hue, threshed_value)

    # Fill Flood
    cnts = cv2.findContours(threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        flood_filled = cv2.drawContours(threshed, [c], 0, (255, 255, 255), -1)

    # Morphology Operation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph_open = cv2.morphologyEx(flood_filled, cv2.MORPH_OPEN, kernel, iterations=2)

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


def label_img(threshed):
    """
    전처리된 이미지를 레이블링하고, 각 레이블마다 픽셀 정보를 담은 리스트를 반환
    :param img: 전처리된 이미지
    :return: 레이블 리스트
    """
    _, labels, stats, _ = cv2.connectedComponentsWithStatsWithAlgorithm(threshed, 4, cv2.CV_16U, cv2.CCL_WU)
    bboxes = cvt_stats_to_bboxes(stats)
    return labels, bboxes


def empty(self):
    pass


def find_plt(img, labels, param1, param2, minDist, minRadius, maxRadius):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    circles = cv2.HoughCircles(gray, method=cv2.HOUGH_GRADIENT, param1=param1, param2=param2, dp=1,
                                                   minDist=minDist, minRadius=minRadius,
                                                   maxRadius=maxRadius)
    bboxes = []
    if circles is not None:
        for circle in circles[0, :]:
            if labels[int(circle[1])][int(circle[0])] == 1:
                bbox = ['Platelets', int(circle[0] - circle[2]), int(circle[1] - circle[2]),
                        int(circle[0] + circle[2]),
                        int(circle[1] + circle[2])]
                bboxes.append(bbox)
    return bboxes


def get_precision(gt_bboxes, dr_bboxes):
    ovmax = 1
    ovmin = 0.5
    tp = [0] * len(dr_bboxes)
    fp = [0] * len(dr_bboxes)
    for idx, bb in enumerate(dr_bboxes):
        for bbgt in gt_bboxes:
            if 'Platelets' in bbgt:
                bi = [max(bb[1], bbgt[1]), max(bb[2], bbgt[2]), min(bb[3], bbgt[3]), min(bb[4], bbgt[4])]
                iw = bi[2] - bi[0] + 1
                ih = bi[3] - bi[1] + 1
                if iw > 0 and ih > 0:
                    # compute overlap (IoU) = area of intersection / area of union
                    ua = (bb[3] - bb[1] + 1) * (bb[4] - bb[2] + 1) + (bbgt[3] - bbgt[1]
                                                                      + 1) * (bbgt[4] - bbgt[2] + 1) - iw * ih
                    ov = iw * ih / ua
                    if ov > ovmax:
                        pass
                    elif ov > ovmin:
                        if 'used' in bbgt:
                            fp[idx] = 1
                        else:
                            tp[idx] = 1
                    else:
                        fp[idx] = 1
                bbgt.append('used')
        if tp[idx] == 0:
            fp[idx] = 1
    
    cumsum = 0
    for idx, val in enumerate(fp):
        fp[idx] += cumsum
        cumsum += val
    cumsum = 0
    for idx, val in enumerate(tp):
        fp[idx] += cumsum
        cumsum += val
    precision = tp[-1]/(fp[-1]+tp[-1])
    return precision


def main():

    original_dir = os.path.join(os.getcwd(), 'original')
    gt_dir = os.path.join(os.getcwd(), 'ground-truth')
    dr_dir = os.path.join(os.getcwd(), 'detection-results')
    file_list = os.listdir(original_dir)

    for file in file_list:
        img = cv2.imread(os.path.join(original_dir, file))
        threshed = post_process(img)
        labels, stats = label_img(threshed)

        title, table = labelimg_xml.read_xml(gt_dir+'/', file.replace('jpg', 'xml'))

        param1s = [1, 100]
        param2s = [1, 100]
        minDists = [100, 1000]
        minRadiuss = [1, 15]
        maxRadiuss = [10, 30]
        precisions = []

        for param1 in range(param1s[0], param1s[1], 10):
            for param2 in range(param2s[0], param2s[1], 10):
                for minDist in range(minDists[0], minDists[1], 150):
                    for minRadius in range(minRadiuss[0], minRadiuss[1], 2):
                        for maxRadius in range(maxRadiuss[0], maxRadiuss[1], 2):
                            bboxes = find_plt(img, labels, param1, param2, minDist, minRadius, maxRadius)
                            if len(bboxes) == 0:
                                precision = 0
                            else:
                                    precision = get_precision(table, bboxes)
                            print([param1, param2, minDist, minRadius, maxRadius], ' : ', precision)
                            precisions.append(precision)
        param_max = precisions.index(max(precisions))
        print('max parameters : ', param_max)


if __name__ == '__main__':
    main()