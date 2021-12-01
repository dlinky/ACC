import glob
import os
import shutil
import cv2
import numpy as np
import argparse

PATH_DIR = os.path.join(os.getcwd(), 'crop')
INPUT_PATH = os.path.join(PATH_DIR, 'input')
OUTPUT_PATH = os.path.join(PATH_DIR, 'output')
SCOPE_PATH = os.path.join(PATH_DIR, 'scope')
os.makedirs(INPUT_PATH, exist_ok=True)
if os.path.isdir(SCOPE_PATH):
    shutil.rmtree(SCOPE_PATH)
if os.path.isdir(OUTPUT_PATH):
    shutil.rmtree(OUTPUT_PATH)

parser = argparse.ArgumentParser(description='Crop image from microscopic image')
parser.add_argument('-i', '--input-path', action='store', default=INPUT_PATH, help='manual input file path')
parser.add_argument('-o', '--output-path', action='store', default=OUTPUT_PATH, help='manual output file path')
parser.add_argument('-v', '--verify', action='store_true', default=False, help='verify scope detection')
parser.add_argument('-m', '--manual-roi', action='store_true', default=False, help='set roi manually')
args = parser.parse_args()

points = []


def clicked(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        global points
        if len(points) == 3:
            points.remove(points[0])
        points.append([x,y])


def select_scope():
    a, b, c = points
    X = np.array(points)
    Y = [1,1]
    n = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
    n = n/np.linalg.norm(n)
    u = np.cross(b-a, n)
    v = np.cross(c-b, n)
    X = np.array([[u[0], -v[0]], [u[1], -v[1]]])
    Y = np.array([(c[0]-a[0])/2, (c[1]-a[1])/2])
    t = np.linalg.inv(X).dot(Y)
    point = (a+b)/2 + u*t[0]
    r = np.linalg.norm(point-a)

    return [int(point[0]), int(point[1]), int(r)]


INPUT_PATH = args.input_path
OUTPUT_PATH = args.output_path
if args.verify:
    os.makedirs(SCOPE_PATH)
os.makedirs(OUTPUT_PATH)

file_list = glob.glob(INPUT_PATH + '/*.jpg')
print(INPUT_PATH, file_list)
for file in file_list:
    filename = os.path.basename(file)
    img = cv2.imread(file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:]
    minRadius = int(min((height, width)) * 400 / 1280)
    maxRadius = int(min((height, width)) * 1000 / 1280)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 0.5, 100, param1=100, param2=100, minRadius=minRadius,
                               maxRadius=maxRadius)
    if circles is not None:
        circles = np.uint16(np.around(circles))

        final_circle = circles[0, 0]
        for count, current_circle in enumerate(circles[0, :]):
            # 중심에 가장 가까운 원 저장
            if abs(final_circle[0] - width / 2) > abs(current_circle[0] - width / 2) and abs(
                    final_circle[1] - height / 2) > abs(current_circle[1] - height / 2):
                final_circle = current_circle
        halflen = int(final_circle[2] / 1.414 * 0.99)  # 스코프 내접 사각형 길이/2 (99%)
        xmin = final_circle[0] - halflen
        ymin = final_circle[1] - halflen
        xmax = final_circle[0] + halflen
        ymax = final_circle[1] + halflen
        square = img[ymin:ymax, xmin:xmax]
        cv2.imwrite(os.path.join(OUTPUT_PATH, filename), square)
        if args.verify:
            cv2.circle(img, (final_circle[0], final_circle[1]), final_circle[2], (0, 255, 0), 2)
            cv2.imwrite(os.path.join(SCOPE_PATH, filename), img)
    else:
        if args.manual_roi:
            global points
            points = []
            while True:
                if cv2.waitKey(10) == 27:
                    break
                cv2.imshow('img', img)
                cv2.setMouseCallback('img', clicked, img)
        else:
            print(f'{filename} : circles not detected. crop manually')