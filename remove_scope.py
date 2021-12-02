import glob
import os
import shutil
import cv2
import numpy as np
import argparse

points = []


def clicked(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        global points
        del points[0]
        points.append([x, y])


def find_scope(img, find_manually):
    """
    find scope from microscopic image
    :param img: microscopic image
    :param find_manually: if couldn't find scope, find scope manually
    :return: coordinate and radius of scope
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:]
    minRadius = int(min((height, width)) * 400 / 1280)
    maxRadius = int(min((height, width)) * 1000 / 1280)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 0.5, 100, param1=100, param2=100, minRadius=minRadius,
                               maxRadius=maxRadius)
    final_circle = circles[0, 0]
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for count, current_circle in enumerate(circles[0, :]):
            # 중심에 가장 가까운 원 저장
            if abs(final_circle[0] - width / 2) > abs(current_circle[0] - width / 2) and abs(
                    final_circle[1] - height / 2) > abs(current_circle[1] - height / 2):
                final_circle = current_circle
    else:
        if find_manually:
            global points
            points = []
            while True:
                if cv2.waitKey(10) == 27:
                    break
                cv2.imshow('img', img)
                cv2.setMouseCallback('img', clicked, img)
            final_circle = select_scope()
        else:
            print('circles not detected')
            return None
    for item in final_circle:
        item = int(item)
    return final_circle


def select_scope():
    """
    get a circle from 3 points
    :return: coordinates, radius of circle
    """
    a, b, c = points
    X = np.array(points)
    Y = [1, 1]
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


def crop_scope(img, scope):
    halflen = int(scope[2] / 1.414 * 0.99)  # 스코프 내접 사각형 길이/2 (99%)
    x_min = int(scope[0] - halflen)
    y_min = int(scope[1] - halflen)
    x_max = int(scope[0] + halflen)
    y_max = int(scope[1] + halflen)
    cropped_img = img[y_min:y_max, x_min:x_max]
    return cropped_img


def get_cropped_img(img, find_manually):
    scope = find_scope(img, find_manually)
    cropped_img = crop_scope(img, scope)
    return cropped_img


def main():
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

    INPUT_PATH = args.input_path
    OUTPUT_PATH = args.output_path

    if args.verify:
        os.makedirs(SCOPE_PATH)
    os.makedirs(OUTPUT_PATH)

    file_list = glob.glob(INPUT_PATH + '/*.jpg')
    for file in file_list:
        filename = os.path.basename(file)
        img = cv2.imread(file)

        scope = find_scope(img, args.manual_roi)
        cropped_img = crop_scope(img, scope)
        cv2.imwrite(os.path.join(OUTPUT_PATH, filename), cropped_img)
        if args.verify:
            cv2.circle(img, (scope[0], scope[1]), scope[2], (0, 255, 0), 2)
            cv2.imwrite(os.path.join(SCOPE_PATH, filename), img)


if __name__ == "__main__":
    main()
