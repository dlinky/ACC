import cv2
import os

import get_roi

path_dir = os.getcwd() + '/manual/'
original_dir = path_dir + 'original/'
result_dir = path_dir + 'result/'

points = []


def clicked(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        save_point([x, y])


def save_point(point):
    global points
    points.append(point)


def select_scope():
    a = float(points[0][0])
    b = float(points[0][1])
    c = float(points[1][0])
    d = float(points[1][1])
    e = float(points[2][0])
    f = float(points[2][1])

    # 연립방정식 정리하면 y0 = mx + x0
    y0 = -(2*d-b-f)/2
    m = ((c-a)/(b-d) + (e-c)/(d-f) - 2*(e-a)/(b-f))
    x0 = -((c**2-a**2)/(b-d) + (e**2-c**2)/(d-f) - 2*(e**2-a**2)/(b-f))/2

    x = (y0-x0)/m
    y = (c-a)/(b-d)*(x-(a+c)/2)+(b+d)/2
    r = ((x-a)**2 + (y-b)**2)**(1/2)

    return [int(x), int(y), int(r)]


def find_scope_manually(img):
    while True:
        if cv2.waitKey(10) == 27:
            break
        cv2.imshow('img', img)
        cv2.setMouseCallback('img', clicked, img)
    final_circle = select_scope()
    return final_circle


def find_square_manually(img):
    circle = find_scope_manually(img)
    halflen = int(circle[2] / 1.414 * 0.99)  # 스코프 내접 사각형 길이/2 (99%)
    square = [circle[0] - halflen, circle[1] - halflen, circle[0] + halflen,
              circle[1] + halflen]
    return square


def find_roi_manually(img, mode):
    if mode == 'circle':
        return find_scope_manually(img)
    if mode == 'square':
        return find_square_manually(img)


def main():
    file_list = os.listdir(original_dir)
    img_list = []
    cv2.namedWindow('img')

    for i in file_list:
        img_list.append(cv2.imread(original_dir + i))

    mode = 'circle'
    for i, img in enumerate(img_list):
        roi = find_roi_manually(img, mode)
        masked = get_roi.process_roi(img, roi)
        cv2.imwrite(result_dir + file_list[i], masked)


if __name__ == '__main__':
    main()