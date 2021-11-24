import os
import cv2
import numpy as np

import manual_roi


def create_folder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)


def print_little_image(filename, img):  # 디버깅용 임시로 작은 이미지 출력
    cv2.imshow(filename, cv2.resize(img, (500, 500)))
    cv2.waitKey(0)
    cv2.destroyWindow(filename)


def draw_circles(filename, img, circles, final_circle):
    temp = img.copy()
    for circle in circles:
        cv2.circle(temp, (circle[0], circle[1]), circle[2], (0, 255, 0))
    cv2.circle(temp, (final_circle[0], final_circle[1]), final_circle[2], (255, 0, 0), 5)
    print_little_image(filename, temp)


# 스코프 원 검출
def find_roi_scope(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 0.5, 100, param1=100, param2=100, minRadius=400, maxRadius=1000)
    if circles is not None:
        circles = np.uint16(np.around(circles))

        # 좌표가 바뀐?
        final_circle = circles[0, 0]
        height, width = gray.shape[:]
        for count, current_circle in enumerate(circles[0, :]):
            # 중심에 가장 가까운 원 저장
            if abs(final_circle[0] - width / 2) > abs(current_circle[0] - width / 2) and abs(final_circle[1] - height / 2) > abs(current_circle[1] - height / 2):
                final_circle = current_circle
            return final_circle  # (x, y, radius)
    else:
        return [0, 0, 0]


def find_roi_square(img):
    circle = find_roi_scope(img)
    halflen = int(circle[2] / 1.414 * 0.99)  # 스코프 내접 사각형 길이/2 (99%)
    square = [circle[0] - halflen, circle[1] - halflen, circle[0] + halflen,
              circle[1] + halflen]
    return square


def find_roi(img, mode):
    if mode == 'circle':
        return find_roi_scope(img)
    elif mode == 'square':
        return find_roi_square(img)
    else:
        return 0


def remove_scope(image, roi):
    mask = np.zeros_like(image)
    mask = cv2.circle(mask, (roi[0], roi[1]), int(roi[2] * 0.95), (255, 255, 255), -1)
    masked = cv2.bitwise_and(image, mask)
    return masked


def crop_square(image, roi):
    xmin, ymin, xmax, ymax = roi  # 좌표 수정해야됨. row랑 x랑 자꾸 헷갈림...
    masked = image[ymin:ymax, xmin:xmax]
    return masked


def process_roi(img, roi):
    if len(roi) == 3:
        return remove_scope(img, roi)
    elif len(roi) == 4:
        return crop_square(img, roi)


def main():
    pathDir = os.getcwd() + '/get_roi/'
    originalDir = pathDir + 'original/'
    circleDir = pathDir + 'scope/'
    create_folder(circleDir)
    squareDir = pathDir + 'square/'
    create_folder(squareDir)

    switch = 'square'
    test = 1

    manual_list = ['1-6.jpg']

    file_list = os.listdir(originalDir)
    img_list = []
    for img in file_list:
        img_list.append(cv2.imread(originalDir + img))

    for i, img in enumerate(img_list):
        if file_list[i] in manual_list:
            roi = manual_roi.find_roi_manually(img, switch)
        else:
            roi = find_roi(img, switch)
        masked = process_roi(img, roi)
        if switch == 'circle':
            cv2.imwrite(circleDir + file_list[i], masked)
        elif switch == 'square':
            cv2.imwrite(squareDir + file_list[i], masked)


if __name__ == '__main__':
    main()