import cv2
import numpy as np


def empty(a):
    pass


# 使用Trackbar调参，效率提高一大截
cv2.namedWindow('param')
cv2.createTrackbar('thresh1', 'param', 150, 255, empty)  # cv2.Canny参数
cv2.createTrackbar('thresh2', 'param', 255, 255, empty)  # cv2.Canny参数
cv2.createTrackbar('area', 'param', 20, 50000, empty)  # cv2.contourArea参数
cv2.createTrackbar('param1', 'param', 1, 100, empty)  # cv2.HoughCircles参数
cv2.createTrackbar('param2', 'param', 1, 100, empty)  # cv2.HoughCircles参数


if __name__ == '__main__':
    print('hello world')
    cv2.waitKey(0)