import cv2
import numpy as np
from matplotlib import pyplot as plt
import detectHorizonalLines as dhl
import detectVerticalLine as dvl
import eraseNoisy as en



def DetectCircle(origin):
    while cv2.getWindowProperty('SeekBar', cv2.WND_PROP_VISIBLE) >= 1:
        blurredSize = cv2.getTrackbarPos('blurredSize', 'SeekBar')
        cannyThreshold1 = cv2.getTrackbarPos('cannyThreshold1', 'SeekBar')
        cannyThreshold2 = cv2.getTrackbarPos('cannyThreshold2', 'SeekBar')
        distance = cv2.getTrackbarPos('distance', 'SeekBar')
        minRadius = cv2.getTrackbarPos('minRadius', 'SeekBar')
        maxRadius = cv2.getTrackbarPos('maxRadius', 'SeekBar')
        image = np.copy(origin)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if blurredSize % 2 == 0: blurredSize += 1
        blurred = cv2.GaussianBlur(gray, (blurredSize, blurredSize), 0)
        edges = cv2.Canny(blurred, cannyThreshold1, cannyThreshold2) #参数1为160，参数2为30
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1,
                           distance, param1=160, param2=30, minRadius=minRadius, maxRadius=maxRadius)
        if circles is not None:
            circles = np.round(circles[0, :]).astype(int)
            for (x, y, r) in circles:
                cv2.circle(image, (x, y), r, (0, 255, 0), 2)
                cv2.circle(image, (x, y), 2, (0, 255, 0), 2)
        cv2.imshow("Main", image)
        cv2.imshow("Main1", edges)

        if cv2.waitKey(1) == 27:
            break

def mean_filter(image, kernel_size):
    # 创建一个 kernel_size x kernel_size 的均值滤波器
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size ** 2)
    # 应用均值滤波器
    filtered_image = cv2.filter2D(image, -1, kernel)
    return filtered_image
def DetectRetangle(origin):
    while cv2.getWindowProperty('SeekBar', cv2.WND_PROP_VISIBLE) >= 1:
        blurredSize = cv2.getTrackbarPos('blurredSize', 'SeekBar')
        cannyThreshold1 = cv2.getTrackbarPos('cannyThreshold1', 'SeekBar')
        cannyThreshold2 = cv2.getTrackbarPos('cannyThreshold2', 'SeekBar')
        distance = cv2.getTrackbarPos('distance', 'SeekBar')
        minRadius = cv2.getTrackbarPos('minRadius', 'SeekBar')
        maxRadius = cv2.getTrackbarPos('maxRadius', 'SeekBar')
        lineThreshold = cv2.getTrackbarPos('lineThreshold', 'SeekBar')
        image = np.copy(origin)
        if blurredSize % 2 == 0: blurredSize += 1
        # blurred = cv2.GaussianBlur(image, (blurredSize, blurredSize), 0)
        blurred = mean_filter(image, 3)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, cannyThreshold1, cannyThreshold2)
        mask = en.eraseNoisy(edges, distance)
        cv2.imshow("Main", mask)
        if cv2.waitKey(1) == 27:
            break
def test(raw):
    order = 1
    while cv2.getWindowProperty('SeekBar', cv2.WND_PROP_VISIBLE) >= 1:
        blurredSize = cv2.getTrackbarPos('blurredSize', 'SeekBar')
        cannyThreshold1 = cv2.getTrackbarPos('cannyThreshold1', 'SeekBar')
        cannyThreshold2 = cv2.getTrackbarPos('cannyThreshold2', 'SeekBar')
        distance = cv2.getTrackbarPos('distance', 'SeekBar')
        minRadius = cv2.getTrackbarPos('minRadius', 'SeekBar')
        maxRadius = cv2.getTrackbarPos('maxRadius', 'SeekBar')
        lineThreshold = cv2.getTrackbarPos('lineThreshold', 'SeekBar')
        if blurredSize % 2 == 0: blurredSize += 1
        image = np.copy(raw)
        blurred = cv2.GaussianBlur(image, (blurredSize, blurredSize), 0)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        # 将HSV图像分割为三个通道
        hue, sat, val = cv2.split(hsv)
        hedges = cv2.Canny(val, cannyThreshold1, cannyThreshold2)
        key = cv2.waitKey(1) & 0xFF  # 等待键盘输入，取低8位
        if key == ord('1'): order = 1
        elif key == ord('2'): order = 2
        elif key == ord('3'): order = 3
        if order == 1:
            cv2.imshow('hue', hue)
            cv2.imshow('sat', sat)
            cv2.imshow('val', val)
            cv2.imshow('hedges', hedges)
        if cv2.waitKey(1) == 27: break

def TraditionWay(raw):
    while cv2.getWindowProperty('SeekBar', cv2.WND_PROP_VISIBLE) >= 1:
        image = np.copy(raw)
        blurredSize = cv2.getTrackbarPos('blurredSize', 'SeekBar')
        cannyThreshold1 = cv2.getTrackbarPos('cannyThreshold1', 'SeekBar')
        cannyThreshold2 = cv2.getTrackbarPos('cannyThreshold2', 'SeekBar')
        distance = cv2.getTrackbarPos('distance', 'SeekBar')
        minRadius = cv2.getTrackbarPos('minRadius', 'SeekBar')
        maxRadius = cv2.getTrackbarPos('maxRadius', 'SeekBar')
        lineThreshold = cv2.getTrackbarPos('lineThreshold', 'SeekBar')
        blurredSize = cv2.getTrackbarPos('blurredSize', 'SeekBar')
        if blurredSize % 2 == 0: blurredSize += 1
        blurred = cv2.GaussianBlur(image, (blurredSize, blurredSize), 0)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        canny = cv2.Canny(gray, cannyThreshold1, cannyThreshold2)
        cv2.imshow('canny', cv2.bitwise_not(canny))
        if cv2.waitKey(1) == 27: break
def main():
    origin = cv2.imread('black.jpg')
    raw = cv2.resize(origin, (1200, 800))
    # DetectCircle(raw)
    # DetectRetangle(raw)
    test(raw)
    # TraditionWay(raw)
    cv2.waitKey(0);