import cv2
import numpy as np
def CutBoard(image):
    blurredSize = cv2.getTrackbarPos('blurredSize', 'SeekBar')
    cannyThreshold1 = cv2.getTrackbarPos('cannyThreshold1', 'SeekBar')
    cannyThreshold2 = cv2.getTrackbarPos('cannyThreshold2', 'SeekBar')
    distance = cv2.getTrackbarPos('distance', 'SeekBar')
    minRadius = cv2.getTrackbarPos('minRadius', 'SeekBar')
    maxRadius = cv2.getTrackbarPos('maxRadius', 'SeekBar')
    if blurredSize % 2 == 0: blurredSize += 1
    #高斯模糊核为11比较合适
    blurred = cv2.GaussianBlur(image, (blurredSize, blurredSize), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    # 将HSV图像分割为三个通道
    hue, sat, val = cv2.split(hsv)
    #显示hsv三通道图像
    # cv2.imshow('hue', hue)
    # cv2.imshow('sat', sat)
    # cv2.imshow('val', val)
    _, binary = cv2.threshold(val, 10, 255, cv2.THRESH_BINARY) #根据实际情况调整
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    edge = cv2.Canny(closed, 100, 50) #二值图像去边缘，基本上可以随意取值
    contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_contour)
    boardImage = image[y:y + h, x:x + w]
    cv2.imshow('edge', boardImage)
    return boardImage