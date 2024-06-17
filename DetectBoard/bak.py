import os
import cv2
import numpy as np
from MVGigE import *
import CutBoard as cb
import Color as cr
def empty(a): pass
def drawTrackbar():
    # 使用Trackbar调参，效率提高一大截
    cv2.namedWindow('SeekBar')
    mat = np.ones((50, 800))  # 调节Trackbar显示的宽度
    cv2.imshow('SeekBar', mat)
    cv2.createTrackbar('blurredSize', 'SeekBar', 7, 255, empty)
    cv2.createTrackbar('threshold1', 'SeekBar', 10, 500, empty)
    cv2.createTrackbar('threshold2', 'SeekBar', 80, 500, empty)
    cv2.createTrackbar('distance', 'SeekBar', 10, 500, empty)
    cv2.createTrackbar('minRadius', 'SeekBar', 10, 500, empty)
    cv2.createTrackbar('maxRadius', 'SeekBar', 50, 500, empty)
    cv2.createTrackbar('lineThreshold', 'SeekBar', 100, 200, empty)

def ProcessImage(img):
    blurredSize = cv2.getTrackbarPos('blurredSize', 'SeekBar')
    threshold1 = cv2.getTrackbarPos('threshold1', 'SeekBar')
    threshold2 = cv2.getTrackbarPos('threshold2', 'SeekBar')
    distance = cv2.getTrackbarPos('distance', 'SeekBar')
    minRadius = cv2.getTrackbarPos('minRadius', 'SeekBar')
    maxRadius = cv2.getTrackbarPos('maxRadius', 'SeekBar')
    if blurredSize % 2 == 0: blurredSize += 1
    image = cv2.resize(img, (1200, 800))
    image = cv2.flip(image, -1)  # 参数为0上下翻转，1为左右翻转，-1为上下左右均翻转
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY)
    edge = cv2.Canny(gray, 20, 20)
    contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_contour)
    boardImage = image[y:y + h, x:x + w]
    # hsv = cv2.cvtColor(boardImage, cv2.COLOR_BGR2HSV)
    # 将HSV图像分割为三个通道
    # hue, sat, val = cv2.split(hsv)
    # hedges = cv2.Canny(hue, threshold1, threshold2) #参数1为160，参数2为30
    # edges = cv2.Ca/nny(hue, 160, 30) #参数1为160，参数2为30
    # boardImage = DetectCircle(boardImage)
    # boardImage = DetectRetangle(boardImage)
    # boardImage = DetectCircle1(boardImage)
    # image[y:y + h, x:x + w] = boardImage
    # cv2.imshow('image', image)
    cb.CutBoard(image)


# def DetectCircle(image):
#     blurred = cv2.GaussianBlur(image, (7, 7), 0)
#     gray = cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY)
#     circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1,
#                        30, param1=160, param2=30, minRadius=5, maxRadius=50)
#     if circles is not None:
#         circles = np.round(circles[0, :]).astype(int)
#         for (x, y, r) in circles:
#             cv2.circle(image, (x, y), r+1, (0, 255, 0), 1)
#             cv2.circle(image, (x, y), 2, (0, 255, 0), 2)
#     return image
def DetectCircle(image):
    blurredSize = cv2.getTrackbarPos('blurredSize', 'SeekBar')
    threshold1 = cv2.getTrackbarPos('threshold1', 'SeekBar')
    threshold2 = cv2.getTrackbarPos('threshold2', 'SeekBar')
    distance = cv2.getTrackbarPos('distance', 'SeekBar')
    minRadius = cv2.getTrackbarPos('minRadius', 'SeekBar')
    maxRadius = cv2.getTrackbarPos('maxRadius', 'SeekBar')
    if blurredSize % 2 == 0: blurredSize += 1
    blurred = cv2.GaussianBlur(image, (blurredSize, blurredSize), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    # 将HSV图像分割为三个通道
    hue, sat, val = cv2.split(hsv)
    _, binary = cv2.threshold(hue, 50, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    hedges = cv2.Canny(opened, 100, 50)
    circles = cv2.HoughCircles(opened, cv2.HOUGH_GRADIENT, 1, minDist=15,
                               param1=100, param2=10, minRadius=5, maxRadius=60)
    if circles is not None:
        circles = np.round(circles[0, :]).astype(int)
        for (x, y, r) in circles:
            cv2.circle(image, (x, y), r, (0, 255, 0), 2)
            cv2.circle(image, (x, y), 2, (0, 255, 0), 2)
    return image



# drawTrackbar()
def empty(a): pass

def drawTrackbar():
    # 使用Trackbar调参，效率提高一大截
    cv2.namedWindow('SeekBar')
    mat = np.ones((50,800)) #调节Trackbar显示的宽度
    cv2.imshow('SeekBar', mat)
    cv2.createTrackbar('blurredSize', 'SeekBar', 7, 255, empty)
    cv2.createTrackbar('threshold1', 'SeekBar', 10, 500, empty)
    cv2.createTrackbar('threshold2', 'SeekBar', 80, 500, empty)
    cv2.createTrackbar('distance', 'SeekBar', 10, 500, empty)
    cv2.createTrackbar('minRadius', 'SeekBar', 10, 500, empty)
    cv2.createTrackbar('maxRadius', 'SeekBar', 50, 500, empty)
    cv2.createTrackbar('lineThreshold', 'SeekBar', 100, 200, empty)


drawTrackbar()

def ProcessImage(img):
        blurredSize = cv2.getTrackbarPos('blurredSize', 'SeekBar')
        threshold1 = cv2.getTrackbarPos('threshold1', 'SeekBar')
        threshold2 = cv2.getTrackbarPos('threshold2', 'SeekBar')
        distance = cv2.getTrackbarPos('distance', 'SeekBar')
        minRadius = cv2.getTrackbarPos('minRadius', 'SeekBar')
        maxRadius = cv2.getTrackbarPos('maxRadius', 'SeekBar')
        if blurredSize % 2 == 0: blurredSize += 1
        image = cv2.resize(img, (1200, 800))
        image = cv2.flip(image, -1)  # 参数为0上下翻转，1为左右翻转，-1为上下左右均翻转
        blurred = cv2.GaussianBlur(image, (3, 3), 0)
        gray = cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY)
        edge = cv2.Canny(gray, 20, 20)
        contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)
        boardImage = image[y:y + h, x:x + w]
        # hsv = cv2.cvtColor(boardImage, cv2.COLOR_BGR2HSV)
        # 将HSV图像分割为三个通道
        # hue, sat, val = cv2.split(hsv)
        # hedges = cv2.Canny(hue, threshold1, threshold2) #参数1为160，参数2为30
        # edges = cv2.Ca/nny(hue, 160, 30) #参数1为160，参数2为30
        # boardImage = DetectCircle(boardImage)
        # boardImage = DetectRetangle(boardImage)
        # boardImage = DetectCircle1(boardImage)
        # image[y:y + h, x:x + w] = boardImage
        # cv2.imshow('image', image)
        cb.CutBoard(image)

# def DetectCircle(image):
#     blurred = cv2.GaussianBlur(image, (7, 7), 0)
#     gray = cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY)
#     circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1,
#                        30, param1=160, param2=30, minRadius=5, maxRadius=50)
#     if circles is not None:
#         circles = np.round(circles[0, :]).astype(int)
#         for (x, y, r) in circles:
#             cv2.circle(image, (x, y), r+1, (0, 255, 0), 1)
#             cv2.circle(image, (x, y), 2, (0, 255, 0), 2)
#     return image
def DetectCircle(image):
    blurredSize = cv2.getTrackbarPos('blurredSize', 'SeekBar')
    threshold1 = cv2.getTrackbarPos('threshold1', 'SeekBar')
    threshold2 = cv2.getTrackbarPos('threshold2', 'SeekBar')
    distance = cv2.getTrackbarPos('distance', 'SeekBar')
    minRadius = cv2.getTrackbarPos('minRadius', 'SeekBar')
    maxRadius = cv2.getTrackbarPos('maxRadius', 'SeekBar')
    if blurredSize % 2 == 0: blurredSize += 1
    blurred = cv2.GaussianBlur(image, (blurredSize, blurredSize), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    # 将HSV图像分割为三个通道
    hue, sat, val = cv2.split(hsv)
    _, binary = cv2.threshold(hue, 50, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    hedges = cv2.Canny(opened, 100, 50)
    circles = cv2.HoughCircles(opened, cv2.HOUGH_GRADIENT, 1, minDist=15,
                               param1=100, param2=10, minRadius=5, maxRadius=60)
    if circles is not None:
        circles = np.round(circles[0, :]).astype(int)
        for (x, y, r) in circles:
            cv2.circle(image, (x, y), r, (0, 255, 0), 2)
            cv2.circle(image, (x, y), 2, (0, 255, 0), 2)
    return image

def DetectShape(image):
    blurredSize = cv2.getTrackbarPos('blurredSize', 'SeekBar')
    threshold1 = cv2.getTrackbarPos('threshold1', 'SeekBar')
    threshold2 = cv2.getTrackbarPos('threshold2', 'SeekBar')
    distance = cv2.getTrackbarPos('distance', 'SeekBar')
    minRadius = cv2.getTrackbarPos('minRadius', 'SeekBar')
    maxRadius = cv2.getTrackbarPos('maxRadius', 'SeekBar')
    if blurredSize % 2 == 0: blurredSize += 1
    blurred = cv2.GaussianBlur(image, (blurredSize, blurredSize), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    # 将HSV图像分割为三个通道
    hue, sat, val = cv2.split(hsv)
    _, binary = cv2.threshold(hue, threshold2, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    hedges = cv2.Canny(opened, 50, 100)
    lines = cv2.HoughLinesP(hedges, 1, np.pi/180, threshold=threshold1,  minLineLength=minRadius, maxLineGap=maxRadius)
    # linImage = np.ones_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # cv2.line(linImage, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # cv2.imshow('ss',linImage)
    # 遍历轮廓
    contours, hierarchy = cv2.findContours(hedges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > 10:
            x, y, w, h = cv2.boundingRect(contour)
            if np.abs(w-h) < 5:
                x += int(w/2)
                y += int(h/2)
                r = int((w+h)/4+1)
                # cv2.circle(image, (x, y), r, cr.Black, -1)
                cv2.circle(image, (x, y), r, cr.Green, 2)
                # cv2.circle(image, (x, y), 2, cr.Green, 2)
                # text = "x=%d"%x+",y=%d"%y+",r=%d"%r
                # cv2.putText(image, text, (x-r-25, y-r-3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cr.Blue, 1)
            else:
                x -= 1
                y -= 1
                w += 1
                h += 1
                # cv2.rectangle(image, (x, y), (x+w, y+h), cr.Black, -1)
                cv2.rectangle(image, (x, y), (x+w, y+h), cr.Red, 2)
                # text = "x=%d"%x+",y=%d"%y+",w=%d"%w+",h=%d"%h
                # cv2.putText(image, text, (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cr.Blue, 1)

    cv2.imshow('imae', image)
    return image



def StaticProcess(raw):
    while cv2.getWindowProperty('SeekBar', cv2.WND_PROP_VISIBLE) >= 1:
        image = np.copy(raw)
        blurredSize = cv2.getTrackbarPos('blurredSize', 'SeekBar')
        threshold1 = cv2.getTrackbarPos('threshold1', 'SeekBar')
        threshold2 = cv2.getTrackbarPos('threshold2', 'SeekBar')
        distance = cv2.getTrackbarPos('distance', 'SeekBar')
        minRadius = cv2.getTrackbarPos('minRadius', 'SeekBar')
        maxRadius = cv2.getTrackbarPos('maxRadius', 'SeekBar')
        if blurredSize % 2 == 0: blurredSize += 1
        # 高斯模糊核为11比较合适
        blurred = cv2.GaussianBlur(image, (blurredSize, blurredSize), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        # 将HSV图像分割为三个通道
        hue, sat, val = cv2.split(hsv)
        # 显示hsv三通道图像
        # cv2.imshow('hue', hue)
        # cv2.imshow('sat', sat)
        # cv2.imshow('val', val)
        _, binary = cv2.threshold(val, threshold1, 255, cv2.THRESH_BINARY)  # 根据实际情况调整
        kernel = np.ones((5, 5), np.uint8)
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        edge = cv2.Canny(closed, 100, 50)  # 二值图像去边缘，基本上可以随意取值
        contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)
        bi = image[y:y + h, x:x + w]
        bi = DetectShape(bi)
        image[y:y + h, x:x + w] = bi
        # cv2.rectangle(image, (x, y), (x+w, y+h), cr.MedSpringGreen, 2)
        text = "w=%d"%w+",h=%d"%h
        # cv2.putText(image, text, (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cr.Wheat, 1)
        cv2.imshow('image', image)
        if cv2.waitKey(1) == 27:
            break



if __name__ == '__main__':
    print('hello world')
    path = os.path.join(os.getcwd(), 'Pictures', '3.jpg')
    print(path)  # 获得当前工作目录
    raw = cv2.imread(path)
    raw = cv2.resize(raw, (1200, 800))
    StaticProcess(raw)
    cv2.waitKey(0)
