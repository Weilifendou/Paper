import cv2
from MVGigE import *
import numpy as np
import detectHorizonalLines as dhl
import detectVerticalLine as dvl
import glob
import matplotlib.pyplot as plt


def OpenCam():
    r, hCam = MVOpenCamByIndex(0)  # 根据相机的索引返回相机句柄

    if hCam == 0:
        if r == MVST_ACCESS_DENIED:
            print('无法打开相机，可能正被别的软件控制!')
        else:
            print('无法打开相机!')
        return 0
    return hCam


def StartGrab(hCam):
    r, img = MVGetImgBuf(hCam)

    if img is None:
        print('error: ', r)
        MVCloseCam(hCam)
        return 0

    if MVStartGrabWindow(hCam) != MVST_SUCCESS:
        print("MVStartGrabWindow error")
        MVCloseCam(hCam)
        return 0

    return 1, img


def RealTime():
    hCam = OpenCam()
    if hCam == 0:
        quit()

    r, img = MVGetImgBuf(hCam)

    if img is None:
        print('error: ', r)
        MVCloseCam(hCam)
        return 0

    if MVStartGrabWindow(hCam) != MVST_SUCCESS:
        print("MVStartGrabWindow error")
        MVCloseCam(hCam)
        return 0
    while cv2.getWindowProperty('SeekBar', cv2.WND_PROP_VISIBLE) >= 1:
        res, id = MVGetSampleGrabBuf(hCam, img, 50)
        if res == MVST_SUCCESS:
            ProcessImage(img)
        if cv2.waitKey(1) == 27: break
    MVStopGrab(hCam)  # 停止采集
    MVCloseCam(hCam)
def ProcessImage(img):
        blurredSize = cv2.getTrackbarPos('blurredSize', 'SeekBar')
        cannyThreshold1 = cv2.getTrackbarPos('cannyThreshold1', 'SeekBar')
        cannyThreshold2 = cv2.getTrackbarPos('cannyThreshold2', 'SeekBar')
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
        # hedges = cv2.Canny(hue, cannyThreshold1, cannyThreshold2) #参数1为160，参数2为30
        # edges = cv2.Ca/nny(hue, 160, 30) #参数1为160，参数2为30
        # boardImage = DetectCircle(boardImage)
        # boardImage = DetectRetangle(boardImage)
        # boardImage = DetectCircle1(boardImage)
        # image[y:y + h, x:x + w] = boardImage
        # cv2.imshow('image', image)
        CutBoard(image)

def CutBoard(image):
    blurredSize = cv2.getTrackbarPos('blurredSize', 'SeekBar')
    cannyThreshold1 = cv2.getTrackbarPos('cannyThreshold1', 'SeekBar')
    cannyThreshold2 = cv2.getTrackbarPos('cannyThreshold2', 'SeekBar')
    distance = cv2.getTrackbarPos('distance', 'SeekBar')
    minRadius = cv2.getTrackbarPos('minRadius', 'SeekBar')
    maxRadius = cv2.getTrackbarPos('maxRadius', 'SeekBar')
    if blurredSize % 2 == 0: blurredSize += 1
    #高斯模糊核为11比较合适
    blurred = cv2.GaussianBlur(image, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    # 将HSV图像分割为三个通道
    hue, sat, val = cv2.split(hsv)
    #显示hsv三通道图像
    # cv2.imshow('hue', hue)
    # cv2.imshow('sat', sat)
    # cv2.imshow('val', val)
    _, binary = cv2.threshold(val, cannyThreshold1, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    edge = cv2.Canny(closed, 100, 50) #二值图像去边缘，基本上可以随意取值
    contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_contour)
    boardImage = image[y:y + h, x:x + w]
    cv2.imshow('binary', boardImage)
    return boardImage


def DetectCircle(image):
    blurred = cv2.GaussianBlur(image, (7, 7), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1,
                       30, param1=160, param2=30, minRadius=5, maxRadius=50)
    if circles is not None:
        circles = np.round(circles[0, :]).astype(int)
        for (x, y, r) in circles:
            cv2.circle(image, (x, y), r, (0, 255, 0), 2)
            cv2.circle(image, (x, y), 2, (0, 255, 0), 2)
    return image
def DetectCircle1(image):
    blurredSize = cv2.getTrackbarPos('blurredSize', 'SeekBar')
    cannyThreshold1 = cv2.getTrackbarPos('cannyThreshold1', 'SeekBar')
    cannyThreshold2 = cv2.getTrackbarPos('cannyThreshold2', 'SeekBar')
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
    cv2.imshow('hedges', hedges)
    circles = cv2.HoughCircles(opened, cv2.HOUGH_GRADIENT, 1, minDist=15,
                               param1=100, param2=10, minRadius=5, maxRadius=60)
    if circles is not None:
        circles = np.round(circles[0, :]).astype(int)
        for (x, y, r) in circles:
            cv2.circle(image, (x, y), r, (0, 255, 0), 2)
            cv2.circle(image, (x, y), 2, (0, 255, 0), 2)
    return image
def DetectRetangle(image):
    blurredSize = cv2.getTrackbarPos('blurredSize', 'SeekBar')
    cannyThreshold1 = cv2.getTrackbarPos('cannyThreshold1', 'SeekBar')
    cannyThreshold2 = cv2.getTrackbarPos('cannyThreshold2', 'SeekBar')
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
    hedges = cv2.Canny(opened, cannyThreshold1, cannyThreshold2)
    contours, _ = cv2.findContours(hedges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 遍历轮廓
    for contour in contours:
        # 计算轮廓的边界矩形
        x, y, w, h = cv2.boundingRect(contour)
        if w * h > 5000:
            # 绘制边界矩形
            cv2.rectangle(image, (x-2, y-2), (x + w + 2, y + h + 2), (0, 255, 0), 2)
    cv2.imshow('hedges', hedges)

    # lines = cv2.HoughLinesP(hedges, 1, np.pi / 180, distance, minLineLength=minRadius, maxLineGap=maxRadius)
    # if lines is not None:
    #     for line in lines:
    #         x1, y1, x2, y2 = line[0]
    #         cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return image
def Demarcation():
    # 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # 准备标定板
    pattern_size = (9, 6)  # 标定板内角点数目
    square_size = 1.0  # 标定板方格大小（单位：毫米）
    obj_points = []  # 世界坐标系中的点
    img_points = []  # 图像坐标系中检测到的点

    # 构建标定板的世界坐标系点
    objp = np.zeros((np.prod(pattern_size), 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size

    # 读取图像并提取角点
    image_paths = ['Dema1.png', 'Dema2.png', 'Dema3.png']  # 替换为实际图像路径
    for path in image_paths:
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

        if ret:
            obj_points.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), criteria)  # 在原角点的基础上寻找亚像素角点
        if [corners2]:
            img_points.append(corners2)
        else:
            img_points.append(corners)
        cv2.drawChessboardCorners(img, pattern_size, corners, ret)
        cv2.imshow(path, img)
        cv2.waitKey(500)


    # 进行相机标定
    ret, camera_matrix, distortion_coeffs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

    print("ret:", ret)
    print("mtx:\n", camera_matrix)  # 内参数矩阵
    print("dist:\n", distortion_coeffs)  # 畸变系数
    print("rvecs:\n", rvecs)  # 旋转向量 # 外参数
    print("tvecs:\n", tvecs)  # 平移向量 # 外参数
    cv2.waitKey(0)

def empty(a): pass
def drawTrackbar():
    # 使用Trackbar调参，效率提高一大截
    cv2.namedWindow('SeekBar')
    mat = np.ones((50,800)) #调节Trackbar显示的宽度
    cv2.imshow('SeekBar', mat)
    cv2.createTrackbar('blurredSize', 'SeekBar', 7, 255, empty)
    cv2.createTrackbar('cannyThreshold1', 'SeekBar', 100, 500, empty)
    cv2.createTrackbar('cannyThreshold2', 'SeekBar', 50, 500, empty)
    cv2.createTrackbar('distance', 'SeekBar', 10, 500, empty)
    cv2.createTrackbar('minRadius', 'SeekBar', 10, 500, empty)
    cv2.createTrackbar('maxRadius', 'SeekBar', 50, 500, empty)
    cv2.createTrackbar('lineThreshold', 'SeekBar', 100, 200, empty)

if __name__ == '__main__':
    drawTrackbar()
    print('hello world')
    raw = cv2.imread('1.jpg')
    raw = cv2.resize(raw, (1200, 800))
    RealTime()
    # DetectRetangle(raw)
    # Demarcation()