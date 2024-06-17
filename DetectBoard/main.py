import os
import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5 import uic, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog

from Color import *
from DiffColor import StasticColor
from Kmean import SustainKmean, kmeans, kmeans_plusplus


def RemoveHoleAndSlot(i, blurredSize, threshold):
    image = np.copy(i)
    if blurredSize % 2 == 0:
        blurredSize += 1
    blurred = cv2.GaussianBlur(image, (blurredSize, blurredSize), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    # 将HSV图像分割为三个通道
    hue, sat, val = cv2.split(hsv)
    _, binary = cv2.threshold(hue, threshold, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    hedges = cv2.Canny(opened, 50, 100)
    # 遍历轮廓
    contours, hierarchy = cv2.findContours(hedges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > 10:
            x, y, w, h = cv2.boundingRect(contour)
            if np.abs(w - h) < 10:
                x += int(w / 2)
                y += int(h / 2)
                r = int((w + h) / 4 + 1)
                cv2.circle(image, (x, y), r, Black, -1)
                # cv2.circle(image, (x, y), r, cr.Green, 2)
                # cv2.circle(image, (x, y), 2, cr.Green, 2)
                # text = "x=%d"%x+",y=%d"%y+",r=%d"%r
                # cv2.putText(image, text, (x-r-25, y-r-3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cr.Blue, 1)
            else:
                x -= 1
                y -= 1
                w += 1
                h += 1
                cv2.rectangle(image, (x, y), (x + w, y + h), Black, -1)
                # cv2.rectangle(image, (x, y), (x+w, y+h), cr.Red, 2)
                # text = "x=%d"%x+",y=%d"%y+",w=%d"%w+",h=%d"%h
                # cv2.putText(image, text, (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cr.Blue, 1)
    return image


def DetectHoleAndSlot(i, blurredSize, threshold):
    image = np.copy(i)
    h, w, c = image.shape
    if blurredSize % 2 == 0:
        blurredSize += 1
    blurred = cv2.GaussianBlur(image, (blurredSize, blurredSize), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    # 将HSV图像分割为三个通道
    hue, sat, val = cv2.split(hsv)
    _, binary = cv2.threshold(hue, threshold, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    hedges = cv2.Canny(opened, 50, 100)
    # 遍历轮廓
    contours, hierarchy = cv2.findContours(hedges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cc = 0
    rc = 0
    pf = PixelFactor
    h *= pf
    w *= pf
    result = ''
    result += f'该板材长为{w:.3f},宽为{h:.3f}\n'
    for contour in contours:
        if cv2.contourArea(contour) > 50:
            x, y, w, h = cv2.boundingRect(contour)
            if np.abs(w - h) < 10:
                cc += 1
                x += int(w / 2)
                y += int(h / 2)
                r = int((w + h) / 4 + 1)
                r += 2
                cv2.circle(image, (x, y), r, Green, 10)
                # cv2.circle(image, (x, y), 5, Green, 5)
                x *= pf
                y *= pf
                r *= pf
                text = f'(x,y,r): ({x:.3f},{y:.3f},{r:.3f})mm'
                # cv2.putText(image, text, (x-r-25, y-r-3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, Blue, 1)
                result += f'圆{cc}的坐标(x,y)为({x:.3f},{y:.3f})mm,其半径为{r:.3f}mm\n'
            else:
                rc += 1
                x -= 1
                y -= 1
                w += 1
                h += 1
                # cv2.rectangle(image, (x, y), (x + w, y + h), Black, -1)
                cv2.rectangle(image, (x, y), (x + w, y + h), Red, 10)

                x *= pf
                y *= pf
                w *= pf
                h *= pf
                # text = f'(x,y,w,h): ({x:.3f},{y:.3f},{w:.3f},{h:.3f})mm'
                # cv2.putText(image, text, (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, Blue, 1)
                result += f'矩形{rc}的坐标(x,y)为({x:.3f},{y:.3f})mm,其长宽为({w:.3f},{h:.3f})mm\n'
    window.textEdits.setText(result)
    return image


def CutPicture(image, blurredSize, binThreshold):
    if blurredSize % 2 == 0:
        blurredSize += 1
    # 高斯模糊核为11比较合适
    blurred = cv2.GaussianBlur(image, (blurredSize, blurredSize), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    # 将HSV图像分割为三个通道
    hue, sat, val = cv2.split(hsv)
    # 显示hsv三通道图像
    # cv2.imshow('hue', hue)
    # cv2.imshow('sat', sat)
    # cv2.imshow('val', val)
    _, binary = cv2.threshold(val, binThreshold, 255, cv2.THRESH_BINARY)  # 根据实际情况调整
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    edge = cv2.Canny(closed, 100, 50)  # 二值图像去边缘，基本上可以随意取值
    contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)
        focusedImage = image[y:y + h, x:x + w]
        return focusedImage
    else:
        return image


def OpenCamera(): #打开电脑自带的摄像头
    global OriginalImage, IsRealTime
    cap = cv2.VideoCapture(0)  # 参数0表示默认摄像头，如果有多个摄像头可以尝试不同的参数

    # 检查摄像头是否成功打开
    if not cap.isOpened():
        print("无法打开摄像头")
        exit()
    if not IsRealTime:
        IsRealTime = True
        window.openCamera.setText('关闭相机')
        # 循环读取摄像头视频流
        while IsRealTime:
            # 读取视频流的帧
            ret, frame = cap.read()

            # 检查帧是否成功读取
            if not ret:
                print("无法获取帧")
                break

            # 显示帧
            # cv2.imshow("Camera", frame)
            # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            OriginalImage = cv2.flip(frame, 1)
            Scale()
            ShowImage(ScaleImage, window.label)
            # 按下 'q' 键退出循环
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # 释放摄像头资源
        cap.release()
    else:
        IsRealTime = False
        window.openCamera.setText('打开相机')


def TakePhoto():
    global IsRealTime
    IsRealTime = False


def OpenPicture():
    global OriginalImage, IsRealTime, RawImage
    IsRealTime = False
    file_dialog = QFileDialog()
    file_dialog.setFileMode(QFileDialog.AnyFile)
    file_dialog.setAcceptMode(QFileDialog.AcceptOpen)

    # 设置文件类型筛选器
    file_dialog.setNameFilters(['Picture Files (*.jpg *.png *.bmp)'])

    if file_dialog.exec_() == QFileDialog.Accepted:
        # 用户点击了确定按钮
        selected_file = file_dialog.selectedFiles()[0]
        OriginalImage = cv2.imread(selected_file)
        # OriginalImage = cv2.resize(OriginalImage, (1200, 800))
        RawImage = np.copy(OriginalImage)
        FitArea(OriginalImage, window.label)
    else:
        # 用户点击了取消按钮
        print("取消")


def SavePicture():
    global OriginalImage, IsRealTime
    IsRealTime = False
    fileDialog = QtWidgets.QFileDialog()
    fileName, _ = fileDialog.getSaveFileName(window, '保存图片文件', os.getcwd(),
                                             'Picture File(*.jpg *.png)')

    OriginalImage = cv2.imwrite(fileName, OriginalImage)


def ScaleValueChanged():
    global OriginalImage, Ratio
    Ratio = window.scaleSlider.value() / 100
    window.scaleSlider.setValue(int(Ratio * 100))
    window.ratioLabel.setText(f'Ratio：{Ratio:.2f}')
    x = window.label.width() / 2
    y = window.label.height() / 2
    ShowScaleImage(x, y)
    # OriginalImage = DetectHoleAndSlot(Board, para)
    # FitArea(OriginalImage, window.label)


def RemoveShape():
    global OriginalImage, FocusedImage, NoShapeBoard, IsMoveShape
    image = RemoveHoleAndSlot(FocusedImage, 50, 100)
    # FocusedImage = np.zeros((edge.shape[0], edge.shape[1], 3), dtype=np.uint8)
    # FocusedImage[:, :, 0] = FocusedImage  # 将边缘图像复制到蓝色通道
    # FocusedImage[:, :, 1] = FocusedImage  # 将边缘图像复制到绿色通道
    # FocusedImage[:, :, 2] = FocusedImage  # 将边缘图像复制到红色通道

    NoShapeBoard = np.copy(image)
    OriginalImage = np.copy(image)
    Scale()
    ShowImage(ScaleImage, window.label)
    IsMoveShape = True


def ExtractColor():
    global NoShapeBoard
    h, w, c = ScaleImage.shape
    h = int(h * 0.05)
    w = int(w * 0.05)
    NoShapeBoard = cv2.resize(NoShapeBoard, (w, h))
    pixels = NoShapeBoard.reshape(-1, 3)
    X = np.array(StasticColor(NoShapeBoard))
    x = X[:, 0]
    SustainKmean(x, 5, 20)
    K = 10
    initialCenters = kmeans_plusplus(x, K)
    #     # 使用一维K-means算法进行聚类
    final_centers, cluster_labels = kmeans(x, K, initialCenters)
    max = np.argmax(np.bincount(cluster_labels))
    maxIndex = np.where(cluster_labels == max)[0]
    x = X[:, 1:4][maxIndex, :]
    h = window.colorResult.height()
    w = window.colorResult.width()
    mean = np.mean(np.array(pixels), axis=0)
    f = np.mean(x, axis=0)
    resultLab = np.full((h, w, 3), mean, dtype=np.uint8)
    # resultRgb = cv2.cvtColor(mean, cv2.COLOR_LAB2RGB)
    # resultRgb = cv2.resize(resultRgb, (w, h))
    resultRgb = cv2.resize(resultLab, (w, h))
    ShowImage(resultRgb, window.colorResult)


def mousePressEvent(event):
    global InitialX, InitialY
    InitialX = event.x()
    InitialY = event.y()
    window.label.setCursor(Qt.ClosedHandCursor)
    # window.info.setText(f'Initial position: ({ix}, {iy})')


def mouseReleaseEvent(event):
    window.label.setCursor(Qt.ArrowCursor)


def mouseMoveEvent(event):
    global InitialX, InitialY, CutX, CutY
    CutX -= event.x() - InitialX
    CutY -= event.y() - InitialY
    InitialX = event.x()
    InitialY = event.y()
    ShowImage(ScaleImage, window.label)


def wheelEvent(event):
    global Ratio
    delta = event.angleDelta().y() / 120
    if delta > 0:
        Ratio += 0.05
        if Ratio > 3:
            Ratio = 3
    else:
        Ratio -= 0.05
        if Ratio < 0.05:
            Ratio = 0.05
    window.scaleSlider.setValue(int(Ratio * 100))
    window.ratioLabel.setText(f'Ratio：{Ratio:.2f}')
    x = event.x()
    y = event.y()
    ShowScaleImage(x, y)


def Scale():
    global ScaleImage
    h, w, c = OriginalImage.shape
    h = int(h * Ratio)
    w = int(w * Ratio)
    ScaleImage = cv2.resize(OriginalImage, (w, h))


def ShowScaleImage(sx, sy):
    global CutX, CutY
    h, w, c = ScaleImage.shape
    xr = (CutX + sx) / w
    yr = (CutY + sy) / h
    Scale()
    h, w, c = ScaleImage.shape
    CutX = int(w * xr - sx)
    CutY = int(h * yr - sy)
    ShowImage(ScaleImage, window.label)


def FitArea(image, label):
    global Ratio, ScaleImage
    hi, wi, c = image.shape
    hl = label.height()
    wl = label.width()
    if hi > wi:
        r = hl / hi
        hi = hl
        wi = int(wi * r)
        if wi > wl:
            r = wl / wi
            wi = wl
            hi = int(hi * r)
    else:
        r = wl / wi
        wi = wl
        hi = int(hi * r)
        if hi > hl:
            r = hl / hi
            hi = hl
            wi = int(wi * r)

    ScaleImage = cv2.resize(image, (wi, hi))
    ho, wo, c = OriginalImage.shape
    Ratio = (wi / wo + hi / ho) / 2
    window.scaleSlider.setValue(int(Ratio * 100))
    ShowImage(ScaleImage, label)


def FillArea(image, label):
    global CutX, CutY, Ratio, ScaleImage
    hi, wi, c = image.shape
    hl = label.height()
    wl = label.width()
    if hi > wi:
        r = wl / wi
        wi = wl
        hi = int(hi * r)
        if hi < hl:
            r = hl / hi
            hi = hl
            wi = int(wi * r)
    else:
        r = hl / hi
        hi = hl
        wi = int(wi * r)
        if wi < wl:
            r = wl / wi
            wi = wl
            hi = int(hi * r)
    ScaleImage = cv2.resize(image, (wi, hi))
    ho, wo, c = OriginalImage.shape
    Ratio = (wi / wo + hi / ho) / 2
    # window.scaleSlider.setValue(int(Ratio * 100))
    CutY = (hi - hl) / 2
    CutY = int(CutY)
    CutX = (wi - wl) / 2
    CutX = int(CutX)
    ShowImage(ScaleImage, label)


def ShowImage(image, label):
    global CutX, CutY
    hi, wi, c = image.shape
    hl = label.height()
    wl = label.width()
    h = hi
    w = wi
    if hi > hl:
        h = hl
        if CutY < 0:
            CutY = 0
        if CutY > hi - hl:
            CutY = hi - hl
        image = image[CutY:CutY + hl, 0:wi]

    if wi > wl:
        w = wl
        if CutX < 0:
            CutX = 0
        if CutX > wi - wl:
            CutX = wi - wl
        image = image[0:hi, CutX:CutX + wl]

    image = cv2.resize(image, (w, h))
    q_image = QtGui.QImage(image.data, w, h, 3 * w, QtGui.QImage.Format_RGB888).rgbSwapped()
    # # 将Qt图像设置到QLabel中
    pixmap = QtGui.QPixmap.fromImage(q_image)
    label.setPixmap(pixmap)


def onFitArea():
    FitArea(OriginalImage, window.label)


def onFillArea():
    FillArea(OriginalImage, window.label)


def onScaleUp():
    global Ratio
    Ratio += 0.05
    if Ratio > 3:
        Ratio = 3
    window.scaleSlider.setValue(int(Ratio * 100))
    window.ratioLabel.setText(f'Ratio：{Ratio:.2f}')
    x = window.label.width() / 2
    y = window.label.height() / 2
    ShowScaleImage(x, y)


def onScaleDown():
    global Ratio
    Ratio -= 0.05
    if Ratio < 0.05:
        Ratio = 0.05
    window.scaleSlider.setValue(int(Ratio * 100))
    window.ratioLabel.setText(f'Ratio：{Ratio:.2f}')
    x = window.label.width() / 2
    y = window.label.height() / 2
    ShowScaleImage(x, y)


def onCutPicture():
    global FocusedImage, OriginalImage, IsMoveShape
    FocusedImage = CutPicture(RawImage, 15, 15)

    OriginalImage = np.copy(FocusedImage)
    FitArea(OriginalImage, window.label)
    IsMoveShape = False

def CutBlurredValueChanged():
    global OriginalImage, FocusedImage
    blurredSize = window.cutBlurredSlider.value()
    threshold = window.cutThresholdSlider.value()
    FocusedImage = CutPicture(RawImage, blurredSize, threshold)
    # FocusedImage = np.zeros((edge.shape[0], edge.shape[1], 3), dtype=np.uint8)
    # FocusedImage[:, :, 0] = FocusedImage  # 将边缘图像复制到蓝色通道
    # FocusedImage[:, :, 1] = FocusedImage  # 将边缘图像复制到绿色通道
    # FocusedImage[:, :, 2] = FocusedImage  # 将边缘图像复制到红色通道

    OriginalImage = np.copy(FocusedImage)
    Scale()
    ShowImage(ScaleImage, window.label)
    window.cutBlurredLabel.setText(f'CutBlurSize：{blurredSize}')
    window.cutThresholdLabel.setText(f'CutBinThresh：{threshold}')


def CutThresholdValueChanged():
    global OriginalImage, FocusedImage
    blurredSize = window.cutBlurredSlider.value()
    threshold = window.cutThresholdSlider.value()
    FocusedImage = CutPicture(RawImage, blurredSize, threshold)
    # FocusedImage = np.zeros((edge.shape[0], edge.shape[1], 3), dtype=np.uint8)
    # FocusedImage[:, :, 0] = FocusedImage  # 将边缘图像复制到蓝色通道
    # FocusedImage[:, :, 1] = FocusedImage  # 将边缘图像复制到绿色通道
    # FocusedImage[:, :, 2] = FocusedImage  # 将边缘图像复制到红色通道
    # edge = cv2.resize(edge, (1200, 800))
    # cv2.imshow('FocusedImage', edge)

    OriginalImage = np.copy(FocusedImage)
    Scale()
    ShowImage(ScaleImage, window.label)
    window.cutBlurredLabel.setText(f'CutBlurSize：{blurredSize}')
    window.cutThresholdLabel.setText(f'CutBinThresh：{threshold}')


def MeaBlurredValueChanged():
    global OriginalImage, FocusedImage, NoShapeBoard
    blurredSize = window.meaBlurredSlider.value()
    threshold = window.meaThresholdSlider.value()
    if IsMoveShape:
        image = RemoveHoleAndSlot(FocusedImage, blurredSize, threshold)
    else:
        image = DetectHoleAndSlot(FocusedImage, blurredSize, threshold)
    # FocusedImage = np.zeros((edge.shape[0], edge.shape[1], 3), dtype=np.uint8)
    # FocusedImage[:, :, 0] = FocusedImage  # 将边缘图像复制到蓝色通道
    # FocusedImage[:, :, 1] = FocusedImage  # 将边缘图像复制到绿色通道
    # FocusedImage[:, :, 2] = FocusedImage  # 将边缘图像复制到红色通道

    OriginalImage = np.copy(image)
    NoShapeBoard = np.copy(image)
    Scale()
    ShowImage(ScaleImage, window.label)
    window.meaBlurredLabel.setText(f'MeasureBlurSize：{blurredSize}')
    window.meaThresholdLabel.setText(f'MeasureBinThresh：{threshold}')


def MeaThresholdValueChanged():
    global OriginalImage, FocusedImage, NoShapeBoard
    blurredSize = window.meaBlurredSlider.value()
    threshold = window.meaThresholdSlider.value()
    if IsMoveShape:
        image = RemoveHoleAndSlot(FocusedImage, blurredSize, threshold)
    else:
        image = DetectHoleAndSlot(FocusedImage, blurredSize, threshold)
    # FocusedImage = np.zeros((edge.shape[0], edge.shape[1], 3), dtype=np.uint8)
    # FocusedImage[:, :, 0] = FocusedImage  # 将边缘图像复制到蓝色通道
    # FocusedImage[:, :, 1] = FocusedImage  # 将边缘图像复制到绿色通道
    # FocusedImage[:, :, 2] = FocusedImage  # 将边缘图像复制到红色通道

    OriginalImage = np.copy(image)
    NoShapeBoard = np.copy(image)
    Scale()
    ShowImage(ScaleImage, window.label)
    window.meaBlurredLabel.setText(f'MeasureBlurSize：{blurredSize}')
    window.meaThresholdLabel.setText(f'MeasureBinThresh：{threshold}')


def onCalibration():
    global K, DisCoe, PixelFactor
    K, DisCoe, PixelFactor = Calibration()
    window.pixelFatorLabel.setText(f'PixelFator：{PixelFactor:.6f}mm/pi')


def Calibration():
    global OriginalImage, RawImage
    realDist = 51.3 / 9
    patternSize = (8, 8)
    objp = np.zeros((np.prod(patternSize), 3), np.float32)
    objp[:, :2] = np.mgrid[0:patternSize[0], 0:patternSize[1]].T.reshape(-1, 2)
    objPoints = []  # 实际世界坐标
    imgPoints = []  # 图像像素坐标
    imagePaths = []  # 替换为实际图像路径
    folder = os.path.join(os.getcwd(), 'Calibration')
    files = os.listdir(folder)

    for file in files:
        filePath = os.path.join(folder, file)
        if os.path.isfile(filePath):
            imagePaths.append(filePath)

    for imagePath in imagePaths:
        image = cv2.imread(imagePath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, patternSize, None)
        if ret:
            objPoints.append(objp)
            imgPoints.append(corners)
            cv2.drawChessboardCorners(image, patternSize, corners, ret)
            OriginalImage = image.copy()
            Scale()
            ShowImage(ScaleImage, window.label)
            cv2.waitKey(1)
        # 计算角点之间的像素距离
    distance = []

    row = patternSize[0]
    for i in range(len(imgPoints)):
        corners = imgPoints[i]
        j = 0
        while j < corners.shape[0]:
            x1, y1 = corners[j].ravel()
            j += row - 1
            x2, y2 = corners[j].ravel()
            dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            distance.append(dist)
            j += 1
    # print(imgPoints)
    averDist = np.mean(distance)
    pixelFatcor = realDist * (row - 1) / averDist
    # print(f'像素尺寸因子F ：\n{pixelFatcor}')
    #
    # # 进行相机标定
    ret, K, disCoe, rvecs, tvecs = cv2.calibrateCamera(objPoints, imgPoints, gray.shape[::-1], None, None)
    #
    # # 打印标定结果
    print("内参矩阵 K:\n", K)
    print("畸变参数 dist_coeffs:\n", disCoe)
    print("旋转向量 rvecs:\n", rvecs)
    print("平移向量 tvecs:\n", tvecs)
    # #
    # path = os.path.join(os.getcwd(), 'Pictures', '0001.jpg')
    # print(path)  # 获得当前工作目录
    # raw = cv2.imread(path)
    # undisImage = cv2.undistort(raw, K, disCoe)
    # OriginalImage = undisImage.copy()
    # RawImage = undisImage.copy()
    # Scale()
    # ShowImage(ScaleImage, window.label)
    return K, disCoe, pixelFatcor


def Undistort():
    global OriginalImage, RawImage
    if DisCoe is not None:
        RawImage = cv2.undistort(RawImage, K, DisCoe)
        OriginalImage = RawImage.copy()
        FitArea(OriginalImage, window.label)


def Measure():
    global FocusedImage, OriginalImage
    FocusedImage = DetectHoleAndSlot(FocusedImage, 50, 100)

    OriginalImage = np.copy(FocusedImage)
    FitArea(OriginalImage, window.label)


def closeEvent(event):
    exit(0)


def BindEvent():
    window.scaleSlider.valueChanged.connect(ScaleValueChanged)
    window.fitArea.clicked.connect(onFitArea)
    window.fillArea.clicked.connect(onFillArea)
    window.scaleUp.clicked.connect(onScaleUp)
    window.scaleDown.clicked.connect(onScaleDown)

    window.openCamera.clicked.connect(OpenCamera)
    window.takePhoto.clicked.connect(TakePhoto)
    window.savePicture.clicked.connect(SavePicture)
    window.openPicture.clicked.connect(OpenPicture)

    window.cutBlurredSlider.valueChanged.connect(CutBlurredValueChanged)
    window.cutThresholdSlider.valueChanged.connect(CutThresholdValueChanged)
    window.meaBlurredSlider.valueChanged.connect(MeaBlurredValueChanged)
    window.meaThresholdSlider.valueChanged.connect(MeaThresholdValueChanged)
    window.cutPicture.clicked.connect(onCutPicture)

    window.measure.clicked.connect(Measure)
    window.removeShape.clicked.connect(RemoveShape)
    window.extractColor.clicked.connect(ExtractColor)
    window.calibration.clicked.connect(onCalibration)
    window.undistort.clicked.connect(Undistort)
    window.test.clicked.connect(Test)

    # window.label.setMouseTracking(True)
    window.label.mouseMoveEvent = mouseMoveEvent
    window.label.mousePressEvent = mousePressEvent
    window.label.mouseReleaseEvent = mouseReleaseEvent
    window.label.wheelEvent = wheelEvent
    window.closeEvent = closeEvent


# 加载ui文件
UIFile = "DetectBoard.ui"
app = QtWidgets.QApplication([])
window = uic.loadUi(UIFile)
OriginalImage = []
ScaleImage = []
RawImage = []
FocusedImage = []
NoShapeBoard = []
Ratio = 1
InitialX = 0
InitialY = 0
CutX = 0
CutY = 0
K = 0
DisCoe = 0
PixelFactor = 0
IsMoveShape = False
IsRealTime = False



if __name__ == '__main__':
    print('hello world')
    fileName = os.path.join(os.getcwd(), 'Pictures', '3.jpg')
    print(fileName)
    OriginalImage = cv2.imread(fileName)
    ScaleImage = np.copy(OriginalImage)
    RawImage = np.copy(OriginalImage)
    FitArea(OriginalImage, window.label)
    RawImage = cv2.resize(RawImage, (1200, 800))
    BindEvent()
    window.show()
    app.exec_()
