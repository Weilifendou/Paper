import cv2
import numpy as np


def empty(a):
    pass
def drawTrackbar():
    # 使用Trackbar调参，效率提高一大截
    cv2.namedWindow('Main')
    # mat = np.ones((30,975)) #调节Trackbar显示的宽度
    # cv2.imshow('Main', mat)
    cv2.createTrackbar('GausSize', 'Main', 17, 255, empty)
    cv2.createTrackbar('CannyThreshold1', 'Main', 46, 255, empty)
    cv2.createTrackbar('CannyThreshold2', 'Main', 190, 255, empty)
    cv2.createTrackbar('SobelWeight', 'Main', 50, 100, empty)
def detectEdge(image):
    while cv2.getWindowProperty('Main', cv2.WND_PROP_VISIBLE) >= 1:
        gausSize = cv2.getTrackbarPos('GausSize', 'Main')
        CannyThreshold1 = cv2.getTrackbarPos('CannyThreshold1', 'Main')
        CannyThreshold2 = cv2.getTrackbarPos('CannyThreshold2', 'Main')
        SobelWeight = cv2.getTrackbarPos('SobelWeight', 'Main')
        SobelWeight /= 100
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, img = cv2.threshold(gray, CannyThreshold1, 255, cv2.THRESH_BINARY);
        medianImg = cv2.medianBlur(img, 5)
        # 使用Sobel算子进行边缘检测
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, 5)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, 5)

        # 将梯度值取绝对值
        sobel_x = cv2.convertScaleAbs(sobel_x)
        sobel_y = cv2.convertScaleAbs(sobel_y)

        # 合并x和y方向的梯度
        sobel = cv2.addWeighted(sobel_x, SobelWeight, sobel_y, 1-SobelWeight, 0)
        if gausSize % 2 == 0: gausSize += 1
        gausBlur = cv2.GaussianBlur(sobel, (gausSize, gausSize), 0)
        canny = cv2.Canny(gausBlur, CannyThreshold1, CannyThreshold2)
        cv2.imshow('Main', medianImg)
        if cv2.waitKey(1) == 27:
            break

def main():
    image = cv2.imread('3.jpg')
    image = cv2.resize(image, (1200, 800))
    drawTrackbar()
    detectEdge(image)


if __name__ == '__main__':
    print('hello world')
    # cv2.waitKey(0)
    main()