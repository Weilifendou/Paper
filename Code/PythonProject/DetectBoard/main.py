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
def main():
    drawTrackbar()
    image = cv2.imread('2.jpg', cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (1200, 800))
    while cv2.getWindowProperty('Main', cv2.WND_PROP_VISIBLE) >= 1:
        gausSize = cv2.getTrackbarPos('GausSize', 'Main')
        CannyThreshold1 = cv2.getTrackbarPos('CannyThreshold1', 'Main')
        CannyThreshold2 = cv2.getTrackbarPos('CannyThreshold2', 'Main')
        if gausSize % 2 == 0: gausSize += 1
        gausBlur = cv2.GaussianBlur(image, (gausSize, gausSize), 0)
        Canny = cv2.Canny(gausBlur, CannyThreshold1, CannyThreshold2)
        cv2.imshow('Main', Canny)
        if cv2.waitKey(1) == 27:
            break

if __name__ == '__main__':
    print('hello world')
    # cv2.waitKey(0)
    main()