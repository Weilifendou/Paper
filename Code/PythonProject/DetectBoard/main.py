import cv2
import numpy as np


def empty(a):
    pass
def drawTrackbar():
    # 使用Trackbar调参，效率提高一大截
    cv2.namedWindow('Main')
    # mat = np.ones((30,975)) #调节Trackbar显示的宽度
    # cv2.imshow('Main', mat)
    cv2.createTrackbar('GausSize', 'Main', 20, 255, empty)
    cv2.createTrackbar('CannyThreshold1', 'Main', 10, 255, empty)
    cv2.createTrackbar('CannyThreshold2', 'Main', 30, 255, empty)
    cv2.createTrackbar('Dist', 'Main', 10, 500, empty)
    cv2.createTrackbar('MinRaduis', 'Main', 10, 500, empty)
    cv2.createTrackbar('MaxRaduis', 'Main', 50, 500, empty)
def detectEdge(image):
    while cv2.getWindowProperty('Main', cv2.WND_PROP_VISIBLE) >= 1:
        gausSize = cv2.getTrackbarPos('GausSize', 'Main')
        CannyThreshold1 = cv2.getTrackbarPos('CannyThreshold1', 'Main')
        CannyThreshold2 = cv2.getTrackbarPos('CannyThreshold2', 'Main')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if gausSize % 2 == 0: gausSize += 1
        gausBlur = cv2.GaussianBlur(gray, (gausSize, gausSize), 0)
        canny = cv2.Canny(gausBlur, CannyThreshold1, CannyThreshold2)
        kernel = np.ones((5, 5), np.uint8)
        canny = cv2.dilate(canny, kernel, 1)
        cv2.imshow('Main', canny)

        if cv2.waitKey(1) == 27:
            break
def DetectCircle(image):
    while cv2.getWindowProperty('Main', cv2.WND_PROP_VISIBLE) >= 1:
        gausSize = cv2.getTrackbarPos('GausSize', 'Main')
        CannyThreshold1 = cv2.getTrackbarPos('CannyThreshold1', 'Main')
        CannyThreshold2 = cv2.getTrackbarPos('CannyThreshold2', 'Main')
        Dist = cv2.getTrackbarPos('Dist', 'Main')
        MinRaduis = cv2.getTrackbarPos('MinRaduis', 'Main')
        MaxRaduis = cv2.getTrackbarPos('MaxRaduis', 'Main')
        origin = np.copy(image)
        gray = cv2.cvtColor(origin, cv2.COLOR_BGR2GRAY)
        if gausSize % 2 == 0: gausSize += 1
        blurred = cv2.GaussianBlur(gray, (gausSize, gausSize), 0)
        canny = cv2.Canny(blurred, CannyThreshold1, CannyThreshold2)
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1,
                           Dist, param1=CannyThreshold1, param2=CannyThreshold2, minRadius=MinRaduis, maxRadius=MaxRaduis)
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                cv2.circle(origin, (x, y), r, (0, 255, 0), 2)
            cv2.imshow("Main", origin)
        else:
            print("未检测到圆形")

        if cv2.waitKey(1) == 27:
            break


def DetectRetangle(image):
    while cv2.getWindowProperty('Main', cv2.WND_PROP_VISIBLE) >= 1:
        gausSize = cv2.getTrackbarPos('GausSize', 'Main')
        CannyThreshold1 = cv2.getTrackbarPos('CannyThreshold1', 'Main')
        CannyThreshold2 = cv2.getTrackbarPos('CannyThreshold2', 'Main')
        Dist = cv2.getTrackbarPos('Dist', 'Main')
        MinRaduis = cv2.getTrackbarPos('MinRaduis', 'Main')
        MaxRaduis = cv2.getTrackbarPos('MaxRaduis', 'Main')
        origin = np.copy(image)
        gray = cv2.cvtColor(origin, cv2.COLOR_BGR2GRAY)
        if gausSize % 2 == 0: gausSize += 1
        blurred = cv2.GaussianBlur(gray, (gausSize, gausSize), 0)
        edges = cv2.Canny(blurred, CannyThreshold1, CannyThreshold2)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(origin, contours, -1, (0, 0, 255), 2)
        background = np.ones_like(origin)
        cv2.drawContours(background, contours, -1, (0, 0, 255), 1)
        cv2.imshow("Main", background)
        if cv2.waitKey(1) == 27:
            break
def main():
    image = cv2.imread('1.jpg')
    image = cv2.resize(image, (1200, 800))
    drawTrackbar()
    DetectCircle(image)
    # detectEdge(image)
    # DetectRetangle(image)

if __name__ == '__main__':
    print('hello world')
    # cv2.waitKey(0)
    main()