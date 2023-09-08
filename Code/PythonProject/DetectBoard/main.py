import cv2
import numpy as np


def empty(a):
    pass
def drawTrackbar():
    # 使用Trackbar调参，效率提高一大截
    cv2.namedWindow('Main')
    # mat = np.ones((30,975)) #调节Trackbar显示的宽度
    # cv2.imshow('Main', mat)
    cv2.createTrackbar('GausSize', 'Main', 5, 255, empty)
    cv2.createTrackbar('CannyThreshold1', 'Main', 10, 255, empty)
    cv2.createTrackbar('CannyThreshold2', 'Main', 30, 255, empty)
    cv2.createTrackbar('Dist', 'Main', 10, 500, empty)
    cv2.createTrackbar('MinRaduis', 'Main', 10, 500, empty)
    cv2.createTrackbar('MaxRaduis', 'Main', 50, 500, empty)
    cv2.createTrackbar('LineThreshold', 'Main', 100, 200, empty)
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


def DetectRetangle(origin):
    while cv2.getWindowProperty('Main', cv2.WND_PROP_VISIBLE) >= 1:
        gausSize = cv2.getTrackbarPos('GausSize', 'Main')
        CannyThreshold1 = cv2.getTrackbarPos('CannyThreshold1', 'Main')
        CannyThreshold2 = cv2.getTrackbarPos('CannyThreshold2', 'Main')
        Dist = cv2.getTrackbarPos('Dist', 'Main')
        MinRaduis = cv2.getTrackbarPos('MinRaduis', 'Main')
        MaxRaduis = cv2.getTrackbarPos('MaxRaduis', 'Main')
        image = np.copy(origin)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if gausSize % 2 == 0: gausSize += 1
        blurred = cv2.GaussianBlur(gray, (gausSize, gausSize), 0)
        edges = cv2.Canny(blurred, CannyThreshold1, CannyThreshold2)
        # 定义结构元素
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (gausSize, gausSize))

        # 执行开运算
        edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
        # contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)# 进行轮廓检测
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)# 进行轮廓检测

        # 检测圆角矩形
        for contour in contours:
            # 计算轮廓的周长
            perimeter = cv2.arcLength(contour, True)

            # 进行多边形逼近，获取近似的轮廓
            epsilon = Dist / 100 * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # 如果近似的轮廓有4个顶点，认为是一个矩形
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                if MinRaduis <= w and MinRaduis <= h:
                    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)


        cv2.imshow("Main", image)
        if cv2.waitKey(1) == 27:
            break


def DetectRetangle2(image):
    while cv2.getWindowProperty('Main', cv2.WND_PROP_VISIBLE) >= 1:
        gausSize = cv2.getTrackbarPos('GausSize', 'Main')
        CannyThreshold1 = cv2.getTrackbarPos('CannyThreshold1', 'Main')
        CannyThreshold2 = cv2.getTrackbarPos('CannyThreshold2', 'Main')
        Dist = cv2.getTrackbarPos('Dist', 'Main')
        MinRaduis = cv2.getTrackbarPos('MinRaduis', 'Main')
        MaxRaduis = cv2.getTrackbarPos('MaxRaduis', 'Main')
        LineThreshold = cv2.getTrackbarPos('LineThreshold', 'Main')
        origin = np.copy(image)
        gray = cv2.cvtColor(origin, cv2.COLOR_BGR2GRAY)
        if gausSize % 2 == 0: gausSize += 1
        blurred = cv2.GaussianBlur(gray, (gausSize, gausSize), 0)
        edges = cv2.Canny(blurred, CannyThreshold1, CannyThreshold2)

        # 进行霍夫直线检测
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=LineThreshold)

        # 绘制检测到的直线
        if lines is not None:
            for line in lines:
                rho, theta = line[0]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * a)
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * a)
                cv2.line(origin, (x1, y1), (x2, y2), (0, 0, 255), 2)

        cv2.imshow("Main", origin)
        if cv2.waitKey(1) == 27:
            break
def main():
    image = cv2.imread('1.jpg')
    image = cv2.resize(image, (1200, 800))
    drawTrackbar()
    # DetectCircle(image)
    # detectEdge(image)
    DetectRetangle(image)

if __name__ == '__main__':
    print('hello world')
    # cv2.waitKey(0)
    main()