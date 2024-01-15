import numpy as np
import cv2

def DetectCircle(origin):
    while cv2.getWindowProperty('Main', cv2.WND_PROP_VISIBLE) >= 1:
        BlurredSize = cv2.getTrackbarPos('BlurredSize', 'SeekBar')
        CannyThreshold1 = cv2.getTrackbarPos('CannyThreshold1', 'SeekBar')
        CannyThreshold2 = cv2.getTrackbarPos('CannyThreshold2', 'SeekBar')
        Dist = cv2.getTrackbarPos('Dist', 'SeekBar')
        MinRaduis = cv2.getTrackbarPos('MinRaduis', 'SeekBar')
        MaxRaduis = cv2.getTrackbarPos('MaxRaduis', 'SeekBar')
        image = np.copy(origin)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if BlurredSize % 2 == 0: BlurredSize += 1
        blurred = cv2.GaussianBlur(gray, (BlurredSize, BlurredSize), 0)
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