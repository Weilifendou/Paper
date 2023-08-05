import cv2


if __name__ == '__main__':
    print('hello world')

    path = '1.jpg'
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    cv2.namedWindow('Original img', cv2.WINDOW_NORMAL)
    cv2.imshow("Original img", img)
    cv2.waitKey(0)
