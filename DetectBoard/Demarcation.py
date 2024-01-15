import cv2
import numpy as np
import os

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
    path1 = os.path.join(os.getcwd(), 'Pictures', 'de1.jpg')
    path2 = os.path.join(os.getcwd(), 'Pictures', 'de2.jpg')
    path3 = os.path.join(os.getcwd(), 'Pictures', 'de3.jpg')
    path4 = os.path.join(os.getcwd(), 'Pictures', 'de4.jpg')
    # 读取图像并提取角点
    image_paths = [path1, path2, path3, path4]  # 替换为实际图像路径
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