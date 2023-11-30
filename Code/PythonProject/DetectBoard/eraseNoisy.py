import numpy as np
import cv2

def eraseNoisy(edges, threshold):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(edges)
    # 创建一个掩码，用于存储去除噪点后的图像
    mask = np.zeros_like(edges)
    # 循环遍历每个连通域
    for label in range(1, num_labels):
        # 获取连通域的面积
        area = stats[label, cv2.CC_STAT_AREA]

        # 如果连通域的面积大于阈值，则保留该连通域
        if area > threshold:
            # 将连通域对应的像素设置为255
            mask[labels == label] = 255
    return mask