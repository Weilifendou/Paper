import cv2
from ciede2000 import CIEDE2000
def StasticColor(rgb):

    lab_image = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    pixels = lab_image.reshape(-1, 3)
    color_differences = []
    for pixel in pixels:
        color_difference = CIEDE2000((0, 0, 0), pixel)
        color_differences.append((color_difference, pixel[0], pixel[1], pixel[2]))
    return color_differences
    # # 打印CIEDE2000色差结果
    # for i, difference in enumerate(color_differences):
    #     print(f"Pixel {i + 1}: {difference}")

