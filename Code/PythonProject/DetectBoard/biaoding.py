# 引入所有函数，绘图板块
from PIL import Image
from imutils.feature import harris
from pylab import *
from numpy import *
# n维图像库，我们这里使用filters函数
from scipy.ndimage import filters


# 使用高斯滤波器进行卷积，标准差为3
# 这个函数得到Harris响应函数值的图像
def compute_harris_response(im, sigma=3):
    """ Compute the Harris corner detector response function
        for each pixel in a graylevel image. """

    # derivatives
    imx = zeros(im.shape)
    # x方向上的高斯导数
    filters.gaussian_filter(im, (sigma, sigma), (0, 1), imx)
    imy = zeros(im.shape)
    # y方向上的高斯导数
    filters.gaussian_filter(im, (sigma, sigma), (1, 0), imy)

    # compute components of the Harris matrix
    # 导数运算的高斯模糊值，第一个参数表示input，通过这样的方式得到矩阵的分量
    Wxx = filters.gaussian_filter(imx * imx, sigma)
    Wxy = filters.gaussian_filter(imx * imy, sigma)
    Wyy = filters.gaussian_filter(imy * imy, sigma)

    # 计算特征值与迹
    Wdet = Wxx * Wyy - Wxy ** 2
    Wtr = Wxx + Wyy

    return Wdet / Wtr


# 从响应图像中获得harris角点检测结果，min_dist表示最少数目的分割角点，threshold是阈值
def get_harris_points(harrisim, min_dist=10, threshold=0.6):
    """ Return corners from a Harris response image
        min_dist is the minimum number of pixels separating
        corners and image boundary. """

    # find top corner candidates above a threshold
    # 寻找高于阈值的候选角点，.max是numpy的函数·
    corner_threshold = harrisim.max() * threshold
    harrisim_t = (harrisim > corner_threshold) * 1

    # get coordinates of candidates
    # nonzeros(a)返回数组a中值不为零的元素的下标，它的返回值是一个长度为a.ndim(数组a的轴数)的元组,.T是矩阵转置操作
    coords = array(harrisim_t.nonzero()).T

    # ...and their values，harrisim的响应值
    candidate_values = [harrisim[c[0], c[1]] for c in coords]

    # 从小到大输出，注意输出的是下标
    index = argsort(candidate_values)

    # 标记分割角点范围的坐标
    allowed_locations = zeros(harrisim.shape)
    allowed_locations[min_dist:-min_dist, min_dist:-min_dist] = 1

    # select the best points taking min_distance into account,选择harris点
    filtered_coords = []
    for i in index:
        if allowed_locations[coords[i, 0], coords[i, 1]] == 1:
            # 添加坐标
            filtered_coords.append(coords[i])
            # 删除过近区域
            allowed_locations[(coords[i, 0] - min_dist):(coords[i, 0] + min_dist),
            (coords[i, 1] - min_dist):(coords[i, 1] + min_dist)] = 0

    return filtered_coords


# 绘制检测角点，灰度图 ,上个函数的角点需要满足的条件，1：角点范围之内；2：高于harrisim响应值的阈值
def plot_harris_points(image, filtered_coords):
    """ Plots corners found in image. """

    figure()
    gray()
    imshow(image)
    # 将坐标以*标出
    plot([p[1] for p in filtered_coords],
         [p[0] for p in filtered_coords], 'ro')
    axis('off')
    show()


# 读入图像
for i in range(1, 14):
    print("第" + str(i) + "张图")
    im = array(Image.open('images1/empire' + str(i) + '.jpg').convert('L'))
    # 检测harris角点
    harrisim = compute_harris_response(im)
    # figure()
    # imshow(harrisim)
    # axis('off')
    # threshold = [0.1]
    # for j, thres in enumerate(threshold):
    filtered_coords = get_harris_points(harrisim, 6)
    plot_harris_points(im, filtered_coords)
# show()