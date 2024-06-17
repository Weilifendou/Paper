from sklearn.metrics import pairwise_distances
from sklearn.metrics import silhouette_score
import numpy as np

def kmeans_plusplus(X, k):
    centers = []
    centers.append(X[np.random.randint(0, len(X))])

    while len(centers) < k:
        distances = []
        for x in X:
            distances.append(np.min([np.abs(x - center) for center in centers]))

        probabilities = distances / np.sum(distances)
        selected_index = np.random.choice(range(len(X)), p=probabilities)
        centers.append(X[selected_index])

    return np.array(centers)


# 实现一维K-means聚类算法
def kmeans(X, k, initial_centers):
    centers = initial_centers.copy()
    n_samples = len(X)
    labels = np.zeros(n_samples)
    distances = np.zeros((n_samples, k))

    while True:
        # 计算每个样本到聚类中心的距离
        for i in range(k):
            distances[:, i] = np.abs(X - centers[i])

        # 为每个样本分配最近的聚类中心
        new_labels = np.argmin(distances, axis=1)

        # 如果聚类结果不再改变，停止迭代
        if np.array_equal(labels, new_labels):
            break

        # 更新聚类中心
        for i in range(k):
            centers[i] = np.mean(X[new_labels == i])

        labels = new_labels

    return centers, labels


def calculate_sse(X, centers, cluster_labels):
    sse = 0
    for i in range(len(X)):
        cluster_center = centers[cluster_labels[i]]
        sse += np.square(np.linalg.norm(X[i] - cluster_center))
    return sse


def calculate_silhouette_coefficient(X, cluster_labels):
    distances = pairwise_distances(X.reshape(-1, 1))
    silhouette_avg = silhouette_score(distances, cluster_labels)
    return silhouette_avg

def SustainKmean(X, start, end):
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    # 使用一维K-means++算法初始化聚类中心
    k = list(range(start, end))
    # k = 1
    sse = []
    coe = []
    for i in k:
        initial_centers = kmeans_plusplus(X, i)
    #     # 使用一维K-means算法进行聚类
        final_centers, cluster_labels = kmeans(X, i, initial_centers)
        sse.append(calculate_sse(X, final_centers, cluster_labels))
        coe.append(calculate_silhouette_coefficient(X, cluster_labels))

    # 设置中文字体为宋体, 西文字体为Times New Roman, 字号为五号
    zh_font = FontProperties(fname='C:/Windows/Fonts/simsun.ttc', size=16)
    en_font = FontProperties(fname='C:/Windows/Fonts/times.ttf', size=16)
    # 创建包含两个子图的图像

    # 绘制曲线
    plt.plot(k, sse, label='SSE')
    plt.legend(loc='upper right')

    # 添加图例
    plt.subplot(2, 1, 1)  # 第一个子图
    plt.plot(k, sse)
    plt.xlabel('K值', fontproperties=en_font)
    plt.ylabel('SSE', fontproperties=en_font)
    plt.title('手肘曲线')

    plt.subplot(2, 1, 2)  # 第二个子图
    plt.plot(k, coe)
    plt.xlabel('K值')
    plt.ylabel('Coe')
    plt.title('轮廓系数曲线')
    plt.tight_layout()  # 调整子图之间的间距
    plt.rcParams['font.sans-serif'] = 'Microsoft YaHei'
    plt.show()
    for i in sse:
        print(f'{i:.3f}')
    for i in coe:
        print(f'{i:.3f}')
