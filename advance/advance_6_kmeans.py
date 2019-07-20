import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


"""
kmeans 即K 值聚类，这个算法是一个迭代过程。
第一步：随机选取两个重心点，C1 和 C2（有时可以选取数据中的两个点 作为起始重心）。 
第二步：计算每个点到这两个重心点的距离，如果距离 C1 比较近就标记为 0，如果距离 C2 比较近就标记为 1。
（如果有更多的重心点，可以标记为 “2”，“3”等） 
第三步：重新计算所有蓝色点的重心，和所有红色点的重心，并以这两个 点更新重心点的位置。
（图片只是为了演示说明而已，并不代表实际数据） 重复步骤 2，更新所有的点标记。 

继续迭代步骤 2 和 3，直到两个重心点的位置稳定下来。
（当然也可以通 过设置迭代次数，或者设置重心移动距离的阈值来终止迭代。）。
此时这些点到 它们相应重心的距离之和最小。简单来说，C1 到红色点的距离与 C2 到蓝色点的距离之和最小。

"""

def kmeans_1_demo():  # 采用单特征，都放在一列，一个样本一行
    """
    cv.kmeans(data, K, bestLabels, criteria, attempts, flags, centers=None)
    1. data: np.ﬂoat32 类型的数据，每个特征应该放在一列。
    2. K: 聚类的最终数目。
    3. criteria: 终止迭代的条件。当条件满足时，算法的迭代终止。
        它应该是 一个含有 3 个成员的元组，它们是（typw，max_iter，epsilon）：
        • type 终止的类型：有如下三种选择：
            – cv2.TERM_CRITERIA_EPS 只有精确度 epsilon 满足是 停止迭代。
            – cv2.TERM_CRITERIA_MAX_ITER 当迭代次数超过阈值 时停止迭代。
            – cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER 上面的任何一个条件满足时停止迭代。
        • max_iter 表示最大迭代次数。
        • epsilon 精确度阈值。
    4. attempts: 使用不同的起始标记来执行算法的次数。算法会返回紧密度 最好的标记。紧密度也会作为输出被返回。
    5. ﬂags：用来设置如何选择起始重心。通常我们有两个选择：cv2.KMEANS_PP_CENTERS 和 cv2.KMEANS_RANDOM_CENTERS。

    :return:
    1. compactness：紧密度，返回每个点到相应重心的距离的平方和。
    2. labels：标志数组（与上一节提到的代码相同），每个成员被标记为 0，1 等
    3. centers：由聚类的中心组成的数组。
    """
    x = np.random.randint(25, 100, 25)  # 在[25, 100)之间随机产生25个整数
    y = np.random.randint(175, 255, 25)
    z = np.hstack((x, y))
    # print(z)
    z = z.reshape((50, 1))
    z = np.float32(z)
    # print(z)

    # 将z分成256个bins，范围是[0, 256),高度是统计出来的相同值个数
    plt.hist(z, 256, [0, 256]), plt.show()

    # Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    # Set flags (Just to avoid line break in the code)
    flags = cv.KMEANS_RANDOM_CENTERS

    # Apply KMeans
    compactness, labels, centers = cv.kmeans(z,2,None,criteria,10,flags)

    # 根据标记分成两组点A，B
    A = z[labels == 0]
    B = z[labels == 1]

    # 现在将A组数用红色表示，将B组数据用蓝色表示，重心用黄色表示。
    plt.hist(A,256,[0,256],color = 'r')
    plt.hist(B,256,[0,256],color = 'b')
    plt.hist(centers,32,[0,256],color = 'y')
    plt.show()


# 采用两个特征，在本例中我们的测试数据适应 50x2 的向量，其中包含 50 个人的身高和 体重。
# 第一列对应与身高，第二列对应与体重。第一行包含两个元素，第一个 是第一个人的身高，第二个是第一个人的体重。
# 剩下的行对应与其他人的身高 和体重。
def kmeans_2_demo():
    X = np.random.randint(25, 50, (25, 2))
    Y = np.random.randint(60, 85, (25, 2))
    Z = np.vstack((X, Y))
    Z = np.float32(Z)

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv.kmeans(Z, 2, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

    A = Z[label.ravel() == 0]
    B = Z[label.ravel() == 1]
    # Plot the data
    plt.scatter(A[:, 0], A[:, 1])
    plt.scatter(B[:, 0], B[:, 1], c='r')
    plt.scatter(center[:, 0], center[:, 1], s=80, c='y', marker='s')
    plt.xlabel('Height'), plt.ylabel('Weight')
    plt.show()


def color_less_kmeans(image):
    """
    颜色量化就是减少图片中颜色数目的一个过程。
    为什么要减少图片中的颜 色呢？减少内存消耗！有些设备的资源有限，只能显示很少的颜色。
    在这种情 况下就需要进行颜色量化。我们使用 K 值聚类的方法来进行颜色量化。 没有什么新的知识需要介绍了。
    现在有 3 个特征：R，G，B。所以我们需 要把图片数据变形成 Mx3（M 是图片中像素点的数目）的向量。

    聚类完成后， 我们用聚类中心值替换与其同组的像素值，这样结果图片就只含有指定数目的颜色了。

    """
    Z = image.reshape(-1, 3)  # 转化成三列
    Z = np.float32(Z)

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 8
    ret, label, center = cv.kmeans(Z, K, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    print(center[0])
    print(label.shape,center.shape, res.shape)
    np.savetxt("data.txt", label.flatten())
    np.savetxt("res.txt", res)
    res2 = res.reshape((image.shape))
    cv.imwrite("color_less.png", res2)
    cv.imshow('res2', res2)



def main():
    src = cv.imread("../images/Crystal.jpg")
    # kmeans_1_demo()
    # kmeans_2_demo()
    color_less_kmeans(src)

    cv.waitKey(0)  # 等有键输入或者1000ms后自动将窗口消除，0表示只用键输入结束窗口
    cv.destroyAllWindows()  # 关闭所有窗口


if __name__ == '__main__':
    main()