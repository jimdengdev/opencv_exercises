import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def shi_tomasi(image):
    """
    Shi-Tomasi 使用的打分函数为：R = min(λ1,λ2)
    如果打分超过阈值，我们就认为它是一个角点。我们可以把它绘制到 λ1 ～ λ2 空间中，
    只有当 λ1 和 λ2 都大于最小值时，才被认为是角点
    OpenCV 提供了函数：cv2.goodFeaturesToTrack()。这个函数可以帮我们使用 Shi-Tomasi方法获取图像中 N 个最好的角点

    通常情况下，输入的应该是灰度图像。然后确定你想要检测到的角点数目。
    再设置角点的质量水平，0 到 1 之间。它代表了角点的最低质量，低于这个数的所有角点都会被忽略。
    最后在设置两个角点之间的最短欧式距离。 根据这些信息，函数就能在图像上找到角点。
    所有低于质量水平的角点都会被忽略。然后再把合格角点按角点质量进行降序排列。
    函数会采用角点质量 最高的那个角点（排序后的第一个），然后将它附近（最小距离之内）的角点都删掉。
    按着这样的方式最后返回 N 个最佳角点
    """
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # 取出25个最佳角点
    corners = cv.goodFeaturesToTrack(gray, 25, 0.01, 10)  # 返回结果是[[311., 250.]] 两层括号数组

    # 取整
    corners = np.int0(corners)

    for i in corners:
        x, y = i.ravel()  # 分解
        cv.circle(image, (x, y), 3, 255, -1)

    plt.imshow(image), plt.show()


def main():
    src = cv.imread("../images/blox.jpg")
    shi_tomasi(src)
    cv.waitKey(0)  # 等有键输入或者1000ms后自动将窗口消除，0表示只用键输入结束窗口
    cv.destroyAllWindows()  # 关闭所有窗口


if __name__ == '__main__':
    main()