import cv2 as cv
import numpy as np


"""
图像特征提取与描述
1. 问题：拿到一张图片的一堆碎片，怎样把这些碎片以正确的方式排列起来从而重建这幅图像。
答案就是：我们要寻找一些唯一的特征，这些特征要适于被跟踪，容易被比较。
找到图像特征的技术被称为特征检测。 
2. 特征是什么，怎样找到？找一张图片的角点。角点的一个特性：向任何方向移动变化都很大。
3. 特征描述：计算机对特征周围的区域进行描述，这样它才能在其他图像中找到相同的特征。
"""


def harris_demo(image):
    """
    利用角点的一个特性：向任何方向移动变化都很大。将窗口向各个方向 移动（u，v）然后计算所有差异的总和。
    E(u, v) = Ew(x,y)[(I(x+u,y+v)-I(x,y)]^2
    根据一个用来判定窗口内是否包含角点的等式 进行打分。R = det(M) - k(trace(M))^2
    所以根据这些特征中我们可以判断一个区域是否是角点，边界或者是平面。
     • 当 λ1 和 λ2 都小时，|R| 也小，这个区域就是一个平坦区域。
     • 当 λ1≫ λ2 或者 λ1≪ λ2，时 R 小于 0，这个区域是边缘
     • 当 λ1 和 λ2 都很大，并且 λ1～λ2 中的时，R 也很大，（λ1 和 λ2 中的最 小值都大于阈值）说明这个区域是角点
    """
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # 输入图像必须是float32 ，最后一个参数在0.04到0.05之间
    gray = np.float32(gray)

    # 参数解释：cornerHarris(src, blockSize, ksize, k, dst=None, borderType=None)
    # src：float32输入图像
    # blockSize：角点检测中要考虑的领域大小。
    # ksize：Sobel 求导中使用的窗口大小（Sobel为一阶算子，请参照tutorial_16_grad）
    # k：Harris 角点检测方程中的自由参数，取值参数为 [0,04，0.06].
    dst = cv.cornerHarris(gray, 2, 3, 0.04)

    dst = cv.dilate(dst, None)  # 交点膨胀，使其突出
    print(dst.max())
    # 取出膨胀后dst.max()的1%以上交点,标记为红色
    image[dst>0.01*dst.max()] = [0, 0, 255]

    cv.imshow('harris corner',image)


def sub_pixel(image):
    """
    有时我们需要最大精度的角点检测。可用函数cv2.cornerSubPix()， 它提供亚像素级别的角点检测。
    首先我们要找到 Harris 角点，然后将角点的重心传给这个函数进行修正。
    Harris 角点用红色像素标出，绿色像素是修正后的像素。
    在使用这个函数是我们要定义一个迭代停止条件。当迭代次数达到或者精度条件满足后迭代就会停止。
    我们同样需要定义进行角点搜索的邻域大小。
    """

    # 找出Harris 角点
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv.cornerHarris(gray, 2, 3, 0.04)
    dst = cv.dilate(dst, None)
    ret, dst = cv.threshold(dst, 0.01 * dst.max(), 255, 0)
    dst = np.uint8(dst)

    # 用connectedComponentsWithStats()找角点重心centroids
    ret, labels, stats, centroids = cv.connectedComponentsWithStats(dst)
    print(ret)

    # 定义迭代停止条件，提取角点
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)

    # cornerSubPix(image, corners, winSize, zeroZone, criteria)
    # corners: 检测到的角点，即是输入也是输出。
    # winSize: 计算亚像素角点时考虑的区域的大小，大小为NXN; N=(winSize*2+1)
    # zeroZone: 作用类似于winSize，但是总是具有较小的范围，通常忽略（即Size(-1, -1)）。
    corners = cv.cornerSubPix(gray, np.float32(centroids), (5,5), (-1,-1), criteria)

    # 将角点重心和角点水平堆砌，例如
    # >>> a = np.array([[1],[2],[3]])
    # >>> b = np.array([[2],[3],[4]])
    # >>> np.hstack((a,b))
    # array([[1, 2],
    #        [2, 3],
    #        [3, 4]])
    res = np.hstack((centroids, corners))

    # np.int0 可以用来省略小数点后面的数字（非四五入）
    res = np.int0(res)

    # 画出角点颜色
    image[res[:, 1], res[:, 0]] = [0, 0, 255]
    image[res[:, 3], res[:, 2]] = [0, 255, 0]
    cv.imwrite('subpixel5.png', image)


def main():
    src = cv.imread("../images/blox.jpg")
    # harris_demo(src)
    sub_pixel(src)
    cv.waitKey(0)  # 等有键输入或者1000ms后自动将窗口消除，0表示只用键输入结束窗口
    cv.destroyAllWindows()  # 关闭所有窗口


if __name__ == '__main__':
    main()