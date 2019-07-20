import cv2 as cv
import numpy as np


"""
开运算:先进性腐蚀再进行膨胀就叫做开运算,它被用来去除噪声。
闭运算:先膨胀再腐蚀。它经常被用来填充前景物体中的小洞，或者前景物体上的小黑点。
这里我们用到的函数是 cv2.morphologyEx()。
开闭操作作用：
1. 去除小的干扰块-开操作
2. 填充闭合区间-闭操作
3. 水平或垂直线提取,调整kernel的row，col值差异。
比如：采用开操作，kernel为(1, 15),提取垂直线，kernel为(15, 1),提取水平线，
"""

"""
其他形态学操作：
顶帽：原图像与开操作之间的差值图像
黑帽：比操作与原图像直接的差值图像
形态学梯度：其实就是一幅图像膨胀与腐蚀的差别。 结果看上去就像前景物体的轮廓
基本梯度：膨胀后图像减去腐蚀后图像得到的差值图像。
内部梯度：用原图减去腐蚀图像得到的差值图像。
外部梯度：膨胀后图像减去原图像得到的差值图像。
"""


def open_demo(image):
    print(image.shape)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    cv.imshow("binary", binary)
    """
    在前面的例子中我们使用Numpy构建了结构化元素，它是正方形的。
    但有时我们需要构建一个椭圆形 / 圆形的核。为了实现这种要求，提供了OpenCV
    函数cv2.getStructuringElement()。你只需要告诉他你需要的核的形状和大小。
    """
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    dst = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel=kernel)
    cv.imshow("open_demo", dst)


def close_demo(image):
    print(image.shape)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    cv.imshow("binary", binary)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    dst = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel=kernel)
    cv.imshow("open_demo", dst)


def other_morphology_demo(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    dst = cv.morphologyEx(gray, cv.MORPH_BLACKHAT, kernel=kernel)
    cimg = np.array(gray.shape, np.uint8)
    cimg = 100
    dst = cv.add(dst, cimg)

    cv.imshow("top_hat_demo", dst)


def main():
    src = cv.imread("../images/lena.jpg")
    open_demo(src)
    # close_demo(src)

    # # 彩色图像腐蚀，膨胀
    # img = cv.imread("../images/lena.jpg")
    # cv.imshow("img", img)
    # kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    # # dst = cv.dilate(img, kernel=kernel)
    # dst = cv.erode(img, kernel=kernel)
    # cv.imshow("dilate", dst)

    # # tophat, blackhat
    #
    # top_hat_demo(src)

    cv.waitKey(0)  # 等有键输入或者1000ms后自动将窗口消除，0表示只用键输入结束窗口
    cv.destroyAllWindows()  # 关闭所有窗口


if __name__ == '__main__':
    main()