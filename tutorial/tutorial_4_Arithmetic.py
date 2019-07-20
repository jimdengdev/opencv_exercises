import cv2 as cv
import numpy as np


# 对图像像素级别的加减乘除
def add_demo(m1, m2):
    dst = cv.add(m1, m2)
    cv.imshow("add_demo", dst)


def subtract_demo(m1, m2):
    dst = cv.subtract(m1, m2)
    cv.imshow("subtract_demo", dst)


def divide_demo(m1, m2):
    dst = cv.divide(m1, m2)
    cv.imshow("divide_demo", dst)


def multiply_demo(m1, m2):
    dst = cv.multiply(m1, m2)
    cv.imshow("multiply_demo", dst)

# 图像的逻辑运算，：AND，OR，NOT，XOR
def logic_demo(m1, m2):
    image = cv.imread("CrystalLiu2.jpg")
    cv.imshow("image1",image)
    dst = cv.bitwise_not(image)
    cv.imshow("logic_demo", dst)

def contrast_brightness_demo(image, c, b):
    h, w, ch = image.shape
    blank = np.zeros([h, w, ch], image.dtype)

    # 图像混合，c, 1-c为这两张图片的权重
    dst = cv.addWeighted(image, c, blank, 1-c, b)
    cv.imshow("contrast_brightness_demo", dst)


# 对照片像素做均值方差
def others(m1, m2):
    M1 = cv.mean(m1)
    M2 = cv.mean(m2)
    print(M1)
    print(M2)

    mean1, dev1 = cv.meanStdDev(m1)
    print("均值：",mean1,"方差：", dev1)


if __name__ == '__main__':
    print("----------Hello World!----------")
    src1 = cv.imread("../images/01.jpg")
    src2 = cv.imread("../images/02.jpg")
    print(src1.shape)
    print(src2.shape)

    cv.namedWindow("image1", cv.WINDOW_AUTOSIZE)  # 创建窗口, 窗口尺寸自动调整
    cv.imshow("image1", src1)
    cv.imshow("image2", src2)

    # add_demo(src1, src2)
    # subtract_demo(src1, src2)
    # divide_demo(src1, src2)
    # multiply_demo(src1,src2)
    # others(src1, src2)

    # logic_demo(src1, src2)
    src = cv.imread("../images/CrystalLiu2.jpg")
    contrast_brightness_demo(src,1.3, 80)
    cv.waitKey(0)

    cv.destroyAllWindows()