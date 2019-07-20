# roi 即兴趣区域，对图像提取想要的部分

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def roi_test(src):
    face = src[100:510, 200:600]
    gray = cv.cvtColor(face, cv.COLOR_BGR2GRAY)  # face彩色图片变成灰度图片
    cv.imshow("gray", gray)
    back_face = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
    cv.imshow("back_face", back_face)
    src[100:510, 200:600] = back_face
    cv.imshow("face", src)

def fill_color_demo(image):
    copyImg = image.copy()
    h, w = image.shape[:2]
    mask = np.zeros([h+2, w+2], np.uint8)

    # 参数：原图，mask图，起始点，起始点值减去该值作为最低值，起始点值加上该值作为最高值，彩色图模式
    cv.floodFill(copyImg, mask, (30, 30), (0, 255, 255), (100,100,100), (50, 50, 50), cv.FLOODFILL_FIXED_RANGE)
    cv.imshow("fill_color_demo", copyImg)

def fill_binary():
    image = np.zeros([400, 400, 3], np.uint8)
    image[100:300, 100:300, :] = 255
    cv.imshow("fill_binary", image)

    mask = np.ones([402, 402, 1], np.uint8)
    mask[101:301, 101:301] = 0

    cv.floodFill(image, mask, (200, 200), (100, 2, 255), cv.FLOODFILL_MASK_ONLY)
    cv.imshow("filled binary", image)

def make_border(image):
    """
    :param image:输入图像
    top, bottom, left, right 对应边界的像素数目。
    borderType 要添加那种类型的边界，类型如下:
    – cv2.BORDER_CONSTANT 添加有颜色的常数值边界，还需要 下一个参数（value）。
    – cv2.BORDER_REFLECT边界元素的镜像。比如: fedcba|abcdefgh|hgfedcb
    – cv2.BORDER_REFLECT_101orcv2.BORDER_DEFAULT 跟上面一样，但稍作改动。例如: gfedcb|abcdefgh|gfedcba
    – cv2.BORDER_REPLICATE重复最后一个元素。例如: aaaaaa| abcdefgh|hhhhhhh
    – cv2.BORDER_WRAP 不知道怎么说了, 就像这样: cdefgh| abcdefgh|abcdefg
    value 边界颜色，如果边界的类型是 cv2.BORDER_CONSTANT
    """
    BLUE = [255, 0, 0]

    replicate = cv.copyMakeBorder(image, 10, 10, 10, 10, cv.BORDER_REPLICATE)
    reflect = cv.copyMakeBorder(image, 10, 10, 10, 10, cv.BORDER_REFLECT)
    reflect101 = cv.copyMakeBorder(image, 10, 10, 10, 10, cv.BORDER_REFLECT_101)
    wrap = cv.copyMakeBorder(image, 10, 10, 10, 10, cv.BORDER_WRAP)
    constant = cv.copyMakeBorder(image, 10, 10, 10, 10, cv.BORDER_CONSTANT, value=BLUE)

    plt.subplot(231), plt.imshow(image, 'gray'), plt.title('ORIGINAL')
    plt.subplot(232), plt.imshow(replicate, 'gray'), plt.title('REPLICATE')
    plt.subplot(233), plt.imshow(reflect, 'gray'), plt.title('REFLECT')
    plt.subplot(234), plt.imshow(reflect101, 'gray'), plt.title('REFLECT_101')
    plt.subplot(235), plt.imshow(wrap, 'gray'), plt.title('WRAP')
    plt.subplot(236), plt.imshow(constant, 'gray'), plt.title('CONSTANT')
    plt.show()

if __name__ == '__main__':
    src = cv.imread("../images/CrystalLiu1.jpg")  # 读入图片放进src中
    cv.namedWindow("Crystal Liu")  # 创建窗口
    cv.imshow("Crystal Liu", src)  # 将src图片放入该创建的窗口中
    # roi_test(src)
    # fill_color_demo(src)
    fill_binary()
    img = cv.imread("../images/opencv_logo.png")
    make_border(img)
    cv.waitKey(0) # 等有键输入或者1000ms后自动将窗口消除，0表示只用键输入结束窗口
    cv.destroyAllWindows()  # 关闭所有窗口
