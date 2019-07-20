import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 学习对图像进行各种几个变换，例如移动，旋转，仿射变换等。
# 函数为：cv2.getPerspectiveTransform
# 代码参考：https://blog.csdn.net/songchunxiao1991/article/details/80226510
# 变换 OpenCV提供了两个变换函数，cv2.warpAﬃne和cv2.warpPerspective，
# 使用这两个函数你可以实现所有类型的变换。
# cv2.warpAﬃne 接收的参数是 2×3 的变换矩阵，而 cv2.warpPerspective 接收的参数是 3×3 的变换矩阵。


# 扩展缩放只是改变图像的尺寸大小。OpenCV 提供的函数 cv2.resize() 可以实现这个功能。
# 图像的尺寸可以自己手动设置，你也可以指定缩放因子。
# 我们可以选择使用不同的插值方法。在缩放时我们推荐使用cv2.INTER_AREA，
# 在扩展时我们推荐使用 v2.INTER_CUBIC（慢) 和 v2.INTER_LINEAR。
#  默认情况下所有改变图像尺寸大小的操作使用的插值方法都是cv2.INTER_LINEAR。
def resize_demo(image):

    print("Origin size:", image.shape)
    # 第一种方法：通过fx，fy缩放因子
    res = cv.resize(image, None, fx=2, fy=2, interpolation=cv.INTER_CUBIC)
    print("After resize 1 size:", res.shape)
    # 第二种方法：直接设置输出图像的尺寸，所以不用设置缩放因子
    height,width = image.shape[:1]
    res=cv.resize(image,(2*width,2*height),interpolation=cv.INTER_CUBIC)
    print("After resize 2 size:", res.shape)

    while(1):
        cv.imshow('res',res)
        cv.imshow('img',image)
        if cv.waitKey(1) & 0xFF == 27:
            break


# 图像偏移：M = np.array([[1, 0, tx], [0, 1, ty]])
def move_demo(image):
    rows, cols = image.shape[:2]
    M = np.float32([[1, 0, 100], [0, 1, 50]])
    dst = cv.warpAffine(image, M, (cols, rows))
    cv.imshow('image', dst)


def rotation_demo(img):
    rows, cols = img.shape[:2]
    # 将图像相对于中心旋转90度，而不进行任何缩放。旋转中心，角度，缩放比率
    M = cv.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)
    dst = cv.warpAffine(img, M, (cols, rows))
    cv.imshow('original', img)
    cv.imshow('result', dst)
    cv.waitKey(0)
    cv.destroyAllWindows()


# 仿射变换
# 在仿射变换中，原始图像中的所有平行线在输出图像中仍然是平行的。
# 为了找到变换矩阵，我们需要输入图像中的三个点以及它们在输出图像中的相应位置。
# 然后cv2.getAffineTransform将创建一个2x3矩阵，并将其传递给cv2.warpAffine。
def affine_demo(img):

    rows, cols, ch = img.shape

    pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
    pts2 = np.float32([[10, 100], [200, 50], [100, 250]])

    M = cv.getAffineTransform(pts1, pts2)

    dst = cv.warpAffine(img, M, (cols, rows))

    plt.subplot(121), plt.imshow(img), plt.title('Input')
    plt.subplot(122), plt.imshow(dst), plt.title('Output')
    plt.show()


# 透视转化 对于透视变换，您需要一个3x3变换矩阵。 即使在改造之后，直线仍将保持直线。
# 要找到这个变换矩阵，您需要输入图像上的4个点和输出图像上的对应点。 在这4点中，其中3个不应该在线。
# 然后可以通过函数cv2.getPerspectiveTransform找到变换矩阵。
# 然后将cv2.warpPerspective应用于这个3x3转换矩阵。
def perspective_demo(img):

    rows, cols, ch = img.shape

    pts1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])
    pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])

    M = cv.getPerspectiveTransform(pts1, pts2)

    dst = cv.warpPerspective(img, M, (300, 300))

    plt.subplot(121), plt.imshow(img), plt.title('Input')
    plt.subplot(122), plt.imshow(dst), plt.title('Output')
    plt.show()


def main():
    src = cv.imread("../images/Crystal.jpg")
    cv.imshow("demo",src)
    # resize_demo(src)
    # move_demo(src)
    # rotation_demo(src)
    # affine_demo(src)
    perspective_demo(src)
    cv.waitKey(0)  # 等有键输入或者1000ms后自动将窗口消除，0表示只用键输入结束窗口
    cv.destroyAllWindows()  # 关闭所有窗口


if __name__ == '__main__':
    main()