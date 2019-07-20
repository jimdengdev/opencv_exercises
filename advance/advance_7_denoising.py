import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


# 可以选取包含目标像素的一个小窗口，然后在图像中搜索相似的窗口，最后求取所有窗口的平均值，并用这个值取代目标像素的值。
def denoising_1_demo(image):
    dst = cv.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    plt.subplot(121), plt.imshow(image)
    plt.subplot(122), plt.imshow(dst)
    plt.show()


def denoising_2_demo():
    """
    第一个参数是一个噪声帧的列表。
    第二个参数 imgtoDenoiseIndex 设定那些帧需要去噪，我们可以传入一 个帧的索引。
    第三个参数 temporaWindowSize 可以设置用于去噪的相邻帧的数目，它应该是一个奇数。
    在这种情况下 temporaWindowSize 帧的 图像会被用于去噪，中间的帧就是要去噪的帧。
    例如，我们传入 5 帧图像， imgToDenoiseIndex = 2 和 temporalWindowSize = 3。
    那么第一帧，第二帧， 第三帧图像将被用于第二帧图像的去噪

    """
    capture = cv.VideoCapture("../images/vtest.avi")
    image = [capture.read()[1] for i in range(5)]
    gray = [cv.cvtColor(i, cv.COLOR_BGR2GRAY) for i in image]
    gray = [np.float64(i) for i in gray]

    noise = np.random.randn(*gray[1].shape)*10
    noisy = [i + noise for i in gray]
    noisy = [np.uint8(np.clip(i, 0, 255)) for i in noisy ]

    # fastNlMeansDenoisingColoredMulti(srcImgs, imgToDenoiseIndex, temporalWindowSize,
    # dst=None, h=None, hColor=None, templateWindowSize=None, searchWindowSize=None):
    dst = cv.fastNlMeansDenoisingMulti(noisy, 2, 5, None, 4, 7, 35)
    plt.subplot(131),plt.imshow(gray[2],'gray')
    plt.subplot(132),plt.imshow(noisy[2],'gray')
    plt.subplot(133),plt.imshow(dst,'gray')
    plt.show()


def main():
    src = cv.imread("../images/lena.jpg")
    denoising_1_demo(src)
    # denoising_2_demo()
    cv.waitKey(0)  # 等有键输入或者1000ms后自动将窗口消除，0表示只用键输入结束窗口
    cv.destroyAllWindows()  # 关闭所有窗口


if __name__ == '__main__':
    main()