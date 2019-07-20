import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 二值图像就是将灰度图转化成黑白图，没有灰，在一个值之前为黑，之后为白
# 有全局和局部两种
# 在使用全局阈值时，我们就是随便给了一个数来做阈值，那我们怎么知道我们选取的这个数的好坏呢？答案就是不停的尝试。
# 如果是一副双峰图像（简 单来说双峰图像是指图像直方图中存在两个峰）呢？
# 我们岂不是应该在两个峰之间的峰谷选一个值作为阈值？这就是 Otsu 二值化要做的。
# 简单来说就是对 一副双峰图像自动根据其直方图计算出一个阈值。
# （对于非双峰图像，这种方法 得到的结果可能会不理想）。


def threshold_demo(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # 这个函数的第一个参数就是原图像，原图像应该是灰度图。
    # 第二个参数就是用来对像素值进行分类的阈值。
    # 第三个参数就是当像素值高于（有时是小于）阈值时应该被赋予的新的像素值
    # 第四个参数来决定阈值方法，见threshold_simple()
    # ret, binary = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
    ret, binary = cv.threshold(gray, 127, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    print("threshold value: %s"%ret)
    cv.imshow("threshold_demo", binary)


def threshold_simple(image):
    img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, thresh1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
    ret, thresh2 = cv.threshold(img, 127, 255, cv.THRESH_BINARY_INV)
    ret, thresh3 = cv.threshold(img, 127, 255, cv.THRESH_TRUNC)
    ret, thresh4 = cv.threshold(img, 127, 255, cv.THRESH_TOZERO)
    ret, thresh5 = cv.threshold(img, 127, 255, cv.THRESH_TOZERO_INV)
    titles = ['Original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
    images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

    for i in range(6):
        plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray')  # 将图像按2x3铺开
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])

    plt.show()


# 在前面的部分我们使用是全局阈值，整幅图像采用同一个数作为阈值。
# 当时这种方法并不适应与所有情况，尤其是当同一幅图像上的不同部分的具有不同亮度时。
# 这种情况下我们需要采用自适应阈值。此时的阈值是根据图像上的 每一个小区域计算与其对应的阈值。
# 因此在同一幅图像上的不同区域采用的是不同的阈值，从而使我们能在亮度不同的情况下得到更好的结果。
# 这种方法需要我们指定三个参数，返回值只有一个
# _MEAN_C：阈值取自相邻区域的平均值,_GAUSSIAN_C：阈值取值相邻区域 的加权和，权重为一个高斯窗口。
# Block Size - 邻域大小（用来计算阈值的区域大小）。
# C - 这就是是一个常数，阈值就等于的平均值或者加权平均值减去这个常数。

def threshold_adaptive(image):
    img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # 中值滤波
    img = cv.medianBlur(img,5)

    ret, th1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
    # 11 为 Block size, 2 为 C 值
    th2 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
    th3 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)

    titles = ['Original Image', 'Global Threshold (v = 127)', 'Adaptive Mean Threshold', 'Adaptive Gaussian Threshold']
    images = [img, th1, th2, th3]

    for i in range(4):
        plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])

    plt.show()


def threshold_custom(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    m = np.reshape(gray, [1, w*h])
    mean = m.sum() / (w*h)  # 求出整个灰度图像的平均值
    print("mean:", mean)
    ret, binary = cv.threshold(gray, mean, 255, cv.THRESH_BINARY)
    cv.imshow("threshold_custom", binary)


# 将大图片拆分成小图片后再用自适应局部阈值比较好
def big_image_demo(image):
    print(image.shape)
    cw = 200
    ch = 200
    h, w = image.shape[:2]
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    cv.imshow("big_image_demo_gray", gray)

    # 将一张图片每隔ch * cw分成一份
    for row in range(0, h, ch):
        for col in range(0, w, cw):
            roi = gray[row:row+ch, col:col+cw]
            dst = cv.adaptiveThreshold(roi, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 127, 2)
            gray[row:row + ch, col:col + cw] = dst
            print(np.std(dst), np.mean(dst))

    cv.imwrite("../images/result_big_image.png", gray)


def main():
    img = cv.imread("../images/02.jpg")
    # threshold_demo(img)
    # threshold_simple(img)
    # threshold_adaptive(img)
    # threshold_custom(img)
    src = cv.imread("../images/big_image.jpg")
    big_image_demo(src)
    cv.waitKey(0)  # 等有键输入或者1000ms后自动将窗口消除，0表示只用键输入结束窗口
    cv.destroyAllWindows()  # 关闭所有窗口


if __name__ == '__main__':
    main()