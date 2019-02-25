import cv2 as cv
import numpy as np


# 图像金字塔和拉普拉斯金字塔(L1 = g1 - expand(g2))：reduce：高斯模糊+降采样，expand：扩大+卷积
# PyrDown降采样，PyrUp还原
def pyramid_demo(image):
    level = 4
    temp = image.copy()
    pyramid_images = []

    for i in range(level):
        dst = cv.pyrDown(temp)
        pyramid_images.append(dst)
        cv.imshow("pyramid_down_"+str(i+1), dst)
        temp = dst.copy()
    return pyramid_images


def laplace_demo(image):  # 注意：图片必须是满足2^n这种分辨率
    pyramid_images = pyramid_demo(image)
    level = len(pyramid_images)

    for i in range(level-1, -1, -1):
        if i-1 < 0:
            expand  = cv.pyrUp(pyramid_images[i], dstsize=image.shape[:2])
            lpls = cv.subtract(image, expand)
            cv.imshow("laplace_demo"+str(i), lpls)
        else:
            expand = cv.pyrUp(pyramid_images[i], dstsize=pyramid_images[i-1].shape[:2])
            lpls = cv.subtract(pyramid_images[i-1], expand)
            cv.imshow("laplace_demo"+str(i), lpls)


src = cv.imread("lena.jpg")  # 读入图片放进src中
cv.imshow("demo", src)  # 将src图片放入该创建的窗口中
# pyramid_demo(src)
laplace_demo(src)
cv.waitKey(0) # 等有键输入或者1000ms后自动将窗口消除，0表示只用键输入结束窗口
cv.destroyAllWindows()  # 关闭所有窗口