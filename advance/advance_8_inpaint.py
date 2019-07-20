import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


"""
为了实现老照片修复，科学家们已经提出了好几种算法，OpenCV 提供了其 中的两种。
这两种算法都可以通过使用函数 cv2.inpaint() 来实施。 第一个算法是根据 Alexandru_Telea 在 2004 发表的文章实现的。
它是 基于快速行进算法的。以图像中一个要修补的区域为例。算法从这个区域的边界开始向区域内部慢慢前进，首先填充区域边界像素。
它要选取待修补像素周 围的一个小的邻域，使用这个邻域内的归一化加权和更新待修复的像素值。权 重的选择是非常重要的。
对于靠近带修复点的像素点，靠近正常边界像素点和 在轮廓上的像素点给予更高的权重。
当一个像素被修复之后，使用快速行进算法（FMM）移动到下一个最近的像素。
FMM 保证了靠近已知（没有退化的）像素点的坏点先被修复，这与手工启发式操作比较类似。
可以通过设置标签参 数为 cv2.INPAINT_TELEA 来使用此算法。 

第二个算法是根据Bertalmio,Marcelo,Andrea_L.Bertozzi,和Guillermo_Sapiro 在 2001 年发表的文章实现的。
这个算法是基于流体动力学并使用了偏微分方 程。基本原理是启发式的。
它首先沿着正常区域的边界向退化区域的前进（因 为边界是连续的，所以退化区域非边界与正常区域的边界应该也是连续的）。
它通过匹配待修复区域中的梯度向量来延伸等光强线（isophotes，由灰度值相 等的点练成的线）。
为了实现这个目的，作者是用来流体动力学中的一些方法。
完成这一步之后，通过填充颜色来使这个区域内的灰度值变化最小。
可以通过设置标签参数为 cv2.INPAINT_NS 来使用此算法。
"""
def inpaint(img1, img2):
    dst =  cv.inpaint(img1, img2, 3, cv.INPAINT_TELEA)
    cv.imshow('dst', dst)


def main():
    img1 = cv.imread("../images/Crystal.jpg", 0)
    img2 = cv.imread("../images/mask_Crystal.jpg", 0)
    cv.imshow("Crystal", img1)
    cv.imshow("Mask", img2)
    inpaint(img1, img2)


    cv.waitKey(0)  # 等有键输入或者1000ms后自动将窗口消除，0表示只用键输入结束窗口
    cv.destroyAllWindows()  # 关闭所有窗口


if __name__ == '__main__':
    main()