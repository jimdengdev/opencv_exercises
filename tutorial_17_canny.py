import cv2 as cv
import numpy as np


# canny运算步骤：5步
# 1. 高斯模糊 - GaussianBlur
# 2. 灰度转换 - cvtColor
# 3. 计算梯度 - Sobel/Scharr
# 4. 非极大值抑制
# 5. 高低阈值输出二值图像

# 非极大值抑制：
# 算法使用一个3×3邻域作用在幅值阵列M[i,j]的所有点上；
# 每一个点上，邻域的中心像素M[i,j]与沿着梯度线的两个元素进行比较，
# 其中梯度线是由邻域的中心点处的扇区值ζ[i,j]给出。
# 如果在邻域中心点处的幅值M[i,j]不比梯度线方向上的两个相邻点幅值大，则M[i,j]赋值为零，否则维持原值；
# 此过程可以把M[i,j]宽屋脊带细化成只有一个像素点宽，即保留屋脊的高度值。

# 高低阈值连接
# T1，T2为阈值，凡是高于T2的都保留，凡是低于T1的都丢弃
# 从高于T2的像素出发，凡是大于T1而且相互连接的都保留。最终得到一个输出二值图像
# 推荐高低阈值比值为T2:T1 = 3:1/2:1,其中T2高阈值，T1低阈值


def edge_demo(image):
    blurred = cv.GaussianBlur(image, (3, 3), 0)
    gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)

    grad_x = cv.Sobel(gray, cv.CV_16SC1, 1, 0)
    grad_y = cv.Sobel(gray, cv.CV_16SC1, 0, 1)

    # edge_output = cv.Canny(grad_x, grad_y, 30, 150)
    edge_output = cv.Canny(gray, 50, 150)
    cv.imshow("gray", gray)
    cv.imshow("Canny demo", edge_output)


def main():
    src = cv.imread("Crystal.jpg")
    cv.imshow("demo",src)

    edge_demo(src)
    cv.waitKey(0)  # 等有键输入或者1000ms后自动将窗口消除，0表示只用键输入结束窗口
    cv.destroyAllWindows()  # 关闭所有窗口


if __name__ == '__main__':
    main()