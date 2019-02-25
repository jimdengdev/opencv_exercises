import cv2 as cv
import numpy as np


def edge_demo(image):
    blurred = cv.GaussianBlur(image, (3, 3), 0)
    gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)

    grad_x = cv.Sobel(gray, cv.CV_16SC1, 1, 0)
    grad_y = cv.Sobel(gray, cv.CV_16SC1, 0, 1)

    # edge_output = cv.Canny(grad_x, grad_y, 30, 150)
    edge_output = cv.Canny(gray, 50, 150)
    return edge_output


def contours_demo(image):
    """
    . 轮廓可以简单认为成将连续的点（连着边界）连在一起的曲线，具有相同 的颜色或者灰度。
    轮廓在形状分析和物体的检测和识别中很有用。
    . 为了更加准确，要使用二值化图像。在寻找轮廓之前，要进行阈值化处理或者 Canny 边界检测
    . 查找轮廓的函数会修改原始图像。如果你在找到轮廓之后还想使用原始图像的话，
    你应该将原始图像存储到其他变量中.
    . 在 OpenCV 中，查找轮廓就像在黑色背景中超白色物体。要找的物体应该是白色而背景应该是黑色。


    """
    # dst = cv.GaussianBlur(image, (3, 3), 0)
    # gray = cv.cvtColor(dst, cv.COLOR_BGR2GRAY)
    # ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    # cv.imshow("binary image", binary)
    binary = edge_demo(image)
    """
    • 函数 cv2.ﬁndContours() 有三个参数, 第一个是输入图像，第二个是轮廓检索模式，第三个是轮廓近似方法。
    • 检索模式：
        • CV_RETR_EXTERNAL - 只提取外层的轮廓  
        • CV_RETR_LIST - 提取所有轮廓，并且放置在 list 中  
        • CV_RETR_CCOMP - 提取所有轮廓，并且将其组织为两层的 hierarchy: 
            顶层为连通域的 外围边界，次层为洞的内层边界。  
        • CV_RETR_TREE - 提取所有轮廓，并且重构嵌套轮廓的全部 hierarchy
    • 逼近方法 (对所有节点, 不包括使用内部逼近的 CV_RETR_RUNS).  点的存贮情况，是不是都被存贮
        • CV_CHAIN_CODE - Freeman 链码的输出轮廓. 其它方法输出多边形(定点序列).  
        • CV_CHAIN_APPROX_NONE - 将所有点由链码形式翻译为点序列形式  
        • CV_CHAIN_APPROX_SIMPLE - 压缩水平、垂直和对角分割，即函数只保留末端的象素 点;  
        • CV_CHAIN_APPROX_TC89_L1,  CV_CHAIN_APPROX_TC89_KCOS - 应用 Teh-Chin 链逼近算法.  
        • CV_LINK_RUNS - 通过连接为 1 的水平碎片使用完全不同的轮廓提取算法。仅有 CV_RETR_LIST 提取模式可以在本方法中应用.  
    • offset:每一个轮廓点的偏移量. 当轮廓是从图像 ROI 中提取出来的时候，使用偏移量有用，因为可以从整个图像上下文来对轮廓做分析. 
    • 返回值有三个，第一个是图像，第二个是轮廓，第三个是（轮廓的）层析结构。
        轮廓（第二个返回值）是一个 Python 列表，其中存储这图像中的所有轮廓。
        每一个轮廓都是一个 Numpy 数组，包含对象边界点（x，y）的坐标。
    """
    contours, hierarchy= cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for i, contour in enumerate(contours):
        # 函数 cv2.drawContours() 可以被用来绘制轮廓。它可以根据你提供的边界点绘制任何形状。
        # 它的第一个参数是原始图像，第二个参数是轮廓，一个 Python 列表。
        # 第三个参数是轮廓的索引（在绘制独立轮廓是很有用，当设 置为 -1 时绘制所有轮廓）。
        # 接下来的参数是轮廓的颜色和厚度等。
        cv.drawContours(image, contours, i, (0, 0, 255), 2)  # 2为像素大小，-1时填充轮廓
        print(i)
    cv.imshow("detect contours", image)


def main():
    src = cv.imread("circle.png")
    cv.imshow("demo",src)
    contours_demo(src)
    cv.waitKey(0)  # 等有键输入或者1000ms后自动将窗口消除，0表示只用键输入结束窗口
    cv.destroyAllWindows()  # 关闭所有窗口


if __name__ == '__main__':
    main()