""" opencv 基础入门 """
# 1. 环境搭配
## win10 + python3.5/3.6 + opencv3.3/或者更高 + pycharm2017
## pip install opencv-python
## pip install openc-contrib_python
## pip install pytesseract

# 2. 课程内容与目标
## opencv 图像读写模块
## opencv 图像处理模块
## opencv 各种编程技巧
## 有用案例
## Tesseract-OCR + opencv
## 解决项目中实际问题
# 创建窗口, 窗口可以调整大小， 但是标签改成cv2.WINDOW_NORMAL，也可调整窗口大小。
# 当图像维度太大， 或者要添加轨迹条时，调整窗口大小将会很有用
# cv.namedWindow("Crystal Liu")

# cv.destroyAllWindows()  # 关闭所有窗口

# cv2.waitKey() 是一个键盘绑定函数。需要指出的是它的时间尺度是毫 秒级。
# 函数等待特定的几毫秒，看是否有键盘输入。
# 特定的几毫秒之内，如果 按下任意键，这个函数会返回按键的 ASCII 码值，程序将会继续运行。
# 如果没 有键盘输入，返回值为 -1，如果我们设置这个函数的参数为 0，那它将会无限 期的等待键盘输入。
# 它也可以被用来检测特定键是否被按下，

# cv2.destroyAllWindows() 可以轻易删除任何我们建立的窗口。
# 如果 你想删除特定的窗口可以使用 cv2.destroyWindow()，在括号内输入你想删 除的窗口名。
import cv2 as cv

def test():
    # src = cv.imread(r"images/lena.jpg")  # 读入图片放进src中
    src = cv.imread("../images/lena.jpg")
    cv.imshow("Crystal", src)  # 将src图片放入该创建的窗口中
    cv.waitKey(1000) # 等有键输入或者1000ms后自动将窗口消除，0表示只用键输入结束窗口


if __name__ == '__main__':
    test()