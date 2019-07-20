import cv2 as cv
import numpy as np


# 颜色空间转换，从bgr到gray，hsv，yuv，ycrcb
def color_space_demo(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    cv.imshow("gray", gray)

    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    cv.imshow("hsv", hsv)

    yuv = cv.cvtColor(image, cv.COLOR_BGR2YUV)
    cv.imshow("yuv", yuv)

    ycrcb = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)
    cv.imshow("ycrcb", ycrcb)


# 从视频中提取指定颜色范围，并将其置为白，其余置为黑
def extract_object_demo():
    capture = cv.VideoCapture("cv.gif")
    while True:
        ret, frame = capture.read()

        if ret is False:  # 如果没有获取到视频帧则返回false
            break
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        lower_hsv = np.array([0, 43, 46])  # hsv中h，s，v的最小值
        upper_hsv = np.array([50, 255, 255])  # hsv中的h，s，v最大值

        # 提取指定范围颜色，保留指定范围颜色, 其余置为黑(0)
        mask = cv.inRange(hsv, lowerb=lower_hsv, upperb=upper_hsv)  # 用inRange函数提取指定颜色范围，这里对hsv来处理
        cv.imshow("video", frame)
        cv.imshow("mask", mask)

        c = cv.waitKey(40)
        if c == 27:
            break


# 对图片三个通道颜色提取并放在三张图片中
def channels_split_merge(image):
    b, g, r = cv.split(image)  # b通道提取时，对该通道颜色保留，其余通道置为0
    cv.imshow("blue", b)
    cv.imshow("green", g)
    cv.imshow("red", r)

    changed_image = image.copy()
    changed_image[:, :, 2] = 0  # 将r通道颜色全部置为0
    cv.imshow("changed_image", changed_image)

    merge_image = cv.merge([b, g, r])
    cv.imshow("merge_image", merge_image)

if __name__ == '__main__':

    src = cv.imread("../images/CrystalLiu1.jpg")  # 读入图片放进src中
    cv.namedWindow("Crystal Liu", cv.WINDOW_AUTOSIZE)  # 创建窗口, 窗口尺寸自动调整
    cv.imshow("Crystal Liu", src)  # 将src图片放入该创建的窗口中
    # color_space_demo(src)
    # extract_object_demo()
    channels_split_merge(src)
    cv.waitKey(0)

    cv.destroyAllWindows()