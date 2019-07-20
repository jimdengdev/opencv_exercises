import cv2 as cv
import numpy as np

def main():
    src = cv.imread("../images/blox.jpg")

    cv.waitKey(0)  # 等有键输入或者1000ms后自动将窗口消除，0表示只用键输入结束窗口
    cv.destroyAllWindows()  # 关闭所有窗口


if __name__ == '__main__':
    main()