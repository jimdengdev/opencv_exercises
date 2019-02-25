import cv2 as cv
import numpy as np
from PIL import Image
import pytesseract as tess


"""
知识点：
预处理-去除干扰线和点
不同的结构元素中选择
Image和numpy array相互转换
识别和输出
"""


def recognition_demo(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    cv.imshow("binary", binary)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (4, 4))
    bin1 = cv.morphologyEx(binary,cv.MORPH_OPEN, kernel=kernel)
    cv.imshow("bin1", bin1)
    textImage = Image.fromarray(bin1)
    text = tess.image_to_string(textImage)

    print("The result:", text)



def main():
    src = cv.imread("yzm.jpg")
    cv.imshow("demo",src)
    recognition_demo(src)
    cv.waitKey(0)  # 等有键输入或者1000ms后自动将窗口消除，0表示只用键输入结束窗口
    cv.destroyAllWindows()  # 关闭所有窗口


if __name__ == '__main__':
    main()