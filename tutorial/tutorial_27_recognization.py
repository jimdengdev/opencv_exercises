import cv2 as cv
import numpy as np
from PIL import Image
import pytesseract as tess

"""
验证码识别
1.步骤：
    1. 预处理-去除干扰线和点
    2.不同的结构元素中选择
    3. Image和numpy array相互转换
    4. 识别和输出 tess.image_to_string
2. 报错与处理
当出现该错误：raise TesseractNotFoundError() pytesseract.pytesseract.TesseractNotFoundError: tesseract is not installed or it's not in your path
不同系统采用不同策略：
On Linux
    sudo apt update
    sudo apt install tesseract-ocr
    sudo apt install libtesseract-dev
On Mac
    brew install tesseract
On Windows
    先下载tesseract包：https://github.com/UB-Mannheim/tesseract/wiki. 
    然后修改pytesseract.py中tesseract_cmd指向的路径：tesseract_cmd = 'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe'
    
references: https://pypi.org/project/pytesseract/ (INSTALLATION section) and https://github.com/tesseract-ocr/tesseract/wiki#installation
"""


def recognition_demo(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    cv.imshow("binary", binary)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (4, 4))
    bin1 = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel=kernel)
    cv.imshow("bin1", bin1)
    textImage = Image.fromarray(bin1)
    text = tess.image_to_string(textImage)

    print("The result:", text)


def main():
    src = cv.imread("../images/yzm.jpg")
    cv.imshow("demo", src)
    recognition_demo(src)
    cv.waitKey(0)  # 等有键输入或者1000ms后自动将窗口消除，0表示只用键输入结束窗口
    cv.destroyAllWindows()  # 关闭所有窗口


if __name__ == '__main__':
    main()