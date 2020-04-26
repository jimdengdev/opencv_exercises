# opencv_exercises
## 视频地址：https://www.bilibili.com/video/av24998616
### 代码主要是跟着贾志刚老师一行一行敲出来的。绝大部分API都查了资料、加了注释，基本上每行要注释的代码都注释了。

#### 下面是一些问题与解答，大家想要补充或者有什么疑问可以在Github上发issue或者b站发消息给我

### 疑问与解答：
0. github代码下载报错或者下载后解压报错
    - 解答：可以在项目"Clone or download"中选择https方式，用本地git工具下载：git clone https://github.com/Betterming/opencv_exercises.git
    - 方便大家下载使用，提供本项目压缩包链接：[opencv-exercises](https://cloud.189.cn/t/ZvENb2bE7BRf) （访问码：my7k）
1. 找不到包：ModuleNotFoundError：No module named 'cv2'
    - 解决：首先要安装opencv包  pip install opencv-python，若还没有解决，需要在pycharm中引入解释器环境，setting->Project Interpreter 点击Project Interpreter右侧锯齿选择python环境，可能需要重启pycharm
2. 报错：error: (-215:Assertion failed) size.width>0 && size.height>0 in function 'cv::imshow' 
    - 解决： 这种问题一般是因为图片/视频的路径有问题，路径做好不能有中文，注意不同系统之间路径可能表示不一样，可以在路径字符串前面加一个字符r
3. 验证码那节报错：raise TesseractNotFoundError() pytesseract.pytesseract.TesseractNotFoundError: tesseract is not installed or it's not in your path
    - 解答：不同系统采用不同策略：
    ```cmd
        On Linux
            sudo apt update
            sudo apt install tesseract-ocr
            sudo apt install libtesseract-dev
        On Mac
            brew install tesseract
        On Windows
            先下载tesseract包：https://github.com/UB-Mannheim/tesseract/wiki. 
            然后修改源码pytesseract.py中tesseract_cmd指向的路径：tesseract_cmd = 'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe'
    ```


# 目录
1. 概述与环境  tutorial_1_demo
2. 图像和视频读取与保存  tutorial_1_RW
3. 绘图函数  tutorial_1_draw
4. 把鼠标当画笔  tutorial_1_set_mouse
5. 用滑动条做调色板  tutorial_1_tracebar
6. numpy在图像处理中的基本使用  tutorial_2_numpy
7. 颜色空间  tutorial_3_colorspace
8. 像素运算  tutorial_4_Arithmetic
9. 图像roi与泛洪填充  tutorial_5_roi
10. 几何变换  tutorial_5_perspective_transform
11. 模糊操作  tutorial_6_7_blur
12. 边缘保留滤波  tutorial_8_EPF
13. 直方图  tutorial_9_10_histogram
14. 直方图反向投影  tutorial_11_backprojection
15. 模板匹配  tutorial_12_template
16. 图像二值化  tutorial_13_14_threshold
17. 图像金字塔  tutorial_15_pyramid
18. 图像梯度  tutorial_16_grad
19. Canny边缘检测  tutorial_17_canny
20. 直线检测和圆检测  tutorial_18_19_Hough
21. 轮廓发现  tutorial_20_contours
22. 对象测量  tutorial_21_measure
23. 膨胀与腐蚀  tutorial_22_erode_dilate
24. 开闭操作  tutorial_23_24_morphology
25. 分水岭算法  tutorial_25_watershed
26. 人脸检测  tutorial_26_face_detection
27. 数字验证码识别  tutorial_27_recognization
