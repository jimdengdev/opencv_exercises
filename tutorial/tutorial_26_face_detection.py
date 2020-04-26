import cv2 as cv

"""
使用Haar分类器进行面部检测

1. 简单介绍Haar特征分类器对象检测技术
    它是基于机器学习的，通过使用大量的正负样本图像训练得到一个cascade_function，最后再用它来做对象检测。
    如果你想实现自己的面部检测分类器，需要大量的正样本图像（面部图像）和负样本图像（不含面部的图像）来训练分类器。
    可参考https://docs.opencv.org/2.4/doc/user_guide/ug_traincascade.html，这里不做介绍，现在我们利用
    OpenCV已经训练好的分类器，直接利用它来实现面部和眼部检测。
    
2. 主要步骤：
    1）加载xml分类器，并将图像或者视频处理成灰度格式 cv.CascadeClassifier()
    2）对灰度图像进行面部检测，返回若干个包含面部的矩形区域 Rect（x,y,w,h）face_detector.detectMultiScale()
    3）创建一个包含面部的ROI，并在其中进行眼部检测
    
3. 重要方法分析：def detectMultiScale(self, image, scaleFactor=None, minNeighbors=None, minSize=None, maxSize=None)
    原理：检测输入图像在不同尺寸下可能含有的目标对象
#minSize – Minimum possible object size. Objects smaller than that are ignored.
#maxSize – Maximum possible object size. Objects larger than that are ignored.
    入参：
        1）image：输入的图像
        2）scaleFactor：比例因子，图像尺寸每次减少的比例，要大于1，这个需要自己手动调参以便获得想要的结果
        3）minNeighbors：最小附近像素值，在每个候选框边缘最小应该保留多少个附近像素
        4）minSize，maxSize：最小可能对象尺寸，所检测的结果小于该值会被忽略。最大可能对象尺寸，所检测的结果大于该值会被忽略
    返回：若干个包含对象的矩形区域
"""


def face_detection(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    face_detector = cv.CascadeClassifier("../images/haarcascade_frontalface_alt_tree.xml")
    eyes_detector = cv.CascadeClassifier("../images/haarcascade_eye.xml")
    faces = face_detector.detectMultiScale(gray, 1.01, 5)
    for x, y, w, h in faces:
        img = cv.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eyes_detector.detectMultiScale(roi_gray, 1.3, 5)

        for ex, ey, ew, eh in eyes:
            cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + ey), (0, 255, 0), 2)

    cv.imshow("result", image)


def main():
    src = cv.imread("../images/CrystalLiu1.jpg")
    cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)

    cv.imshow("input image", src)
    face_detection(src)

    # 打开摄像头进行视频检测
    # capture = cv.VideoCapture(0)
    # cv.namedWindow("result", cv.WINDOW_AUTOSIZE)
    # while True:
    #     ret, frame = capture.read()
    #     # frame = cv.flip(frame, 0)
    #     face_detection(frame)
    #     c = cv.waitKey(10)
    #     if c == 27:
    #         break

    cv.waitKey(0)
    cv.destroyAllWindows()  # 关闭所有窗口


if __name__ == '__main__':
    main()
