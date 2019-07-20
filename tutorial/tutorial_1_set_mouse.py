import cv2 as cv
import numpy as np


# 查看所有被支持的鼠标事件。
def search_event():
    events = [i for i in dir(cv) if 'EVENT' in i]
    print(events)


if __name__ == '__main__':

    img = cv.imread("../images/CrystalLiu1.jpg")  # 读入图片放进src中
    cv.namedWindow("image", cv.WINDOW_AUTOSIZE)  # 创建窗口, 窗口尺寸自动调整
    # search_event()
    # 创建图像与窗口并将窗口与回调函数绑定
    def draw_circle(event, x, y, flags, param):
        if event==cv.EVENT_LBUTTONDBLCLK:
            cv.circle(img,(x,y),100,(255,255 ,0),2)

    cv.setMouseCallback('image', draw_circle)

    while True:
        cv.imshow('image', img)
        if cv.waitKey(20) & 0xFF == 27:
            break

    cv.destroyAllWindows()

