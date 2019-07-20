# coding:utf8
import cv2 as cv
import numpy as np
import time


def mean_shift_demo():
    """
    原理：假设我们有一堆点（比如直方 图反向投影得到的点），和一个小的圆形窗口，
    我们要完成的任务就是将这个窗 口移动到最大灰度密度处（或者是点最多的地方）。

    例如，初始窗口“C1”，圆心“C1_o”，而窗口中所有点质心却是“C1_r”，此时圆心和点的质心没有重合。
    所以移动圆心 C1_o 到质心 C1_r，这样我们就得到了一个新的窗口。再找新窗口内所有点的质心，
    大多数情况下还是不重合的，所以重复上面的操作：将新窗口的中心移动到新的质心。
    就这样不停的迭代操作直到窗口的中心和其所 包含点的质心重合为止（或者有一点小误差）。
    按照这样的操作我们的窗口最终会落在像素值（和）最大的地方。

    通常情况下我们要使用直方图方向投影得到的图像和目标对象的起始位置。
    当目标对象的移动会反映到直方图反向投影图中。
    就这样，meanshift 算法就把我们的窗口移动到图像中灰度密度最大的区域了。

    要在 OpenCV 中使用 Meanshift 算法首先我们要对目标对象进行设置， 计算目标对象的直方图，
    这样在执行 meanshift 算法时我们就可以将目标对象反向投影到每一帧中去了。
    另外我们还需要提供窗口的起始位置。在这里我们只计算 H（Hue）通道的直方图，
    同样为了避免低亮度造成的影响，我们使用函数 cv2.inRange() 将低亮度的值忽略掉。
    """
    # cap = cv.VideoCapture('cv.gif')
    cap = cv.VideoCapture("../images/slow.mp4")
    ret, frame = cap.read()  # 获取第一帧

    # 设置初始窗口位置
    # r, h, c, w = 97, 130, 450, 125
    r, h, c, w = 210, 80, 300, 125  # r,c 左上角坐标，h,w 高宽
    track_window = (c, r, w, h)

    # 设置兴趣区域
    roi = frame[r:r+h, c:c+w]

    hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)

    # 设置掩模，即剔除部分影响亮度的值
    mask = cv.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))

    # 计算直方图
    roi_hist = cv.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv.normalize(roi_hist, roi_hist,0, 255, cv.NORM_MINMAX)  # 归一化

    # 设置迭代条件，10次迭代，或至少移动一次
    term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)

    while True:
        ret, frame = cap.read()

        if ret is True:
            hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            dst = cv.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

            # 用meanShift算法获得新窗口，从而实现迭代
            ret, track_window = cv.meanShift(dst, track_window, term_crit)

            # 在图片中画出该识别出来的窗口
            x, y, w, h = track_window
            img2 = cv.rectangle(frame, (x, y), (x + w, y + h), 255, 2)
            cv.imshow('img2', img2)

            c = cv.waitKey(60) & 0xff
            if c == 27:
                break
            else:
                print(c)
                print(time.strftime('%H_%M_%S',time.localtime(time.time())))
                # cv.imwrite('../images/'+time.strftime('%H_%M_%S',time.localtime(time.time()))+'.jpg', img2)
        else:
            break


def cam_shift_demo():
    """
    我们的窗口的大小是固定的，而汽车由远及近（在视觉上）是一个逐渐变大的过程，固定的窗口是不合适的。
    所以我们需要根据目标的大小和角度来对窗口的大小和角度进行修订。
    OpenCVLabs 为我们带来的解决方案（1988 年）：一个被叫做 CAMshift的算法。
    这个算法首先要使用 meanshift，meanshift 找到（并覆盖）目标之后， 再去调整窗口的大小。
    它还会计算目标对象的最佳外接椭圆的 角度，并以此调节窗口角度。
    然后使用更新后的窗口大小和角度来在原来的位 置继续进行 meanshift。
    重复这个过程知道达到需要的精度。
    OpenCV中Camshift与 Meanshift 基本一样，但是返回的结果是一个带旋转角度的矩形（这是 我们的结果），
    以及这个矩形的参数（被用到下一次迭代过程中）。
    """
    cap = cv.VideoCapture("../images/slow.mp4")
    ret, frame = cap.read()  # 获取第一帧

    # 设置初始窗口位置
    r, h, c, w = 210, 80, 300, 125  # r,c 左上角坐标，h,w 高宽
    track_window = (c, r, w, h)

    # 设置兴趣区域
    roi = frame[r:r + h, c:c + w]

    hsv_roi = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # 设置掩模，即剔除部分影响亮度的值
    mask = cv.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))

    # 计算直方图
    roi_hist = cv.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)  # 归一化

    # 设置迭代条件，10次迭代，或至少移动一次
    term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)

    while True:
        ret, frame = cap.read()

        if ret is True:
            hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            dst = cv.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

            # 用meanShift算法获得新窗口，从而实现迭代
            ret, track_window = cv.CamShift(dst, track_window, term_crit)
            print(ret)

            # ret 为元组类型，分别为矩形左上角和右下角，以及旋转角度，
            # 用boxPoints()和polylines()恰好可以画出这个旋转矩形
            pts = cv.boxPoints(ret)
            pts = np.int0(pts)
            img2 = cv.polylines(frame, [pts], True, 255, 2)
            cv.imshow('img2', img2)

            c = cv.waitKey(60) & 0xff
            if c == 27:
                break
            else:
                print(c)
                print(time.strftime('%H_%M_%S', time.localtime(time.time())))
                # cv.imwrite('../images/'+ time.strftime('%H_%M_%S', time.localtime(time.time())) + '.jpg', img2)
        else:
            break


def main():
    # mean_shift_demo()
    cam_shift_demo()
    cv.waitKey(0)  # 等有键输入或者1000ms后自动将窗口消除，0表示只用键输入结束窗口
    cv.destroyAllWindows()  # 关闭所有窗口


if __name__ == '__main__':
    main()