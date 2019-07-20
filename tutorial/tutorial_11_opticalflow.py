import cv2 as cv
import numpy as np

# 由于目标对象或者摄像机的移动造成的图像对象在连续两帧图像中的移动 被称为光流。
# 它是一个2D向量场，可以用来显示一个点从第一帧图像到第二 帧图像之间的移动。
# 光流在很多领域中都很有用
# • 由运动重建结构
# • 视频压缩
# • Video Stabilization 等
#
# 从使用者的角度来看，想法很简单，我们取跟踪一些点，然后我们就会获得这些点的光流向量。
# 但是还有一些问题。直到现在我们处理的都是很小的运动。
#  如果有大的运动怎么办呢？图像金字塔。我们可以使用图像金字塔的顶层，
# 此时小的运动被移除，大的运动装换成了小的运动，现在再使用 Lucas-Kanade 算法，我们就会得到尺度空间上的光流。


def optical_flow_demo():
    """
    函数目的：跟踪视频中的一些点。使用函数 cv2.goodFeatureToTrack()来确定要跟踪的点。
    首先在视频的第一帧图像中检测一些Shi-Tomasi角点，然后我们使用Lucas Kanade算法迭代跟踪这些角点。

    要给函数cv2.calcOpticlaFlowPyrLK()传入前一帧图像和其中的点，以及下一帧图像。
    函数将返回带有状态数的点， 如果状态数是 1，那说明在下一帧图像中找到了这个点（上一帧中角点），
    如果状态数是 0，就说明没有在下一帧图像中找到这个点。
    我们再把这些点作为参数传给函数，如此迭代下去实现跟踪。

    图像中的一些特征点甚至 在丢失以后，光流还会找到一个预期相似的点。
    所以为了实现稳定的跟踪，应该每个一定间隔就要进行一次角点检测。

    OpenCV 的官方示例中带有这样一个例子，它是每 5 帧进行一个特征点检测。
    它还对光流点使用反向检测来选取好的点进行跟踪
    """
    cap = cv.VideoCapture('../images/slow.mp4')

    # 用字典的方式传给goodFeaturesToTrack() 用来角点检测
    feature_params = dict( maxCorners = 100, qualityLevel = 0.3, minDistance = 7, blockSize = 7 )

    # maxLevel 为使用的图像金字塔层数
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
    # Create some random colors
    color = np.random.randint(0,255,(100,3))

    ret, old_frame = cap.read()
    old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
    p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    mask = np.zeros_like(old_frame)

    while True:
        ret, frame = cap.read()
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # cv.calcOpticalFlowPyrLK()参数说明：前一帧图像，吼一帧图像，前一帧的特征点
        # 返回参数说明，后一帧对应的特征点，状态值（如果前一帧特征点在后一帧存在st=1，不存在st=0），err:前后帧特征点误差
        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # 提取前后帧都存在的特征点
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        for i, (new, old) in enumerate(zip(good_new, good_old)):  # zip打包函数
            a, b = new.ravel()  # 将特征点分解开，得到对应的坐标点
            c, d = old.ravel()
            mask = cv.line(mask, (a, b), (c, d), color[i].tolist(), 2)  # 随机获得线条，圆的颜色
            frame = cv.circle(frame, (a, b), 5, color[i].tolist(), -1)
        img = cv.add(frame, mask)

        cv.imshow("frame",img)
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break

        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

    cv.destroyAllWindows()
    cap.release()


def Farneback():
    """
    ucas-Kanade 法是计算一些特征点的光流。OpenCV 还提供了一种计算稠密光流的方法。
    它会图像中的所有点的光流。这是基于 Gunner_Farneback 的算法 （2003 年）。
    结果是一个带有光流向量 （u，v）的双通道数组。通过计算我们能得到光流的大小和方向。
    使用颜色对结果进行编码以便于更好的观察。
    方向对应于 H（Hue）通道，大小对应 于 V（Value）通道
    :return:
    """
    cap = cv.VideoCapture("../images/vtest.avi")
    ret, frame1 = cap.read()
    prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255  # 将第二列都置为255


    while True:
        ret, frame2 = cap.read()
        next = cv.cvtColor(frame2,cv.COLOR_BGR2GRAY)
        flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
        rgb = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
        cv.imshow('frame2', rgb)
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break
        elif k == ord('s'):
            cv.imwrite('../images/opticalfb.png', frame2)
        cv.imwrite('../images/opticalhsv.png', rgb)
        prvs = next

    cap.release()
    cv.destroyAllWindows()


def main():
    # optical_flow_demo()
    Farneback()
    cv.waitKey(0)  # 等有键输入或者1000ms后自动将窗口消除，0表示只用键输入结束窗口
    cv.destroyAllWindows()  # 关闭所有窗口


if __name__ == '__main__':
    main()