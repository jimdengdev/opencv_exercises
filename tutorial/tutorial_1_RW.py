# 图像的保存于加载
# 图像的基础知识
"""
你将要学习如下函数：cv.imread()，cv.imshow()，cv.imwrite()
"""
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


# 图像信息获得
def get_image_info(image):
    print("图像类型：",type(image))
    print("图像长x宽x通道数：",image.shape)
    print("图像长宽通道数相乘所得值：",image.size)
    print("图像像素值类型：",image.dtype)
    pixel_data = np.array(image)  # 将图片转换成数组
    print("像素大小：", pixel_data)


def save_image(image):
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)  # 将src图片转换成灰度图
    cv.imwrite("output.png", gray)  # 将转化后的灰度图写入到GrayCrystal.png中，可以用来转换图片格式


# 获取视频
def video_demo():
    # 参数可以是设备的索引号，或者是一个视频文件(没有声音)。
    # 设备索引号就是在指定要使用的摄像头。 一般的笔记本电脑都有内置摄像头。所以参数就是 0。
    # 你可以通过设置成 1 或 者其他的来选择别的摄像头
    # 你可以使用函数 capture.get(propId) 来获得视频的一些参数信息。
    # 这里 propId 可以是 0 到 18 之间的任何整数。
    # 其中的一些值可以使用 capture.set(propId,value) 来修改，value 就是 你想要设置成的新值。
    # 例如，我可以使用 capture.get(3) 和 cap.get(4) 来查看每一帧的宽和高。默认情况下得到的值是 640X480。
    # 但是我可以使用 ret=capture.set(3,320) 和 ret=capture.set(4,240) 来把宽和高改成 320X240
    capture = cv.VideoCapture(0)
    print("类型",type(capture))
    while True:
        ret, frame = capture.read()  # 获取相机图像，返回ret(结果为True/False)，和每一帧图片
        frame = cv.flip(frame, 1)  # 将图片水平翻转，竖直翻转为0
        print('1', ret)  # 打印出ret值
        cv.imshow("video", frame)  # 将每一帧图片放入video窗口

        # 警告：如果你用的是64位系统，你需要将k = cv.waitKey(0)这行改成 k = cv.waitKey(0)&0xFF。
        c = cv.waitKey(50) # 等有键输入(这里指c=Esc键)或者50ms后自动将窗口消除
        if c == 27:
            break


# 视频保存，通过创建一个VideoWriter对象。
# 我们应该确定一个输出文件 的名字。接下来指定 FourCC 编码（下面会介绍）。
# 播放频率和帧的大小也都需要确定。
# 最后一个是 isColor 标签。如果是 True，每一帧就是彩色图，否则就是灰度图
# FourCC 就是一个 4 字节码，用来确定视频的编码格式。
# 可用的编码列表 可以从fourcc.org查到。这是平台依赖的。
# 在Windows上常用的是DIVX。FourCC码以cv.VideoWriter_fourcc('D', 'I','V', 'X')形式传给程序
def save_video():
    cap = cv.VideoCapture(0)
    fourcc = cv.VideoWriter_fourcc('D', 'I','V', 'X')

    # 参数说明：输出视频名称，编码格式，播放频率，帧的大小
    out = cv.VideoWriter("../images/output.avi", fourcc, 50.0, (640, 480))

    while cap.isOpened(): # 你可以使用 cap.isOpened()，来检查是否成功初始化了
        ret, frame = cap.read()
        if ret is True:
            out.write(frame)

            cv.imshow('frame', frame)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    out.release()

if __name__ == '__main__':
    src = cv.imread("../images/CrystalLiu1.jpg")  # 读入图片放进src中
    cv.namedWindow("Crystal Liu", cv.WINDOW_AUTOSIZE)  # 创建窗口, 窗口尺寸自动调整
    cv.imshow("Crystal Liu", src)  # 将src图片放入该创建的窗口中
    # get_image_info(src)
    # video_demo()
    # save_image(src)
    save_video()

    # 与从摄像头中捕获一样，你只需要把设备索引号改成视频文件的名字。
    # 在播放每一帧时，使用 cv.waiKey() 设置适当的持续时间。
    # 如果设置的太低视频就会播放的非常快，如果设置的太高就会播放的很慢（你可以使用这种方法控制视频的播放速度）。
    # 通常情况下 25 毫秒就可以了。
    cv.waitKey(1000) # 等有键输入或者1000ms后自动将窗口消除，0表示只用键输入结束窗口
    cv.destroyAllWindows()  # 可以轻易删除任何我们建立的窗口。如果你想删除特定的窗口可以使用 cv.destroyWindow()，在括号内输入你想删除的窗口名。

