import cv2 as cv
import numpy as np


"""
任务：根据我们选择的模式在拖动鼠标时绘制矩形或者是圆圈（就像画图程序中一样）。
所以我们的回调函数包含两部分，一部分画矩形，一部分画圆圈。
这是一个典型的例子他可以帮助我们更好理解与构建人机交互式程序，比如物体跟踪，图像分割等。
"""

# 当鼠标按下时变为 True
drawing=False
# 如果mode为true绘制矩形。按下 'm'变成绘制曲线。
mode = True
ix, iy = -1, -1


# 创建回调函数
def draw_circle(event, x, y, flags, param):
    global ix, iy, drawing, mode

    # 当按下左键是返回起始位置坐标
    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    # 当鼠标左键按下并移动是绘制图型。event 可以查看移动，flag 查看查看是否按下
    elif event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_LBUTTON:
        if drawing is True:
            if mode is True:
                cv.rectangle(img, (ix, iy), (x, y), (0, 255, 0), -1)
            else:
                cv.circle(img, (x, y), 3, (0,255,255), 1)
                # 下面注释掉的代码是起始点为圆心，起点到终点为半径的
                #  r=int(np.sqrt((x-ix)**2+(y-iy)**2))
                #  cv2.circle(img,(x,y),r,(0,0,255),-1)

    elif event == cv.EVENT_LBUTTONUP:
        drawing = False

        # if mode is True:
        #     cv.rectangle(img, (ix, iy), (x, y),(0, 255,0), -1)
        #
        # else:
        #     cv.circle(img, (x, y), 5, (0, 0, 255), -1)


img = cv.imread("Crystal.jpg")
cv.namedWindow("image")
cv.setMouseCallback('image', draw_circle)
while(1):
    cv.imshow('image',img)
    k = cv.waitKey(1)&0xFF
    if k == ord('m'):
        mode = not mode
    elif k == 27:
        break
