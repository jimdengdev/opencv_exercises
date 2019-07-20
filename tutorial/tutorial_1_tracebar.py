import cv2 as cv
import numpy as np


"""
任务：创建一个画板，可以自选各种颜色的画笔绘画各种图形。
"""

# 当鼠标按下时变为 True
drawing=False
# 如果mode为true绘制矩形。按下 'm'变成绘制曲线。
mode = True
ix, iy = -1, -1


# 创建回调函数
def draw_circle(event, x, y, flags, param):
    # 参数：滑动条的名字，滑动条被放置窗口的名字
    r = cv.getTrackbarPos('R', 'image')
    g = cv.getTrackbarPos('G', 'image')
    b = cv.getTrackbarPos('B', 'image')
    color = (b, g, r)

    global ix, iy, drawing, mode

    # 当按下左键是返回起始位置坐标
    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    # 当鼠标左键按下并移动是绘制图型。event 可以查看移动，flag 查看查看是否按下
    elif event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_LBUTTON:
        if drawing is True:
            if mode is True:
                cv.rectangle(img, (ix, iy), (x, y), color, -1)
            else:
                cv.circle(img, (x, y), 3, color, 1)
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


# 回调函数，什么也不做
def nothing(x):
    pass


img = cv.imread("../images/Crystal.jpg")
cv.namedWindow("image")
# 参数：滑动条的名字，滑动条被放置窗口的名字，滑动条的默认位置，滑动条的最大值，回调函数。
# 每次滑动条的滑动都会调用回调函数。回调函数通常都会含有一个默认参数，就是滑动条的位置。
# 在本例中这个函数不用做任何事情，我们只需要 pass 就可以了。
cv.createTrackbar('R','image',0,255,nothing)
cv.createTrackbar('G','image',0,255,nothing)
cv.createTrackbar('B','image',0,255,nothing)
cv.setMouseCallback('image', draw_circle)
while(1):
    cv.imshow('image',img)
    k = cv.waitKey(1)&0xFF
    if k == ord('m'):
        mode = not mode
    elif k == 27:
        break
