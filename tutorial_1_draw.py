import cv2 as cv
import numpy as np


"""
绘图函数：cv2.line()，cv2.circle()，cv2.rectangle()， cv2.ellipse()，cv2.putText() 等。

"""


def draw_demo(img):
    # line(img, pt1, pt2, color[, thickness[, lineType[, shift]]]) -> img
    # pt1,pt2起点终点。
    # thickness线宽。如果给一个闭合图形设置为 -1，那么这个图形就会被填充。默认值是 1.
    # ：线条的类型， 8连接，抗锯齿等。默认情况是8连接。cv2.LINE_AA 为抗锯齿，这样看起来会非常平滑
    cv.line(img, (0, 0), (511, 511), (0, 0, 255), 5)
    cv.rectangle(img,(384,0),(510,128),(0,255,0),3)
    cv.circle(img,(447,63), 63, (0,0,255), -1)  # 中心点，半径，颜色，填充

    # 椭圆一个参数是中心点的位置坐标。 下一个参数是长轴和短轴的长度。椭圆沿逆时针方向旋转的角度。
    # 椭圆弧沿着顺时针方向起始的角度和结束角度（只画出一部分椭圆），如果是 0 很 360，就是整个椭圆。
    cv.ellipse(img,(256,256),(100,50),0,0,180,255,-1)

    # 这里 reshape 的第一个参数为 -1, 表明这一维的长度是根据后面的维度的计算出来的。
    # 注意：如果第三个参数是 False，我们得到的多边形是不闭合的（首尾不相 连）。
    # 注意：cv2.polylines() 可以被用来画很多条线。
    # 只需要把想要画的线放在一 个列表中，将这个列表传给函数就可以了。每条线都会被独立绘制。
    # 这会比用 cv2.line() 一条一条的绘制要快一些
    pts = np.array([[10, 5], [20, 30], [70, 20], [50, 10]], np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv.polylines(img,pts=pts,isClosed=True, color=(255, 255, 255), thickness=3)

    """
    要在图片上绘制文字，你需要设置下列参数： 
    • 你要绘制的文字 
    • 你要绘制的位置 
    • 字体类型（通过查看 cv2.putText() 的文档找到支持的字体） 
    • 字体的大小 
    • 文字的一般属性如颜色，粗细，线条的类型等。
    为了更好看一点推荐使用 linetype=cv2.LINE_AA。 
    
    警告：所有的绘图函数的返回值都是 None，
    所以不能使用 img = cv2.line(img,(0,0),(511,511),(255,0,0),5)。
    """
    font = cv.FONT_HERSHEY_SIMPLEX
    cv.putText(img, 'OpenCV', (10, 500), font, 4, (255, 255, 255), 2)
    cv.imshow("example", img)


def main():
    cv.namedWindow("example")
    img = np.zeros((512, 512, 3), np.uint8)  # 创建一张图片
    draw_demo(img)

    cv.waitKey(0)  # 等有键输入或者1000ms后自动将窗口消除，0表示只用键输入结束窗口
    cv.destroyAllWindows()  # 关闭所有窗口


if __name__ == '__main__':
    main()