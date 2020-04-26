import cv2 as cv
import numpy as np


# 模板匹配，就是在整个图像区域发现与给定子图像匹配的小块区域，
# 需要模板图像T和待检测图像-源图像S
# 工作方法：在待检测的图像上，从左到右，从上倒下计算模板图像与重叠子图像匹配度，
# 匹配度越大，两者相同的可能性越大。
def template_demo():
    tpl = cv.imread("../images/rabbit.jpg")
    target = cv.imread("../images/CrystalLiu22.jpg")
    cv.imshow("template", tpl)
    cv.imshow("target", target)

    methods = [cv.TM_SQDIFF_NORMED, cv.TM_CCORR_NORMED, cv.TM_CCOEFF_NORMED]  # 三种模板匹配方法
    th, tw = tpl.shape[:2]

    for md in methods:
        print(md)
        result = cv.matchTemplate(target, tpl, md)  # 得到匹配结果
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
        if md == cv.TM_SQDIFF_NORMED:  # cv.TM_SQDIFF_NORMED最小时最相似，其他最大时最相似
            tl = min_loc
        else:
            tl = max_loc

        br = (tl[0] + tw, tl[1] + th)
        cv.rectangle(target, tl, br, (0, 0, 255), 2)  # tl为左上角坐标，br为右下角坐标，从而画出矩形
        cv.imshow("match-"+np.str(md), target)


if __name__ == '__main__':

    src = cv.imread("../images/CrystalLiu1.jpg")  # 读入图片放进src中
    cv.namedWindow("Crystal Liu")  # 创建窗口
    cv.imshow("Crystal Liu", src)  # 将src图片放入该创建的窗口中
    template_demo()

    cv.waitKey(0) # 等有键输入或者1000ms后自动将窗口消除，0表示只用键输入结束窗口
    cv.destroyAllWindows()  # 关闭所有窗口