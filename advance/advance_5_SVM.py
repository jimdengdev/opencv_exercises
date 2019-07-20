import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


"""
SVM理解：
1. 线性数据分割:我们找到了一条直线，f (x) = ax1 + bx2 + c， 它可以将所有的数据分割到两个区域。
当我们拿到一个测试数据 X 时，我们只 需要把它代入 f (x)。如果 |f (X)| > 0，它就属于蓝色组，否则就属于红色组。 
我们把这条线称为决定边界（Decision_Boundary）。
很简单而且内存使用 效率也很高。这种使用一条直线（或者是高位空间种的超平面）上述数据分成 两组的方法成为线性分割。

那一条直线是最好的呢？直觉上讲这条直线应该是与两组数据的距离越远越好。为什么呢？
因为测试数据可能有噪音影响（真实数据 + 噪声）。这些数据不应该影响分类 的准确性。
所以这条距离远的直线抗噪声能力也就最强。
所以 SVM 要做就是 找到一条直线，并使这条直线到（训练样本）各组数据的最短距离最大。

要找到决定边界，就需要使用训练数据。我们需要所有的训练数据吗？
不， 只需要那些靠近边界的数据，如上图中一个蓝色的圆盘和两个红色的方块。
我 们叫他们支持向量，经过他们的直线叫做支持平面。有了这些数据就足以找到 决定边界了。
我们担心所有的数据。这对于数据简化有帮助。 

2. 非线性数据分割：
如果一组数据不能被一条直线分为两组怎么办？在一维空间中 X 类包含的数据点有（-3，3），O类包含的数据点有（-1，1）。
很明显不可能使用线性分割将 X 和 O 分开。但是有一个方法可以帮我们解决这个问题。
使用函数 f(x) = x2 对这组数据进行映射，得到的 X 为 9，O 为 1，这时就可以使用线性分割了。 
或者我们也可以把一维数据转换成两维数据。我们可以使用函数 f (x) = (x,x2)对数据进行映射。
这样 X 就变成了（-3，9）和（3，9）而 O 就变成了 （-1，1）和（1，1）。
同样可以线性分割，简单来说就是在低维空间不能线性 分割的数据在高维空间很有可能可以线性分割。 
通常我们可以将 d 维数据映射到 D 维数据来检测是否可以线性分割 （D>d）。
这种想法可以帮助我们通过对低维输入（特征）空间的计算来获得高维空间的点积。
这说明三维空间中的内积可以通过计算二维空间中内积的平方来获得。
这可以扩展到更高维的空间。所以根据低维的数据来计算它们的高维特征。
在进行完映射后，我们就得到了一个高维空间数据。 

除了上面的这些概念之外，还有一个问题需要解决，那就是分类错误。仅仅找到具有最大边缘的决定边界是不够的。
我们还需要考虑错误分类带来的误 差。有时我们找到的决定边界的边缘可能不是最大的但是错误分类是最少的。 
所以我们需要对我们的模型进行修正来找到一个更好的决定边界：最大的边缘， 最小的错误分类。

评判标准就被修改为：
min||w||^2 + C(distance of misclassified samples to their correct regions)

下图显示这个概念。对于训练数据的每一个样本又增加了一个参数 ξi。
它 表示训练样本到他们所属类（实际所属类）的超平面的距离。
对于那些分类正 确的样本这个参数为 0，因为它们会落在它们的支持平面上。
minL(w, b0) = ||w||^2 + C {ξi} 

参数 C 的取值应该如何选择呢？很明显应该取决于你的训练数据。
虽然没有一个统一的答案，但是在选取 C 的取值时我们还是应该考虑一下下面的规 则： 
    • 如果 C 的取值比较大，错误分类会减少，但是边缘也会减小。其实就是错误分类的代价比较高，惩罚比较大。
    （在数据噪声很小时我们可以选取 较大的 C 值。） 
    • 如果 C 的取值比较小，边缘会比较大，但错误分类的数量会升高。其实就是错误分类的代价比较低，惩罚很小。
    整个优化过程就是为了找到一个具有最大边缘的超平面对数据进行分类。（如果数据噪声比较大时，应该考虑）

"""
SZ = 20
affine_flags = cv.WARP_INVERSE_MAP | cv.INTER_LINEAR
bin_n = 16


def deskew(img):
    """
    在 kNN 中我们直接使用像素的灰度值作为特征向量。
    这次我们要使用方向梯度直方图Histogram of Oriented Gradients （HOG）作为特征向量。
    在计算 HOG 前我们使用图片的二阶矩对其进行抗扭斜（deskew）处理。
    所以我们首先要定义一个函数 deskew()，它可以对一个图像进行抗扭斜处理。

    """
    m = cv.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    img = cv.warpAffine(img,M,(SZ, SZ),flags=affine_flags)
    return img


def hog(img):
    """
    接下来我们要计算图像的 HOG 描述符，创建一个函数 hog()。为此我们计算图像 X 方向和 Y 方向的 Sobel导数。
    然后计算得到每个像素的梯度的方向和大小。把这个梯度转换成 16 位的整数。
    将图像分为 4 个小的方块，对每一个小方块计算它们的朝向直方图（16 个 bin），使用梯度的大小做权重。
    这样每一个小方块都会得到一个含有 16 个成员的向量。
    4 个小方块的 4 个向量 就组成了这个图像的特征向量（包含 64 个成员）。这就是我们要训练数据的特征向量
    最后，和前面一样，我们将大图分割成小图。使用每个数字的前 250 个作 为训练数据，后 250 个作为测试数据
    """
    gx = cv.Sobel(img, cv.CV_32F, 1, 0)
    gy = cv.Sobel(img, cv.CV_32F, 0, 1)
    mag, ang = cv.cartToPolar(gx, gy)  # 转换成幅值和角度
    bins = np.int32(bin_n * ang / (2 * np.pi))  # quantizing bin values in (0...16)
    bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
    mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists) # hist is a 64 bit vector return hist
    return hist


def svm_digits_ocr(img):
    cells = [np.hsplit(row, 100) for row in np.vsplit(img, 50)]

    train_cells = [ i[:50] for i in cells ]
    test_cells = [ i[50:] for i in cells ]

    deskewed = [ list(map(deskew, row)) for row in train_cells ]
    hog_data = [ list(map(hog, row)) for row in deskewed ]
    trainData = np.float32(hog_data).reshape(-1, 64)
    responses = np.int32(np.repeat(np.arange(10), 250)[:, np.newaxis])  # responses结果要用int32来存储，不然报错

    svm = cv.ml.SVM_create()
    svm.setType(cv.ml.SVM_C_SVC)
    svm.setKernel(cv.ml.SVM_LINEAR)
    svm.setC(2.67)
    svm.train(trainData, cv.ml.ROW_SAMPLE, responses)
    svm.save('svm_data.dat')

    deskewed = [ list(map(deskew, row)) for row in test_cells ]
    hog_data = [ list(map(hog, row)) for row in deskewed ]
    testData = np.float32(hog_data).reshape(-1, bin_n * 4)
    (p1, result) = svm.predict(testData)
    print(result)
    mask = result == responses
    correct = np.count_nonzero(mask)
    print(correct * 100.0 / result.size)


def test():
    # SVM 本质 寻求一个最优的超平面 分类
    # SVM 线性核 line
    # 核函数 线性核，多项式核 高斯径向基核 sigmoid核函数

    # 身高体重 训练 预测

    # 准备数据
    rand1 = np.array([[155, 48], [159, 50], [164, 53], [164, 56], [172, 60]])
    rand2 = np.array([[152, 53], [156, 55], [160, 56], [172, 64], [176, 65]])

    # lable
    label = np.array([[0], [0], [0], [0], [0], [1], [1], [1], [1], [1]])

    # data 处理
    data = np.vstack((rand1, rand2))
    print(data)
    data = np.array(data, dtype='float32')

    # 所有的数据必须一一对应的lable
    # 监督学习 0负样本1正样本

    # 训练
    svm = cv.ml.SVM_create()  # 创建SVM model
    # 属性设置
    svm.setType(cv.ml.SVM_C_SVC)
    svm.setKernel(cv.ml.SVM_LINEAR)
    svm.setC(0.01)
    # 训练
    result = svm.train(data, cv.ml.ROW_SAMPLE, label)
    # 预测

    pt_data = np.vstack([[167, 55], [162, 57]])
    pt_data = np.array(pt_data, dtype='float32')
    print(pt_data)
    (par1, par2) = svm.predict(pt_data)
    print(par2)


def main():
    src = cv.imread("../images/digits.png",0)
    # deskew(src)
    svm_digits_ocr(src)
    # test()
    cv.waitKey(0)  # 等有键输入或者1000ms后自动将窗口消除，0表示只用键输入结束窗口
    cv.destroyAllWindows()  # 关闭所有窗口


if __name__ == '__main__':
    main()