import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


# k 近邻（kNN）的基本概念:
# KNN是通过测量不同特征值之间的距离进行分类。
# 它的思路是：如果一个样本在特征空间中的k个最相似(即特征空间中最邻近)的样本中的大多数属于某一个类别，
# 则该样本也属于这个类别，其中K通常是不大于20的整数。KNN算法中，所选择的邻居都是已经正确分类的对象。
# 该方法在定类决策上只依据最邻近的一个或者几个样本的类别来决定待分样本所属的类别。
#
# OpenCV 中的 kNN
# 这里我们将红色家族标记为 Class-0，蓝色家族标记为 Class-1。还要 再创建 25 个训练数据，
# 把它们非别标记为 Class-0 或者 Class-1。Numpy 中随机数产生器可以帮助我们完成这个任务。
# 然后借助 Matplotlib 将这些点绘制出来。红色家族显示为红色三角蓝色 家族显示为蓝色方块。


def knn_demo():
    # 在[0, 100)之间产生25x2个数字，即为25个点的坐标
    trainData = np.random.randint(0, 100, (25, 2)).astype(np.float32)

    # 生成这25个点的颜色，0表示红三角，1表示蓝矩形
    responses = np.random.randint(0, 2, (25, 1)).astype(np.float32)

    red = trainData[responses.ravel() == 0]  # ravel() 用于将response拉平
    plt.scatter(red[:,0], red[:,1], 80, 'r', '^')  # ^ 表示上三角， 80为点的大小

    blue = trainData[responses.ravel() == 1]
    plt.scatter(blue[:,0], blue[:,1], 80, 'b', 's')  # s 表示square正方形

    newcomer = np.random.randint(0, 100, (10, 2)).astype(np.float32)  # 单个点设置为(1, 2), 10个点设置为(10, 2)
    plt.scatter(newcomer[:,0],newcomer[:,1], 80, 'g', 'o')

    # 创建knn模型，进行训练，进行预测
    knn = cv.ml.KNearest_create()
    knn.train(trainData, cv.ml.ROW_SAMPLE, responses)
    # 对newcomer进行预测，得出ret（真假），results（分类结果），neighbor（临近K个点），dist（与k个点的距离）
    ret, results, neighbours, dist = knn.findNearest(newcomer,3)

    # 测试点信息
    i = 1
    for new, result, neighbour, distance in zip(newcomer,results, neighbours, dist):
        print(i, "point:(%s, %s) result:%s"%(new[0], new[1], result), "neighbor:", neighbour, "distance:",distance)
        i = i + 1

    plt.show()


def digit_ocr_demo(image):
    """
    OpenCV 安装包中有一副图片（/samples/ python2/data/digits.png）, 其中有 5000 个手写数字（每个数字重复 500 遍）。
    每个数字是一个20x20的小图。所以第一步就是将这个图像分割成5000 个不同的数字。
    我们在将拆分后的每一个数字的图像重排成一行含有 400 个像 素点的新图像。
    这个就是我们的特征集，所有像素的灰度值。这是我们能创建 的最简单的特征集。
    我们使用每个数字的前 250 个样本做训练数据，剩余的 250 个做测试数据。


    """
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    cells = [np.hsplit(row, 100) for row in np.vsplit(gray, 50)]  # 一张图片有5000个cell， 每个cell为一个20*20的图片

    x = np.array(cells)  # x 为50*100*20*20

    train = x[:, :50].reshape(-1, 400).astype(np.float32)
    test = x[:, 50:100].reshape(-1, 400).astype(np.float32)
    print(train.shape, test.shape)
    print(x[:, :50].shape)
    k = np.arange(10)
    train_labels = np.repeat(k, 250)[:, np.newaxis]
    test_labels = train_labels.copy()

    knn = cv.ml.KNearest_create()
    knn.train(train, cv.ml.ROW_SAMPLE, train_labels)
    ret, result, neighbours, dist = knn.findNearest(test, k=5)

    matches = (result == test_labels)
    correct = np.count_nonzero(matches)
    accuracy = correct*100.0/ result.size
    print("accuracy: ",accuracy)
    np.savez('../images/knn_data.npz',train=train, train_labels= train_labels)


def load_knn_data():
    """
    改善准确度的一个办法是提供更多的训练数据，尤其是判断错误的那 些数字。
    为了避免每次运行程序都要准备和训练分类器，我们最好把它保留，
    这样在下次运行是时，只需要从文件中读取这些数据开始进行分类就可以了。
     Numpy 函数 np.savetxt/np.savez，np.load 等可以帮助我们搞定这些。
    :return:
    """
    with np.load('../images/knn_data.npz') as data:
        print(data.files)

        train = data['train']
        train_labels = data['train_labels']
        print(train.shape, train_labels.shape)

def letter_ocr_demo():
    data = np.loadtxt('../images/letter-recognition.data', dtype = 'float32', delimiter= ',', converters={0: lambda ch: ord(ch) - ord('A')})

    train, test = np.vsplit(data, 2)
    responses, trainData = np.hsplit(train, [1])
    labels, testData = np.hsplit(test, [1])

    knn = cv.ml.KNearest_create()
    knn.train(trainData, cv.ml.ROW_SAMPLE, responses)
    ret, result, neighbours, dist = knn.findNearest(testData, k=5)

    correct = np.count_nonzero(result == labels)
    accuracy = correct*100.0/10000
    print("accuracy", accuracy)


def main():
    src = cv.imread("../images/digits.png")
    # knn_demo()
    # digit_ocr_demo(src)
    # load_knn_data()
    letter_ocr_demo()
    cv.waitKey(0)  # 等有键输入或者1000ms后自动将窗口消除，0表示只用键输入结束窗口
    cv.destroyAllWindows()  # 关闭所有窗口


if __name__ == '__main__':
    main()