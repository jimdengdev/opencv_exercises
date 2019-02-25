# 30个python实现
import numpy as np
import os


# 1. 冒泡排序
def bubble_sort(li):
    result = li.copy()
    for i in range(len(result) - 1):  # n-1趟
        for j in range(len(result) - 1 - i):  # n-1-i趟
            if result[j] > result[j+1]:
                result[j], result[j+1] = result[j+1], result[j]

    return result

# 2、计算x的n次方的方法
def power(x, n):
    result = 1
    while n > 0:
        result = result * x
        n = n - 1
    return result

# 3. 计算平方和
def square_calculate(arr):
    sum = 0
    for i in arr:
        sum = sum + i * i
    return sum

# 4. 计算阶乘
def fac(n):
    if n < 0:
        print("错误，负值没有阶乘")
    elif n == 0:
        print("0 的阶乘为 1")
    else:
        if n == 1:
            return 1
        else:
            return n * fac(n - 1)

# 5. 列出当前目录下的所有文件和目录名
def list_file_dir():
    return [i for i in os.listdir('.')]

# 6. 把一个list中所有的字符串变成小写
def str2upper(str):
    return [s.upper() for s in str]

# 7. 输出某个路径下的所有文件和文件夹的路径
def print_dir(d):
    if d == "":
        print("路径错误，请输入正确路径")
    else:
        for i in os.listdir(d):
            print(os.path.join(d,i))  # 路径组合

# 8. 键值对颠倒
def dict_reverse(dic):
    dic2 = {y:x for x, y in dic.items()}
    return dic2

# 9. 九九乘法表
def print_nine_nine():
    for i in range(1, 10):
        for j in range(1, i+1):
            print("%d x %d = %d"%(j, i, j*i), end=" ")
        print()

# 10. 合并去重
def merge_remove_duplicate(lis1, lis2):
    lis3 = set(lis1 + lis2)
    return lis3

def main():
    # 测试冒泡
    li = [22, 33, 23, 56, 984, 65, 99, 12, 9, 1]
    print("1. 冒泡排序：", bubble_sort(li))

    # 测试n次幂
    print("2. 2的10次幂为：", power(2, 10))

    # 测试平方和
    arr =np.array([1, 2, 3, 4, 5])
    print("3. 1到5的平方和为：", square_calculate(arr))

    # 测试阶乘
    print("4. 10的阶乘为：", fac(10))

    # 测试列出当前目录下的所有文件和目录名
    print("5. 列出当前目录下的所有文件和目录名:",list_file_dir())

    # 测试字符串变成大写
    str = ["You", "are", "beautiful"]
    print("6. str所有单词变大写：", str2upper(str))

    # 测试某个路径下的所有文件和文件夹的路径
    print("7. 测试当前路径下文件和文件夹路径：")
    print_dir(".")

    # 测试字典颠倒
    dic = {'A':'a', 'B':'b', 'C':'c'}
    print("8. dic字典颠倒后：",dict_reverse(dic))

    # 打印九九乘法表
    print("9. 打印九九乘法表：")
    print_nine_nine()

    # 合并去重
    li1 = [1, 2, 3, 5, 1]
    li2 = [2, 3, 2, 1, 6]
    print("10. li1并li2去重后：", merge_remove_duplicate(li1, li2))


if __name__ == '__main__':
    main()