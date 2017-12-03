import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# 保存输入数据
def px(prefix, img1, img2, img3, img4):
    with open('./data/' + prefix + '_images.py', 'a+') as f:
        print(img1, file=f, end=",\n")
        print(img2, file=f, end=",\n")
        print(img3, file=f, end=",\n")
        print(img4, file=f, end=",\n")

# 保存标签数据
def py(prefix, code):
    with open('./data/' + prefix + '_labels.py', 'a+') as f:
        for x in range(4):
            tmp = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            tmp[int(code[x])] = 1
            print(tmp, file=f, end=",\n")

# 预处理图片
def paa(file):
    img = Image.open(file).convert('L') # 读取图片并灰度化

    img = img.crop((2, 1, 66, 22)) # 裁掉边变成 64x21

    # 分离数字
    img1 = img.crop((0, 0, 16, 21))
    img2 = img.crop((16, 0, 32, 21))
    img3 = img.crop((32, 0, 48, 21))
    img4 = img.crop((48, 0, 64, 21))

    img1 = np.array(img1).flatten() # 扁平化，把二维弄成一维度
    img1 = list(map(lambda x: 1 if x <= 180 else 0, img1)) # 二值化
    img2 = np.array(img2).flatten()
    img2 = list(map(lambda x: 1 if x <= 180 else 0, img2))
    img3 = np.array(img3).flatten()
    img3 = list(map(lambda x: 1 if x <= 180 else 0, img3))
    img4 = np.array(img4).flatten()
    img4 = list(map(lambda x: 1 if x <= 180 else 0, img4))

    return (img1, img2, img3, img4)

def work(prefix):
    results = {}
    files = []
    with open('./' + prefix + '.txt') as f:
        for x in f.readlines():
            tmp = x.split(':')
            files.append(tmp[0])
            results[tmp[0]] = tmp[1]

    with open('./data/' + prefix + '_images.py', 'w') as f:
        print("data = [", file=f)
    with open('./data/' + prefix + '_labels.py', 'w') as f:
        print("data = [", file=f)

    for file in files:
        print(file)
        img1, img2, img3, img4 = paa(file);
        px(prefix, img1, img2, img3, img4)
        py(prefix, results[file])

    with open('./data/' + prefix + '_images.py', 'a+') as f:
        print("]", file=f)
    with open('./data/' + prefix + '_labels.py', 'a+') as f:
        print("]", file=f)

work('test')
work('train')
