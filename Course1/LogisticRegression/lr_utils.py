import h5py
import numpy as np

'''
train_set_x_orig ：保存的是训练集里面的图像数据（本训练集有209张64x64的图像）。
train_set_y_orig ：保存的是训练集的图像对应的分类值（【0 | 1】，0表示不是猫，1表示是猫）。
test_set_x_orig ：保存的是测试集里面的图像数据（本训练集有50张64x64的图像）。
test_set_y_orig ： 保存的是测试集的图像对应的分类值（【0 | 1】，0表示不是猫，1表示是猫）。
classes ： 保存的是以bytes类型保存的两个字符串数据，数据为：[b’non-cat’ b’cat’]。
'''


def load_dataset():
    # 加载训练集的数据
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    print(train_dataset)  # <HDF5 file "train_catvnoncat.h5" (mode r)>
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])
    print(train_set_x_orig)  # 209张图片，每张图片有64*64个点，每个点有3个通道，每个通道的值是对应颜色的十进制的值。
    print(train_set_x_orig.shape)  # （209, 64, 64, 3)
    train_set_y_orig = np.array(
        train_dataset["train_set_y"][:])  # your train set labels
    print(train_set_y_orig)  # 有209个值的数组，每个值对应每张图片的分类值
    print(train_set_y_orig.shape)  # (209,)

    # 加载测试集的数据
    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])
    test_set_y_orig = np.array(
        test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes
    print(classes)  # [b'non-cat' b'cat']

    # 转成1*209的矩阵
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    print(train_set_y_orig)
    print(train_set_y_orig.shape)  # (1, 209)
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


# load_dataset()
