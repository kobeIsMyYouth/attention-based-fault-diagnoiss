# -*- coding: utf-8 -*-
# @Time    : 3/22/20 10:05
# @Author  : huangting
# @FileName: dataLoader.py
# @Software: PyCharm

import readDataUtils as rdu
import os
import numpy as np
import outlierDetection_variance as odv
import pandas as pd


def window_process(rootPath1, rootPath2, delay, checkpont, timesteps=52, lamb=1):
    """
    对原始数据进行滑窗处理
    :param rootPath1: 原始数据的根路径
    :param rootPath2: 标准化数据的根路径
    :param delay: 设定标签的时延
    :param checkpont：序列中出现故障的点
    :param timesteps: 滑窗的跨时间长度
    :param lamb: 滑窗运行的步长
    :return: 经过滑窗处理之后的数据
    """
    data = []  # 存储每一个类别经过滑窗处理后的数据
    labels = []
    files = sorted(os.listdir(rootPath2))

    mask = []

    for file in files:
        if not file.startswith("."):
            filePath1 = rootPath1 + "/" + file
            filePath2 = rootPath2 + "/" + file

            rawData = rdu.readExcel(filePath1)
            stdData = rdu.readExcel(filePath2)
            label = int(file[1:3])

            seqLen = len(rawData)
            for i in range(0, seqLen, lamb):
                if (seqLen - i) >= timesteps and (seqLen - i - timesteps) >= delay:
                    x = rawData[i:(i + timesteps), :]
                    mask.append(odv.outlierDetection_variance(x))
                    x = stdData[i:(i + timesteps), :]
                    shape = x.shape
                    x = x.reshape(1, shape[0], shape[1])
                    data.append(x.tolist())

                    label_index = i + timesteps - 1  # 标签所在的时序位置
                    if label_index >= checkpont:
                        labels.append(label)
                    else:
                        labels.append(0)
    return data, labels, mask


def window_process2(rootPath1, rootPath2, delay, timesteps=52, lamb=1):
    """
    对原始数据进行滑窗处理
    :param rootPath1: 原始数据的根路径
    :param rootPath2: 标准化数据的根路径
    :param delay: 设定标签的时延
    :param timesteps: 滑窗的跨时间长度
    :param lamb: 滑窗运行的步长
    :return: 经过滑窗处理之后的数据
    """
    data = []  # 存储每一个类别经过滑窗处理后的数据
    labels = []
    files = sorted(os.listdir(rootPath2))

    mask = []

    for file in files:
        if not file.startswith("."):
            filePath1 = rootPath1 + "/" + file
            filePath2 = rootPath2 + "/" + file

            rawData = pd.read_csv(filePath1).values
            stdData = pd.read_csv(filePath2).values

            checkpont = np.sum(rawData[:, 26] == 0)

            rawData = rawData[:, 0:26]
            stdData = stdData[:, 0:26]
            label = 1
            seqLen = len(rawData)

            for i in range(0, seqLen, lamb):
                if (seqLen - i) >= timesteps and (seqLen - i - timesteps) >= delay:
                    x = rawData[i:(i + timesteps), :]
                    mask.append(odv.outlierDetection_variance(x))
                    x = stdData[i:(i + timesteps), :]
                    shape = x.shape
                    x = x.reshape(1, shape[0], shape[1])
                    data.append(x.tolist())

                    label_index = i + timesteps - 1  # 标签所在的时序位置
                    if label_index >= checkpont:
                        labels.append(label)
                    else:
                        labels.append(0)

    return data, labels, mask


# def window_process2(rootPath, delay, timesteps=12, lamb=1):
#     """
#     对原始数据进行滑窗处理
#     :param rootPath: 原始数据的根路径
#     :param delay: 设定标签的时延
#     :param timesteps: 滑窗的跨时间长度
#     :param lamb: 滑窗运行的步长
#     :return: 经过滑窗处理之后的数据
#     """
#     data = []  # 存储每一个类别经过滑窗处理后的数据
#     labels = []
#     files = sorted(os.listdir(rootPath))
#
#     mask = []
#
#     for file in files:
#         if not file.startswith("."):
#             filePath = rootPath + "/" + file
#             rawData = pd.read_csv(filePath).values
#             checkpont = np.sum(rawData[:, 26] == 0)
#             label = 1
#             rawData = rawData[:, 0:26]
#
#             seqLen = len(rawData)
#             for i in range(0, seqLen, lamb):
#                 if (seqLen - i) >= timesteps and (seqLen - i - timesteps) >= delay:
#                     x = rawData[i:(i+timesteps), :]
#                     mask.append(odv.outlierDetection_variance(x))
#                     shape = x.shape
#                     x = x.reshape(1, shape[0], shape[1])
#                     data.append(x.tolist())
#
#                     label_index = i + timesteps - 1  # 标签所在的时序位置
#                     if label_index >= checkpont:
#                         labels.append(label)
#                     else:
#                         labels.append(0)
#
#     return data, labels, mask

if __name__ == "__main__":
    # rootPath1 = "./data/WTB_icing/train"
    # rootPath2 = "./data/train"
    # data, labels, mask = window_process2(rootPath1, 0, 12, 1)
    # print(np.array(data).shape)
    # print(np.array(labels).shape)
    # print(np.array(mask).shape)
    # print("end")
    data = [[[[3.165851845, 1.447232836, 2.927715498, 0.949004755, -0.329332788, -0.704559741, 0.589485535, 2.864113037,
               2.847812896, 2.849867821, -4.637592047, -5.042342428, -4.288782513, 1.7732694619999998, 1.634238504,
               1.586636711, 0.030739705, 0.305876054, 0.203647155, 0.047123798, 0.602451422, 0.463029525, 0.270139251,
               0.287592147, 2.53811553, -0.19942484100000002],
              [2.95026256, 1.4607547969999999, 2.9233124989999997, 0.831161935, 1.18647243, -0.704559741, 0.589485535,
               2.70127944, 2.665922911, 2.668518336, 6.003896375, 4.934247719, 5.30166782, 1.801702434,
               1.6769971999999997, 1.615305754, 1.3800086459999998, -1.5220762190000001, 0.19036345899999998,
               0.047123798, 0.602451422, 0.463029525, 0.207350778, 0.326322105, 0.7504112490000001, -0.26905015],
              [2.762620034, 1.4708962680000002, 2.9233124989999997, 0.724619111, 0.26723137199999997, -0.704559741,
               0.392549793, 2.821896918, 2.768993902, 2.759193078, -0.107057372, -0.7366561540000001, -0.678260035,
               1.828781455, 1.705502998, 1.629640275, 0.030739705, -1.5220762190000001, 0.203647155, 0.047123798,
               0.602451422, 0.404978778, 0.207350778, -0.215897311, 2.294337674, 0.218327012],
              [2.8384755239999997, 1.403286461, 2.932118496, 0.7972619459999999, 0.461281053, -0.704559741, 0.392549793,
               2.074068548, 2.03537096, 2.070065037, -2.846450431, -1.891840276, -2.7091789289999997, 1.855860476,
               1.719755896, 1.6583093169999998, 0.030739705, -3.350028492, 0.19036345899999998, 0.047123798,
               0.602451422, 0.404978778, 0.207350778, -0.060977477999999995, -0.184070535, 0.810142138],
              [2.8185135530000003, 1.491179211, 2.9167208760000003, 0.8295476490000001, 0.13971301, -0.704559741,
               0.19561405, 2.924421775, 2.92056889, 2.837777855, 0.630471528, -0.211572462, 0.7885147220000001,
               1.881585546, 1.776767491, 1.686978359, 1.3800086459999998, 0.305876054, 0.203647155, 0.047123798,
               0.5317840229999999, 0.404978778, 0.13758581, 1.217111144, -0.062181607, 0.253139667],
              [2.886384253, 1.440471855, 2.927715498, 0.79564766, 1.276289711, -0.730160615, -2.364550575,
               2.3454578759999998, 2.326394938, 2.342089264, -0.42314118700000003, 0.628561445, 0.11154175699999999,
               1.896479008, 1.79102039, 1.70131288, 0.030739705, -1.5220762190000001, 0.203647155, 0.047123798,
               0.467540935, 0.34047795, 0.13758581, -0.835576642, -1.402959818, -0.060174223],
              [2.926308195, 1.4607547969999999, 2.9167208760000003, 0.816633368, 0.351504376, -0.704559741, 0.786421272,
               2.70127944, 2.708363907, 2.6987432510000002, -0.739225001, -0.106555724, -0.00128707, 1.908664567,
               1.8181008969999999, 1.758650965, 0.030739705, 0.305876054, 0.203647155, 0.047123798, 0.467540935,
               0.34047795, 0.13758581, 0.636161771, 0.506633392, -0.26905015],
              [2.7266884869999997, 1.447232836, 2.93871012, 0.847304787, 0.24505426600000002, -0.704559741, 0.983357015,
               2.4600444809999997, 2.4415919280000002, 2.48112387, -2.530366617, -3.4670913519999997,
               -2.7091789289999997, 1.937097539, 1.832353796, 1.758650965, -1.318529236, 0.305876054, 0.203647155,
               0.047123798, 0.467540935, 0.275977118, 0.067820838, -0.254627269, -1.321700533, 0.04426374],
              [2.846460312, 1.48441823, 2.925526873, 0.840847645, 0.564404597, -0.704559741, 0.786421272,
               2.6228780780000003, 2.629544913, 2.559708645, -1.266031359, -1.261739846, -0.11411589800000001,
               1.96417656, 1.862284883, 1.787320007, 0.030739705, 0.305876054, 0.218406817, 0.047123798, 0.467540935,
               0.275977118, 0.067820838, -0.8743065999999999, 0.46600374899999997, 0.009451086],
              [2.7426580639999996, 1.447232836, 2.925526873, 0.928019047, 1.6311234119999998, -0.698159522, 1.574164237,
               2.713341186, 2.708363907, 2.722923181, -1.266031359, -0.526622677, -1.242404172, 1.9912555809999999,
               1.889365391, 1.800221076, 1.3800086459999998, 0.305876054, 0.218406817, 0.047123798, 0.409722153,
               0.275977118, 0.067820838, 0.287592147, 1.847411603, 0.462015594],
              [2.942277771, 1.48441823, 2.93871012, 0.960304752, 0.5122883970000001, -0.704559741, 0.392549793,
               2.972668768, 2.956946887, 2.922407614, 0.103665171, 0.103477753, 0.11154175699999999, 2.016980651,
               1.9192964780000001, 1.83032357, 1.3800086459999998, 0.305876054, 0.218406817, 0.047123798, 0.409722153,
               0.275977118, 0.067820838, 0.016482438999999998, -0.143440892, 1.7500838090000002],
              [2.7865743999999997, 1.453993817, 2.9233124989999997, 0.98129046, 0.818332464, -0.704559741, 0.589485535,
               2.8882365310000004, 2.884190893, 2.8740477510000004, -0.844586273, -1.156723108, -0.7910888620000001,
               2.016980651, 1.960629884, 1.8446580909999999, 0.030739705, 0.305876054, 0.203647155, 0.047123798,
               0.409722153, 0.205026205, -0.008920629000000001, 1.3720309769999999, 2.09118946, 0.740516829]]]]
    data = np.array(data)
    print(data.shape)