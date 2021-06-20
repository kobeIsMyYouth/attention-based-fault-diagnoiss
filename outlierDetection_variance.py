# -*- coding: utf-8 -*-
# @Time    : 2/4/21 7:33 PM
# @Author  : huangting
# @FileName: outlierDetection_variance.py
# @Software: PyCharm
import numpy as np
import readDataUtils as rdu

def outlierDetection_variance(series):
    """
    @:param: series: 需要进行异常值判定的序列
    :return: outlier: 如果series某索引处的值被判定为异常值，那么在outlier索引处为1，否则为0.
    """
    baseCaseValues = np.loadtxt("./data/prior/means.txt")
    std = np.loadtxt("./data/prior/stds.txt")
    outlier = np.zeros(series.shape)
    for i in range(series.shape[0]):
        for j in range(series.shape[1]):
            if abs(baseCaseValues[j] - series[i,j]) >= 4*std[j]:
                outlier[i, j] = 1
    return outlier

# d = rdu.readExcel("./data/orig1/train/d09.xlsx")
# outlier = outlierDetection_variance(d)
# print("end!")

# d = rdu.readExcel("./data/orig1/train/d00.xlsx")
#
# d.mean(axis=0)
#
# np.savetxt("baseCaseValues1.txt",d.mean(axis=0))


