# -*- coding: utf-8 -*-
# @Time    : 3/22/20 10:11
# @Author  : huangting
# @FileName: visualizationData.py
# @Software: PyCharm

import readDataUtils as rdu
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_csv("./data/WTB_icing/train/15_3.csv").values
checkpont = np.sum(data[:, 26] == 0)

for i in range(26):
    plt.figure()
    plt.title(i)
    x = np.linspace(0, len(data), len(data))
    plt.plot(x, data[:, i])
    plt.axvline(x=checkpont, ls="-", c="r")
    # plt.plot(x, d[:, i], c="b")

    # plt.savefig("./dataVisualization_05_" + str(i) + ".jpg")

    plt.show()

