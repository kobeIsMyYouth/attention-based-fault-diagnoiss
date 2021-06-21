# -*- coding: utf-8 -*-
# @Time    : 3/22/20 10:43
# @Author  : huangting
# @FileName: CNN_LSTM1.py
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.optim as optim

class MixA_Module(nn.Module):
    """ 注意力模块"""

    def __init__(self):
        super(MixA_Module, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.AVGpool = nn.AdaptiveAvgPool1d(1)
        self.MAXpool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x, ct):
        """
            inputs :
                x : 输入的特征map ( B X C x W X H)
                ct : 注意力区域 ( B X W X H) Q值
            returns :
                out : 注意力值
                spatial attention: W x H
        """
        m_batchsize, C, W, H = x.size()
        B_C_WH = x.view(m_batchsize, C, -1)
        B_WH_C = B_C_WH.permute(0, 2, 1)
        B_WH_C_AVG = torch.sigmoid(self.AVGpool(B_WH_C)).view(m_batchsize, W, H)
        B_WH_C_MAX = torch.sigmoid(self.MAXpool(B_WH_C)).view(m_batchsize, W, H)
        B_WH_C_Fusion = B_WH_C_AVG + B_WH_C_MAX + ct
        Attention_weight = self.softmax(B_WH_C_Fusion.view(m_batchsize, -1))
        Attention_weight = Attention_weight.view(m_batchsize, W, H)
        # mask1 mul
        output = x.clone()
        for i in range(C):
            output[:, i, :, :] = output[:, i, :, :].clone() * Attention_weight

        return output, Attention_weight

class CNN_LSTM(nn.Module):
    def __init__(self, seq):
        super(CNN_LSTM, self).__init__()


        self.Conv1 = nn.Sequential(
            nn.Conv2d(1, 10, [3, 3], stride=1, padding=1),
            nn.BatchNorm2d(10),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d((12, 52))
        )

        self.Conv2 = nn.Sequential(
            nn.Conv2d(10, 20, [3, 3], stride=1, padding=1),
            nn.BatchNorm2d(20),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d((12, 52))
        )

        self.Conv3 = nn.Sequential(
            nn.Conv2d(20, 35, [3, 3], stride=1, padding=1),
            nn.BatchNorm2d(35),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d((6, 26))
        )

        self.Conv4 = nn.Sequential(
            nn.Conv2d(35, 45, [3, 3], stride=1, padding=1),
            nn.BatchNorm2d(45),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d((3, 13))
        )

        # self.LSTM = nn.Sequential(
        #     nn.LSTM(
        #         input_size=52,
        #         hidden_size=52,
        #         num_layers=1,
        #         batch_first=True
        #     )
        # )

        self.Linear = nn.Sequential(
            nn.Linear(1755, 22),
            nn.Dropout(0.5),
            nn.Softmax(1)
        )

        self.MixA_Module = MixA_Module()


    def forward(self, x, ct):  # x[batch, 1, 52, 52]

        x1 = self.Conv1(x)  #  x1[batch, 3, 52, 52]
        #x1_atten, atten1 = self.MixA_Module(x1, ct)
        x2 = self.Conv2(x1)  #  x2[batch, 5, 52, 52]
        #x2_atten, atten2 = self.MixA_Module(x2, ct)

        x3 = self.Conv3(x2)  #  x3[batch, 10, 52, 13]

        x4 = self.Conv4(x3)  #  x4[batch, 1, 52, 13]

        # x5 = self.Conv5(x4_atten)  # x4[batch, 1, 52, 13]
        # x5_atten, atten5 = self.MixA_Module(x5, ct)

        batch, chan, seq, fea = x4.size()
        # x = x4.reshape(batch, seq, fea)  #  x[batch, 52, 13]

        # x, _ = self.LSTM(x)  #  x[batch, 52, 13]

        x = x4.reshape(batch, -1)
        x = self.Linear(x)  #  x[batch, 22]
        return x1, x2, x3, x4, x


if __name__ == '__main__':

    inputs = torch.randn(52, 1, 52, 52)
    labels = torch.randn(52, 22)
    ct = torch.randn(52, 52, 52)

    net = CNN_LSTM(52)
    checkpoint = torch.load("../checkpoints/res1_52/CNN-LSTM-82.t7")
    net.load_state_dict(checkpoint['net'])

    out = net(inputs, ct)

    print(out[9])


