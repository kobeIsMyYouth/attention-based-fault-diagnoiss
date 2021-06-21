# -*- coding: utf-8 -*-
# @Time    : 3/22/20 19:15
# @Author  : huangting
# @FileName: train.py
# @Software: PyCharm

import argparse
import os
import dataLoader
from model import CNN_LSTM_11
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from keras.utils.np_utils import to_categorical

parser = argparse.ArgumentParser(description='面部表情识别')
parser.add_argument("--model", default="CNN-LSTM", type=str, help="需要训练的模型")
parser.add_argument("--bs", default=50, type=int, help="batchsize")
parser.add_argument("--lr", default=0.0001, type=float, help="学习率")
parser.add_argument("--ts", default=10, type=int, help="网络处理数据的时间步长")
parser.add_argument("--dl", default=0, type=int, help="延时预测的长度")
parser.add_argument("--lb", default=1, type=int, help="滑窗处理的步长")
parser.add_argument("--dr", default="./data", type=str, help="数据所在的根路径")
parser.add_argument("--ep", default=100, type=int, help="训练的总轮数")
parser.add_argument('--re', default=True, help='是否从checkpoint处开始训练')
parser.add_argument("--ckpt", default="./checkpoints/res_10/CNN-LSTM-15.t7", help="checkpoint的地址")
parser.add_argument("--gpu", default=False, help="是否使用GPU进行训T练")
parser.add_argument("--cuda", default="0，1，2，3", type=str, help="用于训练GPU的代号")

cfg, unknown = parser.parse_known_args()
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.cuda # 配置环境中的GPUF

start_epoch = 0 # 开始训练的轮数
best_test_acc = 0 # 最好的测试准确率
best_test_acc_epoch = 0 # 最好的测试准确率时的轮数


print("--------------加载测试数据中------------")
testRoot = cfg.dr + "/orig1/test"
X_test, y_test, mask_test = dataLoader.window_process(testRoot, "./data/test1", cfg.dl, cfg.ts, cfg.lb)
X_test = np.array(X_test)
y_test = np.array(y_test)
mask_test = np.array(mask_test)
y_test = to_categorical(y_test, 22)

# 对数据进行随机打乱
permutation = np.random.permutation(X_test.shape[0])
X_test = X_test[permutation, :, :, :]
y_test = y_test[permutation, :]
mask_test = mask_test[permutation, :, :]

testData_sum = len(X_test) # 训练样本总数
print("--------------加载测试数据完成------------")

use_cuda = torch.cuda.is_available()  # 环境中是否有GPU
net = CNN_LSTM_11.CNN_LSTM(cfg.ts)

if use_cuda and cfg.gpu:
    net = net.cuda()
    net = nn.DataParallel(net)

if cfg.re:
    print('------------------------------')
    print('==> 加载checkpoint ')
    if not os.path.exists(cfg.ckpt):
        raise AssertionError['找不到路径']
    checkpoint = torch.load(cfg.ckpt)
    net.load_state_dict(checkpoint['net'])
    best_test_acc = checkpoint['best_test_acc']
    print('best_test_acc is %.4f%%'%best_test_acc)
    best_test_acc_epoch = checkpoint['best_test_acc_epoch']
    print('best_test_acc_epoch is %d'%best_test_acc_epoch)
    start_epoch = checkpoint['best_test_acc_epoch'] + 1
else:
    print('------------------------------')
    print('==> 构建新的模型')

optimizer = optim.Adam(net.parameters(), lr=cfg.lr)  # 实例化梯度下降算法
MSELoss = nn.MSELoss()  # 实例化loss

def test(epoch):
    global test_acc
    global best_test_acc
    global best_test_acc_epoch
    net.eval()
    total = 0
    correct = 0
    confuse = torch.zeros(22, 22)  # 定义混淆矩阵

    for i in range(0, testData_sum, cfg.bs):
        if testData_sum - i >= cfg.bs:
            inputs = X_test[i:(i+cfg.bs), :, :, :]
            target = y_test[i:(i+cfg.bs), :]
            mask = mask_test[i:(i + cfg.bs), :, :]
        else:
            inputs = X_test[i:testData_sum, :, :, :]
            target = y_test[i:testData_sum, :]
            mask = mask_test[i:testData_sum, :, :]


        inputs = torch.Tensor(inputs)
        target = torch.Tensor(target)
        mask = torch.Tensor(mask)

        if use_cuda and cfg.gpu:
            inputs = inputs.cuda()
            target = target.cuda()

        with torch.no_grad():
            outputs = net(inputs, mask)

        _, predicted = torch.max(outputs[8].data, 1)
        _, trueValue = torch.max(target.data, 1)

        for j in range(predicted.size()[0]):
            confuse[predicted[j], trueValue[j]] += 1

        total += target.size(0)
        correct += predicted.eq(trueValue.data).sum()

    for categorical in range(22):
        confuse[categorical] = confuse[categorical] / confuse[categorical].sum()
    print(confuse.data)
    test_acc = 100.0 * int(correct.data) / total
    print('在 %d 个样本中, %d 个被准确预测' % (total, correct))
    print('测试准确率为 %.4f%%' % test_acc)
    print("一轮测试已经完成")

    return

if __name__ == "__main__":

    test(0)




