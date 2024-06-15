import torch
import torch.nn as nn
import torch.nn.functional as F

from .ConvCell import ConvGRUCell as RNNCell
from .sign import Sign

class EncoderCell(nn.Module):
    def __init__(self):
        super(EncoderCell, self).__init__()

        self.conv = nn.Conv2d(
            3, 64, kernel_size=3, stride=2, padding=1, bias=False)

        self.rnn1 = RNNCell(
            64,
            256,
            kernel_size=3,
            stride=2,
            padding=1,
            hidden_kernel_size=1,
            bias=False)
        self.rnn2 = RNNCell(
            256,
            512,
            kernel_size=3,
            stride=2,
            padding=1,
            hidden_kernel_size=1,
            bias=False)
        self.rnn3 = RNNCell(
            512,
            512,
            kernel_size=3,
            stride=2,
            padding=1,
            hidden_kernel_size=1,
            bias=False)

    def forward(self, input, hidden1, hidden2, hidden3):
        
        x = self.conv(input)

        hidden1 = self.rnn1(x, hidden1)
        ## 문제1
        # LSTM에서는 hidden1[0]를 넣어줌. hidden은 hx, cx로 구성되므로 다음 input으로 hx, 즉 hidden state를 넣어주고 있음.
        # 하지만 GRU에서는 hidden이 hx로만 구성되어 있으므로 hidden1만 넣어주면 됨.
        x = hidden1 

        hidden2 = self.rnn2(x, hidden2)
        x = hidden2

        hidden3 = self.rnn3(x, hidden3)
        x = hidden3

        return x, hidden1, hidden2, hidden3


class Binarizer(nn.Module):
    def __init__(self):
        super(Binarizer, self).__init__()
        self.conv = nn.Conv2d(512, 32, kernel_size=1, bias=False)
        self.sign = Sign()

    def forward(self, input):
        feat = self.conv(input)
        x = F.tanh(feat)
        return self.sign(x)


class DecoderCell(nn.Module):
    def __init__(self):
        super(DecoderCell, self).__init__()

        self.conv1 = nn.Conv2d(
            32, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.rnn1 = RNNCell(
            512,
            512,
            kernel_size=3,
            stride=1,
            padding=1,
            hidden_kernel_size=1,
            bias=False)
        self.rnn2 = RNNCell(
            128,
            512,
            kernel_size=3,
            stride=1,
            padding=1,
            hidden_kernel_size=1,
            bias=False)
        self.rnn3 = RNNCell(
            128,
            256,
            kernel_size=3,
            stride=1,
            padding=1,
            hidden_kernel_size=3,
            bias=False)
        self.rnn4 = RNNCell(
            64,
            128,
            kernel_size=3,
            stride=1,
            padding=1,
            hidden_kernel_size=3,
            bias=False)
        self.conv2 = nn.Conv2d(
            32, 3, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, input, hidden1, hidden2, hidden3, hidden4):
        x = self.conv1(input)

        hidden1 = self.rnn1(x, hidden1)
        x = hidden1
        x = F.pixel_shuffle(x, 2)

        hidden2 = self.rnn2(x, hidden2)
        x = hidden2
        x = F.pixel_shuffle(x, 2)

        hidden3 = self.rnn3(x, hidden3)
        x = hidden3
        x = F.pixel_shuffle(x, 2)

        hidden4 = self.rnn4(x, hidden4)
        x = hidden4
        x = F.pixel_shuffle(x, 2)

        x = F.tanh(self.conv2(x)) / 2
        return x, hidden1, hidden2, hidden3, hidden4
