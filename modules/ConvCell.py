import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from torch.nn.modules.utils import _pair

class ConvRNNCellBase(nn.Module):
    def __repr__(self):
        """
        ConvRNNCellBase 클래스의 문자열 표현을 반환합니다.

        Returns:
            str: ConvRNNCellBase 클래스의 문자열 표현
        """
        s = (
            '{name}({input_channels}, {hidden_channels}, kernel_size={kernel_size}'
            ', stride={stride}')
        if self.padding != (0, ) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1, ) * len(self.dilation):
            s += ', dilation={dilation}'
        s += ', hidden_kernel_size={hidden_kernel_size}'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class ConvLSTMCell(ConvRNNCellBase):
    def __init__(self,
                 input_channels,
                 hidden_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 dilation=1,
                 hidden_kernel_size=1,
                 bias=True):
        """
        컨볼루션 LSTM 셀.

        Args:
            input_channels (int): 입력 채널의 개수.
            hidden_channels (int): 은닉 상태의 채널 개수.
            kernel_size (int or tuple): 컨볼루션 커널의 크기. 기본값은 3.
            stride (int or tuple): 컨볼루션의 스트라이드. 기본값은 1.
            padding (int or tuple): 입력에 추가되는 패딩. 기본값은 0.
            dilation (int or tuple): 커널 요소 사이의 간격. 기본값은 1.
            hidden_kernel_size (int or tuple): 은닉 상태의 컨볼루션 커널의 크기. 기본값은 1.
            bias (bool): True인 경우 출력에 학습 가능한 편향을 추가합니다. 기본값은 True입니다.
        """
        super(ConvLSTMCell, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)

        self.hidden_kernel_size = _pair(hidden_kernel_size)

        hidden_padding = _pair(hidden_kernel_size // 2)

        gate_channels = 4 * self.hidden_channels
        self.conv_ih = nn.Conv2d(
            in_channels=self.input_channels,
            out_channels=gate_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=bias)

        self.conv_hh = nn.Conv2d(
            in_channels=self.hidden_channels,
            out_channels=gate_channels,
            kernel_size=hidden_kernel_size,
            stride=1,
            padding=hidden_padding,
            dilation=1,
            bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        """
        컨볼루션 레이어의 매개변수를 초기화합니다.
        """
        self.conv_ih.reset_parameters()
        self.conv_hh.reset_parameters()

    def forward(self, input, hidden):
        """
        ConvLSTMCell의 순방향 전파.

        Args:
            input (torch.Tensor): (batch_size, input_channels, height, width) 모양의 입력 텐서.
            hidden (tuple): (batch_size, hidden_channels, height, width) 모양의 은닉 상태와 셀 상태를 포함하는 튜플.

        Returns:
            tuple: (batch_size, hidden_channels, height, width) 모양의 업데이트된 은닉 상태와 셀 상태를 포함하는 튜플.
        """
        hx, cx = hidden
        gates = self.conv_ih(input) + self.conv_hh(hx)

        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * F.tanh(cy)

        return hy, cy

class ConvGRUCell(ConvRNNCellBase):
    def __init__(self,
                 input_channels,
                 hidden_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 dilation=1,
                 hidden_kernel_size=1,
                 bias=True):
        super(ConvGRUCell, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)

        self.hidden_kernel_size = _pair(hidden_kernel_size)

        hidden_padding = _pair(hidden_kernel_size // 2)

        #gate_channels = 4 * self.hidden_channels
        gate_channels = 3 * self.hidden_channels

        self.conv_ih = nn.Conv2d(
            in_channels=self.input_channels,
            out_channels=gate_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=bias)

        self.conv_hh = nn.Conv2d(
            in_channels=self.hidden_channels,
            out_channels=gate_channels,
            kernel_size=hidden_kernel_size,
            stride=1,
            padding=hidden_padding,
            dilation=1,
            bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        self.conv_ih.reset_parameters()
        self.conv_hh.reset_parameters()

    def forward(self, input, hidden):
        x = input
        gate_x = self.conv_ih(input) 
        gate_h = self.conv_hh(hidden)
        
        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)
        
        resetgate = F.sigmoid(i_r + h_r)
        inputgate = F.sigmoid(i_i + h_i)
        newgate = F.tanh(i_n + (resetgate * h_n))
        
        hy = newgate + inputgate * (hidden - newgate)
        
        return hy