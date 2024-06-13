import torch
import torch.nn as nn
from torch.autograd import Function

# 즉, 입력 데이터에 대해서 -1 또는 1로 양자화하는 함수입니다.
class SignFunction(Function):
    """
    입력 데이터에 대해서 -1 또는 1로 양자화
    Variable Rate Image Compression with Recurrent Neural Networks
    https://arxiv.org/abs/1511.06085
    """
    def __init__(self):
        super(Sign, self).__init__()

    @staticmethod
    def forward(ctx, input, is_training=True):
        # 학습 중에는 양자화 잡음을 추가하여 입력 데이터를 처리하고, 평가 중에는 단순히 입력 데이터의 부호를 반환
        # Apply quantization noise while only training
        if is_training:
            prob = input.new(input.size()).uniform_() # input과 같은 크기의 tensor를 생성하고 0과 1 사이의 균일한 분포에서 랜덤한 값을 채웁니다.
            x = input.clone() # input을 복사합니다.
            x[(1 - input) / 2 <= prob] = 1 # (1 - input) / 2 값이 prob 이하인 요소를 1로 설정합니다.
            x[(1 - input) / 2 > prob] = -1 # (1 - input) / 2 값이 prob 초과인 요소를 -1로 설정합니다.
            return x
        else:
            return input.sign() # input의 부호를 반환합니다.

    @staticmethod
    def backward(ctx, grad_output):
        # 역전파에서 입력에 대한 기울기는 단순히 출력 기울기 grad_output을 반환하며, 두 번째 반환 값은 None입니다. 이는 is_training 플래그에 대한 기울기는 필요 없음을 의미합니다.
        return grad_output, None


class Sign(nn.Module):
    """
    입력 텐서에 대해 원소별로 부호 함수를 적용하는 모듈입니다.

    Args:
        None

    Returns:
        torch.Tensor: 입력 텐서와 동일한 모양을 가진 출력 텐서입니다.

    Examples:
        >>> sign = Sign()
        >>> input_tensor = torch.tensor([-2.5, 0, 3.7])
        >>> output_tensor = sign(input_tensor)
        >>> print(output_tensor)
        tensor([-1.,  0.,  1.])
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return SignFunction.apply(x, self.training)
