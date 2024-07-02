import torch
from pytorch_msssim import ssim
import torch.nn.functional as F
import numpy as np

def gaussian_kernel(size: int, mean: float, std: float):
    """가우시안 커널을 생성합니다."""
    d = torch.distributions.Normal(mean, std)
    vals = d.log_prob(torch.arange(start=-size, end=size + 1, dtype=torch.float32))
    gauss_kernel = torch.exp(vals).unsqueeze(1) @ torch.exp(vals).unsqueeze(0)
    return gauss_kernel / gauss_kernel.sum()

def apply_gaussian_weights(image: torch.Tensor, kernel: torch.Tensor):
    """이미지에 가우시안 가중치를 적용합니다."""
    channels = image.shape[1]
    kernel = kernel.expand(channels, 1, kernel.size(0), kernel.size(1))
    weighted_image = F.conv2d(image, kernel, padding='same', groups=channels)
    return weighted_image

def ms_ssim_loss(y_true: torch.Tensor, y_pred: torch.Tensor):
    """MS-SSIM 손실을 계산합니다."""
    return 1 - ssim(y_pred, y_true, data_range=1.0, size_average=True)

def mixed_loss(y_true: torch.Tensor, y_pred: torch.Tensor, alpha: float = 0.84):
    """MS-SSIM과 L1 손실의 혼합 손실을 계산합니다."""
    # MS-SSIM 계산
    ms_ssim = ms_ssim_loss(y_true, y_pred)
    
    # 가우시안 커널 생성
    gaussian_kernel_size = 5  # 커널 크기
    gaussian_mean = 0.0
    gaussian_std = 1.0
    kernel = gaussian_kernel(gaussian_kernel_size, gaussian_mean, gaussian_std).to(y_true.device)
    
    # 가우시안 가중치를 적용한 L1 손실 계산
    weighted_true = apply_gaussian_weights(y_true, kernel)
    weighted_pred = apply_gaussian_weights(y_pred, kernel)
    l1_loss = torch.mean(torch.abs(weighted_true - weighted_pred))
    
    # 혼합 손실 계산
    mixed = alpha * ms_ssim + (1 - alpha) * l1_loss
    return mixed
