import torch
from pytorch_msssim import ms_ssim, MS_SSIM

# # X: (N,3,H,W) a batch of non-negative RGB images (0~255)
# # Y: (N,3,H,W)  

# # calculate ssim & ms-ssim for each image
# ms_ssim_val = ms_ssim( X, Y, data_range=255, size_average=False ) #(N,)

# # set 'size_average=True' to get a scalar value as loss. see tests/tests_loss.py for more details
# ms_ssim_loss = 1 - ms_ssim( X, Y, data_range=255, size_average=True )

# # reuse the gaussian kernel with SSIM & MS_SSIM. 
# ms_ssim_module = MS_SSIM(data_range=255, size_average=True, channel=3)

# ms_ssim_loss = 1 - ms_ssim_module(X, Y)

def cal_ms_ssim(img1, img2):
    img1 = img1.unsqueeze(0).permute(0, 3, 1, 2)
    img2 = img2.unsqueeze(0).permute(0, 3, 1, 2)
    ms_ssim_val = ms_ssim(img1, img2, data_range=1.0, size_average=True)
    return ms_ssim_val.item()