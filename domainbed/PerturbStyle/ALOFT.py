import torch
import math
from math import sqrt
import numpy as np


def fourier_stylized_images(img, mask_size=0.5, mix_or_model=1, alpha=1.0, factor=0.8):
    B, C, h, w = img.shape
    img = img.to(torch.float32)
    img_fft = torch.fft.fft2(img, dim=(2, 3), norm='ortho')
    img_abs, img_pha = torch.abs(img_fft), torch.angle(img_fft)

    _, _, h_fft, w_fft = img_fft.shape
    h_crop = int(h_fft * sqrt(mask_size))
    w_crop = int(w_fft * sqrt(mask_size))
    h_start = h_fft // 2 - h_crop // 2
    w_start = w_fft // 2 - w_crop // 2

    img_abs = torch.fft.fftshift(img_abs, dim=(2, 3))
    if mix_or_model == 0:
        perm = torch.randperm(B)
        lam = (torch.rand(B, 1, 1, 1) * alpha).cuda()
        img1_abs_ = img_abs.clone()
        img2_abs_ = img1_abs_[perm]
        img_abs[:, :, h_start:h_start + h_crop, w_start:w_start + w_crop] \
            = lam * img2_abs_[:, :, h_start:h_start + h_crop, w_start:w_start + w_crop] + \
              (1 - lam) * img1_abs_[:, :, h_start:h_start + h_crop, w_start:w_start + w_crop]
    else:
        img_abs_ = img_abs.clone()

        var_of_elem = torch.var(img_abs_[:, :, h_start:h_start + h_crop, w_start:w_start + w_crop], dim=0, keepdim=True)
        sig_of_elem = (var_of_elem + 1e-6).sqrt()  # 1xHxWxC

        epsilon_sig = torch.randn_like(
            img_abs[:, :, h_start:h_start + h_crop, w_start:w_start + w_crop])  # BxHxWxC N(0,1)
        gamma = epsilon_sig * sig_of_elem * factor

        img_abs[:, :, h_start:h_start + h_crop, w_start:w_start + w_crop] = \
            img_abs[:, :, h_start:h_start + h_crop, w_start:w_start + w_crop] + gamma

    img_abs = torch.fft.ifftshift(img_abs, dim=(2, 3))
    img_stylized = img_abs * (np.e ** (1j * img_pha))

    img_stylized = torch.fft.ifft2(img_stylized, s=(h, w), dim=(2, 3), norm='ortho')
    img_stylized = torch.real(img_stylized)
    return img_stylized


