import math
import logging
from functools import partial
from collections import OrderedDict
from copy import Error, deepcopy
from re import S
from numpy.lib.arraypad import pad
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.fft
from torch.nn.modules.container import Sequential
from math import sqrt
from functools import partial, reduce
from operator import mul

import einops
from munch import Munch
import random

from .PerturbStyle.DSU import DSU

_logger = logging.getLogger(__name__)


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GlobalFilter(nn.Module):
    def __init__(self, dim, h=14, w=8):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(h, w, dim, 2, dtype=torch.float32) * 0.02)
        self.w = w
        self.h = h

    def forward(self, x, spatial_size=None):
        B, N, C = x.shape
        if spatial_size is None:
            a = b = int(math.sqrt(N))
        else:
            a, b = spatial_size

        x = x.view(B, a, b, C)
        x = x.to(torch.float32)
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft2(x, s=(a, b), dim=(1, 2), norm='ortho')
        x = x.reshape(B, N, C)
        return x


class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, h=14, w=8):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.filter = GlobalFilter(dim, h=h, w=w)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.mlp(self.norm2(self.filter(self.norm1(x)))))
        return x


class BlockLayerScale(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, h=14, w=8, init_values=1e-5):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.filter = GlobalFilter(dim, h=h, w=w)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gamma = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.gamma * self.mlp(self.norm2(self.filter(self.norm1(x)))))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def init_weights(self, stop_grad_conv1=0):
        val = math.sqrt(6. / float(3 * reduce(mul, self.patch_size, 1) + self.embed_dim))
        nn.init.uniform_(self.proj.weight, -val, val)
        nn.init.zeros_(self.proj.bias)

        if stop_grad_conv1:
            self.proj.weight.requires_grad = False
            self.proj.bias.requires_grad = False

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class DownLayer(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=56, dim_in=64, dim_out=128):
        super().__init__()
        self.img_size = img_size
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.proj = nn.Conv2d(dim_in, dim_out, kernel_size=2, stride=2)
        self.num_patches = img_size * img_size // 4

    def forward(self, x):
        B, N, C = x.size()
        x = x.view(B, self.img_size, self.img_size, C).permute(0, 3, 1, 2)
        x = self.proj(x).permute(0, 2, 3, 1)
        x = x.reshape(B, -1, self.dim_out)
        return x


class GFNet(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 mlp_ratio=4., representation_size=None, uniform_drop=False,
                 drop_rate=0., drop_path_rate=0., norm_layer=None,
                 dropcls=0):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            hybrid_backbone (nn.Module): CNN backbone to use in-place of PatchEmbed module
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        h = img_size // patch_size
        w = h // 2 + 1

        if uniform_drop:
            print('using uniform droppath with expect rate', drop_path_rate)
            dpr = [drop_path_rate for _ in range(depth)]  # stochastic depth decay rule
        else:
            print('using linear droppath with expect rate', drop_path_rate * 0.5)
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        # dpr = [drop_path_rate for _ in range(depth)]  # stochastic depth decay rule

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, mlp_ratio=mlp_ratio,
                drop=drop_rate, drop_path=dpr[i], norm_layer=norm_layer, h=h, w=w)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        if dropcls > 0:
            print('dropout %.2f before classifier' % dropcls)
            self.final_dropout = nn.Dropout(p=dropcls)
        else:
            self.final_dropout = nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x).mean(1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.final_dropout(x)
        x = self.head(x)
        return x


class GFNetPyramid(nn.Module):

    def __init__(self, img_size=224, patch_size=4, num_classes=1000, embed_dim=[64, 128, 256, 512], depth=[2, 2, 10, 4],
                 mlp_ratio=[4, 4, 4, 4],
                 drop_rate=0., drop_path_rate=0., norm_layer=None, init_values=0.001, no_layerscale=False, dropcls=0,
                 patch_embed_fix=0,

                 patchmix_prob=0.5,
                 fourier_patch_mixup_flag=0,
                 fourier_mix_or_cutmix_prob=0.5,
                 fourier_patch_mixup_alpha=0,
                 fourier_patch_mixup_beta=0,
                 patch_cutmix_beta_flag=0,

                 phase_mask_size=0.5,
                 phase_batch_mean_flag=0,
                 phase_rfft_flag=0,
                 fourier_patch_phase_rate=0.35,

                 fourier_ablation_flag=0,
                 fourier_patch_protect_flag=0,

                 stylize_mask_size=0.5,
                 stylize_factor=0.8,

                 DSU_flag=0,
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim[-1]  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = nn.ModuleList()

        patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim[0])
        num_patches = patch_embed.num_patches

        patch_embed.init_weights(stop_grad_conv1=patch_embed_fix)

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim[0]))
        self.num_patches = num_patches

        self.patch_embed.append(patch_embed)

        sizes = [56, 28, 14, 7]
        for i in range(4):
            sizes[i] = sizes[i] * img_size // 224

        for i in range(3):
            patch_embed = DownLayer(sizes[i], embed_dim[i], embed_dim[i + 1])
            num_patches = patch_embed.num_patches
            self.patch_embed.append(patch_embed)

        self.pos_drop = nn.Dropout(p=drop_rate)
        self.blocks = nn.ModuleList()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]  # stochastic depth decay rule
        cur = 0
        for i in range(4):
            h = sizes[i]
            w = h // 2 + 1

            if no_layerscale:
                print('using standard block')
                blk = nn.Sequential(*[
                    Block(
                        dim=embed_dim[i], mlp_ratio=mlp_ratio[i],
                        drop=drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer, h=h, w=w)
                    for j in range(depth[i])
                ])
            else:
                print('using layerscale block')
                blk = nn.Sequential(*[
                    BlockLayerScale(
                        dim=embed_dim[i], mlp_ratio=mlp_ratio[i],
                        drop=drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer, h=h, w=w,
                        init_values=init_values)
                    for j in range(depth[i])
                ])
            self.blocks.append(blk)
            cur += depth[i]

        # Classifier head
        self.norm = norm_layer(embed_dim[-1])

        self.head = nn.Linear(self.num_features, num_classes)

        if dropcls > 0:
            print('dropout %.2f before classifier' % dropcls)
            self.final_dropout = nn.Dropout(p=dropcls)
        else:
            self.final_dropout = nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

        self.target = torch.arange(self.num_patches)

        self.fourier_patch_mixup_flag = fourier_patch_mixup_flag
        self.fourier_mix_or_cutmix_prob = fourier_mix_or_cutmix_prob

        self.phase_mask_size = phase_mask_size
        self.phase_batch_mean_flag = phase_batch_mean_flag
        self.phase_rfft_flag = phase_rfft_flag
        self.fourier_patch_phase_rate = fourier_patch_phase_rate
        self.fourier_patch_mixup_alpha = fourier_patch_mixup_alpha
        self.fourier_patch_mixup_beta = fourier_patch_mixup_beta
        self.patch_cutmix_beta_flag = patch_cutmix_beta_flag
        self.patchmix_prob = patchmix_prob


        self.fourier_ablation_flag = fourier_ablation_flag
        self.fourier_patch_protect_flag = fourier_patch_protect_flag

        self.stylize_mask_size = stylize_mask_size
        self.stylize_factor = stylize_factor

        self.DSU_flag = DSU_flag
        self.DSU = DSU()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def phase_scores(self, img, mask_size=0.5, batch_mean_flag=0, rfft_flag=0):
        B, N, C = img.shape
        h = w = int(math.sqrt(N))
        img = img.view(B, h, w, C)
        img = img.to(torch.float32)

        if rfft_flag == 1:
            img_fft = torch.fft.rfft2(img, dim=(1, 2), norm='ortho')
        else:
            img_fft = torch.fft.fft2(img, dim=(1, 2), norm='ortho')
        img_abs, img_pha = torch.abs(img_fft), torch.angle(img_fft)

        _, h_fft, w_fft, _ = img_fft.shape
        h_crop = int(h_fft * sqrt(mask_size))
        w_crop = int(w_fft * sqrt(mask_size))
        h_start = h_fft // 2 - h_crop // 2
        masks = torch.ones_like(img_abs)

        if rfft_flag == 1:
            w_start = 0
            img_abs = torch.fft.fftshift(img_abs, dim=(1))
        else:
            w_start = w // 2 - w_crop // 2
            img_abs = torch.fft.fftshift(img_abs, dim=(1, 2))

        masks[:, h_start:h_start + h_crop, w_start:w_start + w_crop, :] = 0

        img_abs_temp = img_abs.clone()
        img_abs = img_abs * masks.cuda()
        if batch_mean_flag == 0:
            freq_avg = torch.mean(img_abs_temp[:, h_start:h_start + h_crop, w_start:w_start + w_crop, :],
                                  dim=(1, 2, 3), keepdim=True)  # Bx1x1x1
        else:
            freq_avg = torch.mean(img_abs_temp[:, h_start:h_start + h_crop, w_start:w_start + w_crop, :],
                                  dim=(0, 1, 2, 3), keepdim=True)  # 1x1x1x1

        freq_avg_mask = torch.zeros_like(img_abs)
        freq_avg_mask[:, h_start:h_start + h_crop, w_start:w_start + w_crop, :] = 1
        freq_avg_mask = freq_avg * freq_avg_mask.cuda()
        img_abs += freq_avg_mask

        if rfft_flag == 1:
            img_abs = torch.fft.ifftshift(img_abs, dim=(1))
        else:
            img_abs = torch.fft.ifftshift(img_abs, dim=(1, 2))

        img_phase_masked = img_abs * (np.e ** (1j * img_pha))
        if rfft_flag == 1:
            img_phase_masked = torch.fft.irfft2(img_phase_masked, s=(h, w), dim=(1, 2), norm='ortho')
        else:
            img_phase_masked = torch.fft.ifft2(img_phase_masked, s=(h, w), dim=(1, 2), norm='ortho')
            img_phase_masked = torch.real(img_phase_masked)
        img_phase_masked = img_phase_masked.reshape(B, N, C)
        return img_phase_masked

    def fourier_stylized_images(self, img, mask_size=0.5, alpha=1.0, factor=0.8):
        B, N, C = img.shape
        h = w = int(math.sqrt(N))
        img = img.view(B, h, w, C)
        img = img.to(torch.float32)

        img_fft = torch.fft.fft2(img, dim=(1, 2), norm='ortho')
        img_abs, img_pha = torch.abs(img_fft), torch.angle(img_fft)

        _, h_fft, w_fft, _ = img_fft.shape
        h_crop = int(h_fft * sqrt(mask_size))
        w_crop = int(w_fft * sqrt(mask_size))
        h_start = h_fft // 2 - h_crop // 2
        w_start = w_fft // 2 - w_crop // 2

        img_abs = torch.fft.fftshift(img_abs, dim=(1, 2))
        img_abs_ = img_abs.clone()
        # miu_of_elem = torch.mean(img_abs_[:, h_start:h_start + h_crop, w_start:w_start + w_crop, :], dim=0, keepdim=True)
        var_of_elem = torch.var(img_abs_[:, h_start:h_start + h_crop, w_start:w_start + w_crop, :], dim=0, keepdim=True)
        sig_of_elem = (var_of_elem + 1e-6).sqrt()  # 1xHxWxC

        epsilon_sig = torch.randn_like(img_abs[:, h_start:h_start + h_crop, w_start:w_start + w_crop, :])  # BxHxWxC N(0,1)
        gamma = epsilon_sig * sig_of_elem * factor

        img_abs[:, h_start:h_start + h_crop, w_start:w_start + w_crop, :] = \
            img_abs[:, h_start:h_start + h_crop, w_start:w_start + w_crop,:] + gamma

        img_abs = torch.fft.ifftshift(img_abs, dim=(1, 2))
        img_stylized = img_abs * (np.e ** (1j * img_pha))

        img_stylized = torch.fft.ifft2(img_stylized, s=(h, w), dim=(1, 2), norm='ortho')
        img_stylized = torch.real(img_stylized)
        img_stylized = img_stylized.reshape(B, N, C)
        return img_stylized

    def phase_patch_mixup(self, img, mixup_alpha=4.0, mixup_beta=1.0, mixup_prob=0.5,
                          patch_protect_flag=0, phase_patch_rate=0.75, labels=None):
        B, N, C = img.shape

        img_phase = self.phase_scores(img, mask_size=self.phase_mask_size, batch_mean_flag=self.phase_batch_mean_flag,
                                      rfft_flag=self.phase_rfft_flag)
        if patch_protect_flag == 1:
            mask_phase = self.fourier_phase_mask(img_phase, mask_prob=phase_patch_rate).unsqueeze(dim=2)  # BxN
            img_phase = mask_phase * img

        # shuffle the img along the patch dim
        noise = torch.rand(B, N, device=img.device)  # noise in [0, 1] B x N
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        img_patch_shuffled = torch.gather(img, dim=1, index=ids_shuffle.unsqueeze(-1).repeat(1, 1, C))

        if self.fourier_ablation_flag == 0:
            # generate mixed images
            perm = torch.randperm(B)
            lam = np.random.beta(mixup_alpha, mixup_beta)
            img_ELEAS = img_phase * lam + img_patch_shuffled[perm] * (1 - lam)
        elif self.fourier_ablation_flag == 1:
            img_ELEAS = img_phase
        elif self.fourier_ablation_flag == 2:
            img_ELEAS = img_patch_shuffled
        elif self.fourier_ablation_flag == 3:
            # Edge and Original Image
            perm = torch.randperm(B)
            lam = np.random.beta(mixup_alpha, mixup_beta)
            img_ELEAS = img_phase * lam + img[perm] * (1 - lam)
        elif self.fourier_ablation_flag == 4:
            # Edge and Stylized Original Image
            img_stylized = self.fourier_stylized_images(img, mask_size=self.stylize_mask_size, alpha=1.0,
                                                        factor=self.stylize_factor)
            img_patch_shuffled_stylized = torch.gather(img_stylized, dim=1, index=ids_shuffle.unsqueeze(-1).repeat(1, 1, C))

            perm = torch.randperm(B)
            lam = np.random.beta(mixup_alpha, mixup_beta)
            img_ELEAS = img_phase * lam + img_patch_shuffled_stylized[perm] * (1 - lam)
        elif self.fourier_ablation_flag == 5:
            # Edge + random sampled patches of two samples
            perm_1 = torch.randperm(B)
            perm_2 = torch.randperm(B)

            mask_shuffled = np.where(np.random.rand(N) > 0.5, 1.0, 0.0) # 0-1 mask
            mask_shuffled = torch.tensor(mask_shuffled, device=img_phase.device).reshape(1, N, 1)
            img_shuffled_mixed = img_patch_shuffled[perm_1] * mask_shuffled \
                                 + img_patch_shuffled[perm_2] * (1 - mask_shuffled)
            lam = np.random.beta(mixup_alpha, mixup_beta)

            img_ELEAS = img_phase * lam + img_shuffled_mixed * (1 - lam)
        elif self.fourier_ablation_flag == 6:
            # Edge + random sampled patches of the whole batch
            img_patch_bank = img_patch_shuffled.view(B*N, -1)
            perm = torch.randperm(B*N)
            img_patch_sampled = img_patch_bank[perm].view(B, N, -1)
            lam = np.random.beta(mixup_alpha, mixup_beta)
            img_ELEAS = img_phase * lam + img_patch_sampled * (1 - lam)
        elif self.fourier_ablation_flag == 7:
            perm = torch.randperm(B)
            lam = np.random.beta(mixup_alpha, mixup_beta)
            img_ELEAS = img * lam + img_patch_shuffled[perm] * (1 - lam)
        elif self.fourier_ablation_flag == 8:
            img_stylized = self.DSU(img)
            img_patch_shuffled_stylized = torch.gather(img_stylized, dim=1,
                                                       index=ids_shuffle.unsqueeze(-1).repeat(1, 1, C))
            perm = torch.randperm(B)
            lam = np.random.beta(mixup_alpha, mixup_beta)
            img_ELEAS = img_phase * lam + img_patch_shuffled_stylized[perm] * (1 - lam)
        elif self.fourier_ablation_flag == 9:
            # Origin and Origin Mixing
            perm = torch.randperm(B)
            lam = np.random.beta(mixup_alpha, mixup_beta)
            img_ELEAS = img * lam + img[perm] * (1 - lam)

        img_mask = np.where(np.random.rand(B) > mixup_prob, 1.0, 0.0)
        img_mask = torch.tensor(img_mask, device=img.device).reshape(-1, 1, 1)
        img_batch = img * img_mask + img_ELEAS * (1 - img_mask)

        img_batch = img_batch.to(torch.float32)
        return img_batch

    def fourier_phase_mask(self, scores, mask_prob=0.35):
        # scores: B, N, C
        B, N, C = scores.shape
        scores_channel_mean = torch.mean(scores, dim=2, keepdim=False)  # BxN
        K_phase = int(N * mask_prob)  # percent of the phase-related patches

        if mask_prob == 1.0:
            K_phase = N - 1

        threshold = torch.sort(scores_channel_mean, dim=1, descending=True)[0][:, K_phase]
        threshold_expand = threshold.view(B, 1).expand(B, N)
        mask_phase = torch.where(scores_channel_mean > threshold_expand,
                                   torch.tensor(1.).cuda(), torch.tensor(0.).cuda())
        # values, indices = torch.topk(scores_channel_mean, k=K, dim=1, largest=True, sorted=False)
        return mask_phase

    def phase_patch_cutmix(self, img, phase_patch_rate=0.75, mixup_alpha=4.0, mixup_beta=1.0,
                           mixup_prob=0.5, beta_flag=0, labels=None):
        B, N, C = img.shape
        scores = self.phase_scores(img, mask_size=self.phase_mask_size, batch_mean_flag=self.phase_batch_mean_flag,
                                   rfft_flag=self.phase_rfft_flag)

        # generate the mixup mask
        lam = phase_patch_rate
        if beta_flag == 1:
            lam = np.random.beta(mixup_alpha, mixup_beta)
        mask_phase = self.fourier_phase_mask(scores, mask_prob=lam).unsqueeze(dim=2)  # BxN

        # shuffle the img along the patch dim
        noise = torch.rand(B, N, device=img.device)  # noise in [0, 1] B x N
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        img_patch_shuffled = torch.gather(img, dim=1, index=ids_shuffle.unsqueeze(-1).repeat(1, 1, C))

        if self.fourier_ablation_flag == 0:
            # generate mixed images
            perm = torch.randperm(B)
            img_mixed = mask_phase * img + (1 - mask_phase) * img_patch_shuffled[perm]
        elif self.fourier_ablation_flag == 1:
            img_mixed = mask_phase * img
        elif self.fourier_ablation_flag == 2:
            img_mixed = img_patch_shuffled
        elif self.fourier_ablation_flag == 3:
            # Edge and Original Image
            perm = torch.randperm(B)
            img_mixed = img * mask_phase + img[perm] * (1 - mask_phase)
        elif self.fourier_ablation_flag == 4:
            # Edge and Stylized Original Image
            img_stylized = self.fourier_stylized_images(img, mask_size=self.stylize_mask_size,
                                                        alpha=1.0, factor=self.stylize_factor)
            img_patch_shuffled_stylized = torch.gather(img_stylized, dim=1,
                                                       index=ids_shuffle.unsqueeze(-1).repeat(1, 1, C))

            perm = torch.randperm(B)
            img_mixed = img * mask_phase + img_patch_shuffled_stylized[perm] * (1 - mask_phase)
        elif self.fourier_ablation_flag == 5:
            # Edge + random sampled patches of two samples
            perm_1 = torch.randperm(B)
            perm_2 = torch.randperm(B)

            mask_shuffled = np.where(np.random.rand(N) > 0.5, 1.0, 0.0)  # 0-1 mask
            mask_shuffled = torch.tensor(mask_shuffled, device=img.device).reshape(1, N, 1)

            img_shuffled_mixed = img_patch_shuffled[perm_1] * mask_shuffled \
                                 + img_patch_shuffled[perm_2] * (1 - mask_shuffled)
            img_mixed = img * mask_phase + img_shuffled_mixed * (1 - mask_phase)
        elif self.fourier_ablation_flag == 6:
            # Edge + random sampled patches of the whole batch
            img_patch_bank = img.clone().reshape(B * N, -1)
            perm = torch.randperm(B * N)
            img_patch_sampled = img_patch_bank[perm].view(B, N, -1)
            img_mixed = img * mask_phase + img_patch_sampled * (1 - mask_phase)
        elif self.fourier_ablation_flag == 7:
            perm = torch.randperm(B)
            # random generate mask
            mask_random = (torch.rand([B, N, 1]) < lam).float().cuda()
            img_mixed = img * mask_random + img_patch_shuffled[perm] * (1 - mask_random)
        elif self.fourier_ablation_flag == 8:
            # Edge and Stylized Original Image
            img_stylized = self.DSU(img)
            img_patch_shuffled_stylized = torch.gather(img_stylized, dim=1,
                                                       index=ids_shuffle.unsqueeze(-1).repeat(1, 1, C))

            perm = torch.randperm(B)
            img_mixed = img * mask_phase + img_patch_shuffled_stylized[perm] * (1 - mask_phase)
        elif self.fourier_ablation_flag == 9:
            # Origin and Origin Mixing
            perm = torch.randperm(B)
            # random generate mask
            mask_random = (torch.rand([B, N, 1]) < lam).float().cuda()
            img_mixed = img * mask_random + img[perm] * (1 - mask_random)


        # apply patch_cutmix_phase to half of the images
        img_mask = np.where(np.random.rand(B) > mixup_prob, 1.0, 0.0)
        img_mask = torch.tensor(img_mask, device=img.device).reshape(-1, 1, 1)
        img_batch = img * img_mask + img_mixed * (1 - img_mask)
        img_batch = img_batch.to(torch.float32)

        return img_batch

    def patch_mix(self, x, lam, perm):
        x1_ = x.clone()
        x2_ = x1_[perm]
        x = x1_ * lam + x2_ * (1 - lam)
        return x

    def forward_features(self, x, PatchMix_layer=[], labels=None):
        lam, perm = None, None
        for i in range(4):
            x = self.patch_embed[i](x)

            if self.training and (i in PatchMix_layer):
                if self.fourier_patch_mixup_flag != 0:
                    if random.random() < self.fourier_mix_or_cutmix_prob:
                        x = self.phase_patch_mixup(img=x, mixup_alpha=self.fourier_patch_mixup_alpha,
                                                   mixup_beta=self.fourier_patch_mixup_beta,
                                                   mixup_prob=self.patchmix_prob,
                                                   patch_protect_flag=self.fourier_patch_protect_flag,
                                                   phase_patch_rate=self.fourier_patch_phase_rate,
                                                   labels=labels)
                    else:
                        x = self.phase_patch_cutmix(img=x, phase_patch_rate=self.fourier_patch_phase_rate,
                                                    mixup_alpha=self.fourier_patch_mixup_alpha,
                                                    mixup_beta=self.fourier_patch_mixup_beta,
                                                    mixup_prob=self.patchmix_prob,
                                                    beta_flag=self.patch_cutmix_beta_flag,
                                                    labels=labels)

            if self.training and self.DSU_flag == 1:
                x = self.DSU(x)

            if i == 0:
                x = x + self.pos_embed
            x = self.blocks[i](x)
            if self.training and self.DSU_flag == 2:
                x = self.DSU(x)

        return x, [lam, perm]

    def forward(self, x, PatchMix_layer=None, labels=None):
        x_f, [lam_forward, perm_forward] = self.forward_features(x, PatchMix_layer=PatchMix_layer, labels=labels)
        x_norm = self.norm(x_f)

        x_feat = x_norm.mean(1)
        x_feat = self.final_dropout(x_feat)
        x_cls = self.head(x_feat)

        outs = Munch(suip=x_cls)
        outs.lam_forward = lam_forward
        outs.perm_forward = perm_forward
        return outs


def resize_pos_embed(posemb, posemb_new):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    _logger.info('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
    ntok_new = posemb_new.shape[1]
    if True:
        posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
        ntok_new -= 1
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    gs_new = int(math.sqrt(ntok_new))
    _logger.info('Position embedding grid-size from %s to %s', gs_old, gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(gs_new, gs_new), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new * gs_new, -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb


def checkpoint_filter_fn(state_dict, model):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    if 'model' in state_dict:
        # For deit models
        state_dict = state_dict['model']
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
            # For old models that I trained prior to conv based patchification
            O, I, H, W = model.patch_embed.proj.weight.shape
            v = v.reshape(O, -1, H, W)
        elif k == 'pos_embed' and v.shape != model.pos_embed.shape:
            # To resize pos embedding when using model at different size from pretrained weights
            v = resize_pos_embed(v, model.pos_embed)
        out_dict[k] = v
    return out_dict
