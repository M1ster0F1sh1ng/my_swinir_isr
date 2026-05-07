"""
增强版工具函数
包含：NIQE友好损失函数、退化模型、评估指标、EMA、训练辅助工具
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import random
from scipy import ndimage
from collections import deque

# ═══════════════════════════════════════════════════════════════════════════════
# 基础工具
# ═══════════════════════════════════════════════════════════════════════════════

class AverageMeter:
    """计算和存储平均值和当前值"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# ═══════════════════════════════════════════════════════════════════════════════
# 可微分的 NIQE 近似损失
# ═══════════════════════════════════════════════════════════════════════════════

class NIQEFriendlyLoss(nn.Module):
    """
    NIQE (Natural Image Quality Evaluator) 友好损失函数

    NIQE 通过测量图像与自然图像统计模型的距离来评估质量。
    为了使输出对 NIQE 友好，我们需要匹配自然图像的局部统计特性。

    包含以下组件：
    1. 局部对比度匹配（MS-SSIM）
    2. 频域统计匹配（FFT Loss）
    3. 梯度/边缘分布匹配
    4. 局部方差匹配
    5. 高阶统计量（偏度、峰度）匹配
    """

    def __init__(self, w_ms_ssim=0.8, w_fft=0.3, w_grad=0.4, w_contrast=0.2,
                 w_skewness=0.1, w_kurtosis=0.1, w_color=0.15,
                 device='cuda'):
        super().__init__()
        self.w_ms_ssim = w_ms_ssim
        self.w_fft = w_fft
        self.w_grad = w_grad
        self.w_contrast = w_contrast
        self.w_skewness = w_skewness
        self.w_kurtosis = w_kurtosis
        self.w_color = w_color
        self.device = device

        # Sobel 算子用于边缘检测
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

        # Laplacian 算子
        laplacian = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                                 dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('laplacian', laplacian)

        # 高斯核用于多尺度
        self.gaussian_kernels = self._create_gaussian_pyramid_kernels()

    def _create_gaussian_pyramid_kernels(self):
        """创建高斯金字塔核"""
        kernels = []
        for sigma in [0.5, 1.0, 2.0, 4.0]:
            size = int(6 * sigma) | 1  # 确保奇数
            x = torch.arange(size, dtype=torch.float32) - size // 2
            g = torch.exp(-x**2 / (2 * sigma**2))
            g = g / g.sum()
            kernel_2d = g.outer(g).unsqueeze(0).unsqueeze(0)
            kernels.append(kernel_2d.to(self.device))
        return kernels

    def _apply_sobel(self, img):
        """应用 Sobel 算子提取边缘"""
        b, c, h, w = img.shape
        # 对每通道分别计算
        grad_x = F.conv2d(img.view(b * c, 1, h, w), self.sobel_x,
                          padding=1).view(b, c, h, w)
        grad_y = F.conv2d(img.view(b * c, 1, h, w), self.sobel_y,
                          padding=1).view(b, c, h, w)
        magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
        return magnitude

    def _apply_laplacian(self, img):
        """应用 Laplacian 算子"""
        b, c, h, w = img.shape
        lap = F.conv2d(img.view(b * c, 1, h, w), self.laplacian,
                       padding=1).view(b, c, h, w)
        return lap

    def _gaussian_blur(self, img, kernel):
        """高斯模糊"""
        b, c, h, w = img.shape
        kernel_size = kernel.shape[-1]
        pad = kernel_size // 2
        # 分别对每通道应用
        img_pad = F.pad(img, (pad, pad, pad, pad), mode='reflect')
        blurred = F.conv2d(img_pad.view(b * c, 1, h + 2*pad, w + 2*pad),
                           kernel.to(img.device),
                           padding=0).view(b, c, h, w)
        return blurred

    def _fft_loss(self, pred, target):
        """
        频域损失 - 匹配频域统计特性
        NIQE 对频域分布非常敏感
        """
        # 计算幅值谱
        pred_fft = torch.fft.rfft2(pred, dim=(-2, -1))
        target_fft = torch.fft.rfft2(target, dim=(-2, -1))

        pred_amp = torch.log(torch.abs(pred_fft) + 1e-8)
        target_amp = torch.log(torch.abs(target_fft) + 1e-8)

        # 频域L1损失
        amp_loss = F.l1_loss(pred_amp, target_amp)

        # 频域相位损失（低频区域更重要）
        pred_phase = torch.angle(pred_fft)
        target_phase = torch.angle(target_fft)
        phase_diff = 1 - torch.cos(pred_phase - target_phase)
        phase_loss = phase_diff.mean()

        return amp_loss + 0.1 * phase_loss

    def _ms_ssim_loss(self, pred, target, levels=3):
        """
        多尺度 SSIM 损失
        比单尺度 SSIM 更能捕捉结构信息
        """
        weights = torch.tensor([0.0448, 0.2856, 0.6696], device=pred.device)

        msssim_vals = []
        for i in range(levels):
            if i > 0:
                # 下采样
                pred = F.avg_pool2d(pred, 2)
                target = F.avg_pool2d(target, 2)

            ssim_val = self._ssim(pred, target)
            msssim_vals.append(ssim_val)

        # 加权组合
        msssim = 1
        for i in range(levels):
            msssim *= msssim_vals[i] ** weights[i]

        return 1 - msssim

    def _ssim(self, pred, target, window_size=11):
        """计算单尺度 SSIM"""
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2

        mu1 = F.avg_pool2d(pred, window_size, stride=1, padding=window_size//2)
        mu2 = F.avg_pool2d(target, window_size, stride=1, padding=window_size//2)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.avg_pool2d(pred ** 2, window_size, stride=1, padding=window_size//2) - mu1_sq
        sigma2_sq = F.avg_pool2d(target ** 2, window_size, stride=1, padding=window_size//2) - mu2_sq
        sigma12 = F.avg_pool2d(pred * target, window_size, stride=1, padding=window_size//2) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / \
                   ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))

        return ssim_map.mean()

    def _gradient_loss(self, pred, target):
        """
        梯度损失 - 保持边缘锐利但不产生伪影
        使用 Sobel 和 Laplacian 的组合
        """
        pred_grad = self._apply_sobel(pred)
        target_grad = self._apply_sobel(target)

        # Charbonnier 损失（对梯度更友好）
        eps = 1e-6
        grad_loss = torch.mean(torch.sqrt((pred_grad - target_grad) ** 2 + eps))

        # Laplacian 损失（二阶导数，防止过度平滑）
        pred_lap = self._apply_laplacian(pred)
        target_lap = self._apply_laplacian(target)
        lap_loss = torch.mean(torch.sqrt((pred_lap - target_lap) ** 2 + eps))

        return grad_loss + 0.5 * lap_loss

    def _local_contrast_loss(self, pred, target, window_size=8):
        """
        局部对比度损失
        NIQE 非常依赖局部对比度的统计特性
        """
        # 计算局部均值和方差
        pred_mean = F.avg_pool2d(pred, window_size, stride=window_size)
        target_mean = F.avg_pool2d(target, window_size, stride=window_size)

        pred_var = F.avg_pool2d(pred ** 2, window_size, stride=window_size) - pred_mean ** 2
        target_var = F.avg_pool2d(target ** 2, window_size, stride=window_size) - target_mean ** 2

        # 对比度 = 标准差 / 均值（局部变异系数近似）
        pred_contrast = torch.sqrt(pred_var + 1e-8)
        target_contrast = torch.sqrt(target_var + 1e-8)

        return F.l1_loss(pred_contrast, target_contrast)

    def _higher_order_stats_loss(self, pred, target):
        """
        高阶统计量损失 - 匹配偏度和峰度
        这对 NIQE 非常重要，因为 NIQE 基于 MSCN 系数的统计模型
        """
        # 计算 MSCN-like 系数（均值减除对比度归一化）
        def compute_mscn(x, window_size=7):
            mu = F.avg_pool2d(x, window_size, stride=1, padding=window_size//2)
            sigma = torch.sqrt(F.avg_pool2d(x**2, window_size, stride=1,
                                             padding=window_size//2) - mu**2 + 1e-8)
            return (x - mu) / (sigma + 1e-8)

        pred_mscn = compute_mscn(pred)
        target_mscn = compute_mscn(target)

        # 偏度损失
        pred_skew = torch.mean(pred_mscn ** 3, dim=[2, 3])
        target_skew = torch.mean(target_mscn ** 3, dim=[2, 3])
        skew_loss = F.mse_loss(pred_skew, target_skew)

        # 峰度损失
        pred_kurt = torch.mean(pred_mscn ** 4, dim=[2, 3])
        target_kurt = torch.mean(target_mscn ** 4, dim=[2, 3])
        kurt_loss = F.mse_loss(pred_kurt, target_kurt)

        return skew_loss, kurt_loss

    def _color_distribution_loss(self, pred, target):
        """
        颜色分布损失 - 匹配颜色直方图
        """
        # 计算每个通道的直方图近似（使用软量化）
        bins = 32
        pred_hist = self._soft_histogram(pred, bins)
        target_hist = self._soft_histogram(target, bins)

        # 直方图交叉损失
        hist_loss = F.l1_loss(pred_hist, target_hist)

        # 颜色均值和标准差匹配
        pred_mean = pred.mean(dim=[2, 3])
        target_mean = target.mean(dim=[2, 3])
        mean_loss = F.mse_loss(pred_mean, target_mean)

        pred_std = pred.std(dim=[2, 3])
        target_std = target.std(dim=[2, 3])
        std_loss = F.mse_loss(pred_std, target_std)

        return hist_loss + mean_loss + std_loss

    def _soft_histogram(self, x, bins=32, sigma=0.02):
        """
        可微分的软直方图
        """
        bin_centers = torch.linspace(0, 1, bins, device=x.device)
        # x: [B, C, H, W]
        x_flat = x.flatten(2)  # [B, C, H*W]
        # 计算到每个 bin 中心的距离
        dist = x_flat.unsqueeze(-1) - bin_centers  # [B, C, H*W, bins]
        # 高斯核
        weights = torch.exp(-dist**2 / (2 * sigma**2))
        # 归一化
        hist = weights.sum(dim=2)  # [B, C, bins]
        hist = hist / (hist.sum(dim=-1, keepdim=True) + 1e-8)
        return hist

    def forward(self, pred, target):
        """
        计算 NIQE 友好损失

        Returns:
            loss: 总损失
            loss_dict: 各组件损失字典
        """
        # 确保值在 [0, 1] 范围
        pred = torch.clamp(pred, 0, 1)
        target = torch.clamp(target, 0, 1)

        losses = {}

        # MS-SSIM 损失
        if self.w_ms_ssim > 0:
            losses['ms_ssim'] = self._ms_ssim_loss(pred, target)
        else:
            losses['ms_ssim'] = torch.tensor(0.0, device=pred.device)

        # 频域损失
        if self.w_fft > 0:
            losses['fft'] = self._fft_loss(pred, target)
        else:
            losses['fft'] = torch.tensor(0.0, device=pred.device)

        # 梯度损失
        if self.w_grad > 0:
            losses['gradient'] = self._gradient_loss(pred, target)
        else:
            losses['gradient'] = torch.tensor(0.0, device=pred.device)

        # 局部对比度损失
        if self.w_contrast > 0:
            losses['contrast'] = self._local_contrast_loss(pred, target)
        else:
            losses['contrast'] = torch.tensor(0.0, device=pred.device)

        # 高阶统计量损失
        if self.w_skewness > 0 or self.w_kurtosis > 0:
            skew_loss, kurt_loss = self._higher_order_stats_loss(pred, target)
            losses['skewness'] = skew_loss
            losses['kurtosis'] = kurt_loss
        else:
            losses['skewness'] = torch.tensor(0.0, device=pred.device)
            losses['kurtosis'] = torch.tensor(0.0, device=pred.device)

        # 颜色分布损失
        if self.w_color > 0:
            losses['color'] = self._color_distribution_loss(pred, target)
        else:
            losses['color'] = torch.tensor(0.0, device=pred.device)

        # 加权总损失
        total_loss = (
            self.w_ms_ssim * losses['ms_ssim'] +
            self.w_fft * losses['fft'] +
            self.w_grad * losses['gradient'] +
            self.w_contrast * losses['contrast'] +
            self.w_skewness * losses['skewness'] +
            self.w_kurtosis * losses['kurtosis'] +
            self.w_color * losses['color']
        )

        losses['total'] = total_loss
        return total_loss, losses


# ═══════════════════════════════════════════════════════════════════════════════
# Charbonnier 损失（比 L1 更平滑）
# ═══════════════════════════════════════════════════════════════════════════════

class CharbonnierLoss(nn.Module):
    """Charbonnier 损失 = sqrt((x-y)^2 + eps)"""
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps))


# ═══════════════════════════════════════════════════════════════════════════════
# 频域损失
# ═══════════════════════════════════════════════════════════════════════════════

class FFTLoss(nn.Module):
    """频域损失 - 在频域中比较图像"""
    def __init__(self, loss_type='l1'):
        super().__init__()
        self.loss_type = loss_type

    def forward(self, pred, target):
        # 计算 FFT
        pred_fft = torch.fft.rfft2(pred, dim=(-2, -1))
        target_fft = torch.fft.rfft2(target, dim=(-2, -1))

        # 幅值
        pred_amp = torch.abs(pred_fft)
        target_amp = torch.abs(target_fft)

        if self.loss_type == 'l1':
            amp_loss = F.l1_loss(pred_amp, target_amp)
        elif self.loss_type == 'l2':
            amp_loss = F.mse_loss(pred_amp, target_amp)
        elif self.loss_type == 'log':
            amp_loss = F.l1_loss(torch.log(pred_amp + 1), torch.log(target_amp + 1))

        return amp_loss


# ═══════════════════════════════════════════════════════════════════════════════
# 边缘感知损失（带方向性）
# ═══════════════════════════════════════════════════════════════════════════════

class EdgeAwareLoss(nn.Module):
    """边缘感知损失 - 使用多尺度方向性梯度"""

    def __init__(self, scales=[1, 2, 4]):
        super().__init__()
        self.scales = scales

        # Sobel 算子
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def _gradients(self, img):
        """计算多尺度梯度"""
        b, c, h, w = img.shape
        grads = []
        for scale in self.scales:
            if scale > 1:
                img_scaled = F.avg_pool2d(img, scale)
            else:
                img_scaled = img

            _, _, hs, ws = img_scaled.shape
            grad_x = F.conv2d(img_scaled.view(b * c, 1, hs, ws),
                              self.sobel_x.to(img.device), padding=1).view(b, c, hs, ws)
            grad_y = F.conv2d(img_scaled.view(b * c, 1, hs, ws),
                              self.sobel_y.to(img.device), padding=1).view(b, c, hs, ws)

            grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
            grads.append(grad_mag)
        return grads

    def forward(self, pred, target):
        pred_grads = self._gradients(pred)
        target_grads = self._gradients(target)

        loss = 0
        for pg, tg in zip(pred_grads, target_grads):
            # Charbonnier 形式
            loss += torch.mean(torch.sqrt((pg - tg) ** 2 + 1e-6))

        return loss / len(self.scales)


# ═══════════════════════════════════════════════════════════════════════════════
# 组合损失函数（完整版）
# ═══════════════════════════════════════════════════════════════════════════════

class CompleteLoss(nn.Module):
    """
    完整的损失函数组合
    支持两阶段训练：
    - 第一阶段：侧重像素级精度（Charbonnier + SSIM）
    - 第二阶段：侧重感知质量（NIQE + LPIPS + 感知损失）
    """

    def __init__(self, stage='pixel', device='cuda', **kwargs):
        """
        Args:
            stage: 'pixel' 或 'perceptual'
            device: 设备
            **kwargs: 各损失的权重参数
        """
        super().__init__()
        self.stage = stage
        self.device = device

        # 像素级损失
        self.charbonnier = CharbonnierLoss(eps=1e-6)

        # NIQE 友好损失
        self.niqe_loss = NIQEFriendlyLoss(device=device, **kwargs.get('niqe_kwargs', {}))

        # 频域损失
        self.fft_loss = FFTLoss()

        # 边缘感知损失
        self.edge_loss = EdgeAwareLoss()

        # LPIPS（如果可用）
        try:
            import lpips
            self.lpips = lpips.LPIPS(net='alex').to(device)
            self.has_lpips = True
        except ImportError:
            print("警告: lpips 未安装，跳过 LPIPS 损失")
            self.has_lpips = False

        # 权重（根据阶段自动调整）
        self._set_weights(stage, kwargs)

    def _set_weights(self, stage, kwargs):
        """根据训练阶段设置权重"""
        if stage == 'pixel':
            # 第一阶段：追求像素精度
            self.w_charbonnier = kwargs.get('w_charbonnier', 1.0)
            self.w_ssim = kwargs.get('w_ssim', 0.5)
            self.w_fft = kwargs.get('w_fft', 0.3)
            self.w_edge = kwargs.get('w_edge', 0.2)
            self.w_niqe = kwargs.get('w_niqe', 0.3)
            self.w_lpips = kwargs.get('w_lpips', 0.0)
        elif stage == 'perceptual':
            # 第二阶段：追求感知质量
            self.w_charbonnier = kwargs.get('w_charbonnier', 0.5)
            self.w_ssim = kwargs.get('w_ssim', 0.5)
            self.w_fft = kwargs.get('w_fft', 0.5)
            self.w_edge = kwargs.get('w_edge', 0.4)
            self.w_niqe = kwargs.get('w_niqe', 1.0)
            self.w_lpips = kwargs.get('w_lpips', 0.3)
        else:
            # 自定义权重
            self.w_charbonnier = kwargs.get('w_charbonnier', 1.0)
            self.w_ssim = kwargs.get('w_ssim', 0.5)
            self.w_fft = kwargs.get('w_fft', 0.3)
            self.w_edge = kwargs.get('w_edge', 0.3)
            self.w_niqe = kwargs.get('w_niqe', 0.5)
            self.w_lpips = kwargs.get('w_lpips', 0.2)

    def forward(self, pred, target):
        """
        计算完整损失

        Returns:
            total_loss: 加权总损失
            loss_dict: 各组件损失字典
        """
        losses = {}

        # Charbonnier 损失（像素级）
        if self.w_charbonnier > 0:
            losses['charbonnier'] = self.charbonnier(pred, target)
        else:
            losses['charbonnier'] = torch.tensor(0.0, device=self.device)

        # NIQE 友好损失（包含 MS-SSIM, FFT, 梯度, 对比度等）
        if self.w_niqe > 0:
            _, niqe_dict = self.niqe_loss(pred, target)
            losses['niqe_total'] = niqe_dict['total']
            losses['ms_ssim'] = niqe_dict['ms_ssim']
            losses['fft'] = niqe_dict['fft']
            losses['gradient'] = niqe_dict['gradient']
            losses['contrast'] = niqe_dict['contrast']
        else:
            losses['niqe_total'] = torch.tensor(0.0, device=self.device)
            losses['ms_ssim'] = torch.tensor(0.0, device=self.device)
            losses['fft'] = torch.tensor(0.0, device=self.device)
            losses['gradient'] = torch.tensor(0.0, device=self.device)
            losses['contrast'] = torch.tensor(0.0, device=self.device)

        # 单独的频域损失
        if self.w_fft > 0 and 'fft' not in losses:
            losses['fft'] = self.fft_loss(pred, target)

        # 边缘感知损失
        if self.w_edge > 0:
            losses['edge'] = self.edge_loss(pred, target)
        else:
            losses['edge'] = torch.tensor(0.0, device=self.device)

        # LPIPS 损失
        if self.w_lpips > 0 and self.has_lpips:
            # LPIPS 需要 [-1, 1] 范围
            pred_lpips = pred * 2 - 1
            target_lpips = target * 2 - 1
            losses['lpips'] = self.lpips(pred_lpips, target_lpips).mean()
        else:
            losses['lpips'] = torch.tensor(0.0, device=self.device)

        # 加权总损失
        total = (
            self.w_charbonnier * losses['charbonnier'] +
            self.w_niqe * losses['niqe_total'] +
            self.w_fft * losses.get('fft', torch.tensor(0.0, device=self.device)) +
            self.w_edge * losses['edge'] +
            self.w_lpips * losses['lpips']
        )

        losses['total'] = total
        return total, losses

    def set_stage(self, stage):
        """切换训练阶段"""
        self.stage = stage
        self._set_weights(stage, {})


# ═══════════════════════════════════════════════════════════════════════════════
# 退化模型（Real-ESRGAN 风格）
# ═══════════════════════════════════════════════════════════════════════════════

class DegradationModel:
    """
    真实世界退化模型
    模拟从清晰图像到低质量图像的退化过程

    支持：
    - 各向异性高斯模糊
    - 运动模糊
    - 噪声（高斯、泊松、椒盐）
    - JPEG 压缩
    - 缩放（最近邻、双线性、双三次）
    - 二阶退化（Real-ESRGAN 风格）
    """

    def __init__(self, scale=2, mode='second_order', device='cpu'):
        """
        Args:
            scale: 下采样倍率
            mode: 'first_order', 'second_order', 'clean'
            device: 计算设备
        """
        self.scale = scale
        self.mode = mode
        self.device = device

        # 退化参数范围（可调）
        self.blur_kernel_range = [3, 13]  # 模糊核大小范围
        self.sigma_range = [0.2, 3.0]     # 高斯模糊 sigma 范围
        self.noise_range = [0, 25]        # 噪声水平范围（0-255）
        self.jpeg_range = [30, 95]        # JPEG 质量范围

    def _random_gaussian_kernel(self, kernel_size=None, sigma=None):
        """生成随机高斯模糊核"""
        if kernel_size is None:
            kernel_size = random.randrange(
                self.blur_kernel_range[0], self.blur_kernel_range[1] + 1, 2)
        if sigma is None:
            sigma = random.uniform(self.sigma_range[0], self.sigma_range[1])

        # 创建二维高斯核
        ax = np.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
        kernel = kernel / np.sum(kernel)
        return torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)

    def _apply_blur(self, img, kernel):
        """应用模糊"""
        b, c, h, w = img.shape
        pad = kernel.shape[-1] // 2
        img_pad = F.pad(img, (pad, pad, pad, pad), mode='reflect')
        # 分别对每通道应用
        blurred = []
        for i in range(c):
            ch = img_pad[:, i:i+1, :, :]
            bl = F.conv2d(ch, kernel.to(img.device), padding=0)
            blurred.append(bl)
        return torch.cat(blurred, dim=1)

    def _add_noise(self, img, noise_type='gaussian', noise_level=None):
        """添加噪声"""
        if noise_level is None:
            noise_level = random.uniform(self.noise_range[0], self.noise_range[1]) / 255.0

        if noise_type == 'gaussian':
            noise = torch.randn_like(img) * noise_level
            return img + noise
        elif noise_type == 'poisson':
            # 泊松噪声（模拟光子噪声）
            noise = torch.poisson(img * 255.0) / 255.0 - img
            scale = noise_level / (noise.abs().mean() + 1e-8)
            return img + noise * scale
        elif noise_type == 'speckle':
            noise = torch.randn_like(img) * noise_level
            return img * (1 + noise)
        else:
            return img

    def _apply_jpeg(self, img, quality=None):
        """模拟 JPEG 压缩（简化版 - 使用量化噪声近似）"""
        if quality is None:
            quality = random.randint(self.jpeg_range[0], self.jpeg_range[1])

        # JPEG 质量转换为量化步长（简化模型）
        q_factor = (100 - quality) / 100.0
        if q_factor < 0.01:
            return img

        # 在 DCT 域近似量化噪声
        # 使用 8x8 块的近似
        b, c, h, w = img.shape

        # 分块
        block_size = 8
        pad_h = (block_size - h % block_size) % block_size
        pad_w = (block_size - w % block_size) % block_size

        if pad_h > 0 or pad_w > 0:
            img_padded = F.pad(img, (0, pad_w, 0, pad_h))
        else:
            img_padded = img

        _, _, hp, wp = img_padded.shape
        nh, nw = hp // block_size, wp // block_size

        # 重塑为块
        img_blocks = img_padded.view(b, c, nh, block_size, nw, block_size)
        img_blocks = img_blocks.permute(0, 1, 2, 4, 3, 5).contiguous()
        img_blocks = img_blocks.view(b, c, nh * nw, block_size * block_size)

        # 模拟量化：添加与 q_factor 相关的噪声
        quant_noise = (torch.rand_like(img_blocks) - 0.5) * q_factor
        img_blocks = img_blocks + quant_noise

        # 恢复形状
        img_blocks = img_blocks.view(b, c, nh, nw, block_size, block_size)
        img_blocks = img_blocks.permute(0, 1, 2, 4, 3, 5).contiguous()
        img_dct = img_blocks.view(b, c, hp, wp)

        # 移除填充
        if pad_h > 0 or pad_w > 0:
            img_dct = img_dct[:, :, :h, :w]

        return img_dct

    def _downsample(self, img, method=None):
        """下采样"""
        methods = ['bicubic', 'bilinear', 'nearest']
        if method is None:
            method = random.choice(methods)

        h, w = img.shape[2:]
        new_h, new_w = h // self.scale, w // self.scale

        if method == 'bicubic':
            return F.interpolate(img, size=(new_h, new_w), mode='bicubic', align_corners=False)
        elif method == 'bilinear':
            return F.interpolate(img, size=(new_h, new_w), mode='bilinear', align_corners=False)
        else:
            return F.interpolate(img, size=(new_h, new_w), mode='nearest')

    def degrade(self, img):
        """
        应用退化

        Args:
            img: [B, C, H, W] 范围 [0, 1]

        Returns:
            lr_img: 退化后的低分辨率图像
        """
        img = img.to(self.device)

        if self.mode == 'clean':
            # 仅下采样（原始方法）
            return self._downsample(img, 'bicubic')

        elif self.mode == 'first_order':
            # 一阶退化：模糊 -> 下采样 -> 加噪
            kernel = self._random_gaussian_kernel()
            img = self._apply_blur(img, kernel)
            img = self._downsample(img)
            img = self._add_noise(img)
            return torch.clamp(img, 0, 1)

        elif self.mode == 'second_order':
            # 二阶退化（Real-ESRGAN 风格）
            # 第一阶退化
            kernel1 = self._random_gaussian_kernel()
            img = self._apply_blur(img, kernel1)

            # 随机下采样（模拟不同相机分辨率）
            if random.random() < 0.5:
                # 先下采样到一个中间分辨率
                h, w = img.shape[2:]
                intermediate_scale = random.uniform(0.5, 1.0)
                new_h, new_w = int(h * intermediate_scale), int(w * intermediate_scale)
                img = F.interpolate(img, size=(new_h, new_w), mode='bicubic', align_corners=False)

            # 第一阶噪声
            if random.random() < 0.7:
                noise_types = ['gaussian', 'poisson', 'speckle']
                img = self._add_noise(img, noise_type=random.choice(noise_types))

            # 第二阶退化
            if random.random() < 0.8:
                kernel2 = self._random_gaussian_kernel()
                img = self._apply_blur(img, kernel2)

            # JPEG 压缩
            if random.random() < 0.5:
                img = self._apply_jpeg(img)

            # 最终下采样
            img = self._downsample(img)

            # 最终噪声（传感器噪声）
            if random.random() < 0.5:
                img = self._add_noise(img, noise_type='gaussian',
                                      noise_level=random.uniform(0, 10) / 255.0)

            return torch.clamp(img, 0, 1)

        return self._downsample(img)


# ═══════════════════════════════════════════════════════════════════════════════
# 评估指标
# ═══════════════════════════════════════════════════════════════════════════════

def calc_psnr(img1, img2, max_val=1.0):
    """计算 PSNR"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse < 1e-10:
        return float('inf')
    return 20 * torch.log10(torch.tensor(max_val) / torch.sqrt(mse))


def calc_ssim(img1, img2, window_size=11):
    """计算 SSIM"""
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=window_size//2)
    mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=window_size//2)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.avg_pool2d(img1 ** 2, window_size, stride=1, padding=window_size//2) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 ** 2, window_size, stride=1, padding=window_size//2) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, window_size, stride=1, padding=window_size//2) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))

    return ssim_map.mean()


class ImageMetricsEvaluator:
    """图像指标评估器"""

    def __init__(self, device='cuda', calc_niqe=False):
        self.device = device
        self.calc_niqe = calc_niqe

        # LPIPS
        try:
            import lpips
            self.lpips_model = lpips.LPIPS(net='alex').to(device)
            self.has_lpips = True
        except ImportError:
            self.has_lpips = False

    def evaluate(self, pred, target):
        """评估所有指标"""
        pred = pred.to(self.device)
        target = target.to(self.device)

        # 尺寸对齐
        min_h = min(pred.shape[2], target.shape[2])
        min_w = min(pred.shape[3], target.shape[3])
        pred = pred[:, :, :min_h, :min_w]
        target = target[:, :, :min_h, :min_w]

        metrics = {}
        metrics['psnr'] = calc_psnr(pred, target).item()
        metrics['ssim'] = calc_ssim(pred, target).item()

        if self.has_lpips:
            pred_lpips = pred * 2 - 1
            target_lpips = target * 2 - 1
            with torch.no_grad():
                metrics['lpips'] = self.lpips_model(pred_lpips, target_lpips).mean().item()
        else:
            metrics['lpips'] = 0.0

        return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# EMA（指数移动平均）
# ═══════════════════════════════════════════════════════════════════════════════

class EMA:
    """
    指数移动平均
    用于在训练过程中维护一个平滑的模型副本，通常有更好的泛化性能
    """

    def __init__(self, model, decay=0.999):
        """
        Args:
            model: 要跟踪的模型
            decay: EMA 衰减率，越接近 1 更新越慢
        """
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # 初始化 shadow 参数
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        """更新 EMA 参数"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self, model):
        """应用 EMA 参数到模型"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name].clone()

    def restore(self, model):
        """恢复原始参数"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name].clone()
        self.backup = {}

    def state_dict(self):
        """返回 EMA 状态"""
        return {'shadow': self.shadow, 'decay': self.decay}

    def load_state_dict(self, state_dict):
        """加载 EMA 状态"""
        self.shadow = state_dict['shadow']
        self.decay = state_dict['decay']


# ═══════════════════════════════════════════════════════════════════════════════
# 训练辅助工具
# ═══════════════════════════════════════════════════════════════════════════════

class WarmupScheduler:
    """预热学习率调度器"""

    def __init__(self, optimizer, warmup_epochs, base_lr, target_lr=None):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr
        self.target_lr = target_lr or base_lr
        self.current_epoch = 0

    def step(self):
        """更新学习率"""
        if self.current_epoch < self.warmup_epochs:
            lr = self.base_lr + (self.target_lr - self.base_lr) * \
                 (self.current_epoch + 1) / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        self.current_epoch += 1

    def state_dict(self):
        return {'current_epoch': self.current_epoch}

    def load_state_dict(self, state_dict):
        self.current_epoch = state_dict['current_epoch']


class TwoStageTrainer:
    """
    两阶段训练管理器
    第一阶段：像素级精度（Charbonnier + SSIM）
    第二阶段：感知质量（NIQE友好 + LPIPS）
    """

    def __init__(self, loss_fn, stage1_epochs=200, transition_epochs=50):
        """
        Args:
            loss_fn: CompleteLoss 实例
            stage1_epochs: 第一阶段 epoch 数
            transition_epochs: 过渡阶段 epoch 数（线性过渡权重）
        """
        self.loss_fn = loss_fn
        self.stage1_epochs = stage1_epochs
        self.transition_epochs = transition_epochs
        self.current_epoch = 0
        self.stage = 'pixel'

    def step(self):
        """每 epoch 调用，自动管理阶段切换"""
        self.current_epoch += 1

        if self.current_epoch <= self.stage1_epochs:
            # 第一阶段：纯像素级
            self.stage = 'pixel'
            self.loss_fn.set_stage('pixel')
        elif self.current_epoch <= self.stage1_epochs + self.transition_epochs:
            # 过渡阶段：线性插值
            self.stage = 'transition'
            alpha = (self.current_epoch - self.stage1_epochs) / self.transition_epochs
            self.loss_fn.w_charbonnier = 1.0 * (1 - alpha) + 0.5 * alpha
            self.loss_fn.w_niqe = 0.3 * (1 - alpha) + 1.0 * alpha
            self.loss_fn.w_lpips = 0.0 * (1 - alpha) + 0.3 * alpha
            self.loss_fn.w_fft = 0.3 * (1 - alpha) + 0.5 * alpha
            self.loss_fn.w_edge = 0.2 * (1 - alpha) + 0.4 * alpha
        else:
            # 第二阶段：感知质量
            self.stage = 'perceptual'
            self.loss_fn.set_stage('perceptual')

        return self.stage

    def get_stage_info(self):
        """获取当前阶段信息"""
        return {
            'epoch': self.current_epoch,
            'stage': self.stage,
            'weights': {
                'charbonnier': self.loss_fn.w_charbonnier,
                'niqe': self.loss_fn.w_niqe,
                'lpips': self.loss_fn.w_lpips,
                'fft': self.loss_fn.w_fft,
                'edge': self.loss_fn.w_edge,
            }
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 数据增强函数
# ═══════════════════════════════════════════════════════════════════════════════

class DataAugmentation:
    """
    训练时数据增强
    支持随机翻转、旋转、颜色抖动等
    """

    def __init__(self, flip_prob=0.5, rotation_prob=0.3, color_jitter_prob=0.3,
                 scale_range=(0.8, 1.2)):
        self.flip_prob = flip_prob
        self.rotation_prob = rotation_prob
        self.color_jitter_prob = color_jitter_prob
        self.scale_range = scale_range

    def __call__(self, lr, hr):
        """
        对 LR-HR 对应用数据增强

        Args:
            lr: 低分辨率图像 [C, H, W]
            hr: 高分辨率图像 [C, H, W]

        Returns:
            aug_lr, aug_hr: 增强后的图像对
        """
        # 确保在同一设备上
        device = lr.device
        hr = hr.to(device)

        # 随机水平翻转
        if random.random() < self.flip_prob:
            lr = torch.flip(lr, dims=[2])
            hr = torch.flip(hr, dims=[2])

        # 随机垂直翻转
        if random.random() < self.flip_prob:
            lr = torch.flip(lr, dims=[1])
            hr = torch.flip(hr, dims=[1])

        # 随机旋转 90 度
        if random.random() < self.rotation_prob:
            k = random.choice([1, 2, 3])  # 90, 180, 270
            lr = torch.rot90(lr, k, dims=[1, 2])
            hr = torch.rot90(hr, k, dims=[1, 2])

        # 颜色抖动（仅对亮度/对比度，不改变颜色平衡）
        if random.random() < self.color_jitter_prob:
            # 亮度调整
            brightness_factor = random.uniform(0.8, 1.2)
            lr = torch.clamp(lr * brightness_factor, 0, 1)
            hr = torch.clamp(hr * brightness_factor, 0, 1)

            # 对比度调整
            contrast_factor = random.uniform(0.8, 1.2)
            lr_mean = lr.mean()
            lr = torch.clamp((lr - lr_mean) * contrast_factor + lr_mean, 0, 1)
            hr_mean = hr.mean()
            hr = torch.clamp((hr - hr_mean) * contrast_factor + hr_mean, 0, 1)

        # 随机缩放（保持 SR 比例）
        if random.random() < 0.3:
            scale = random.uniform(*self.scale_range)
            h, w = lr.shape[1:]
            new_h, new_w = int(h * scale), int(w * scale)
            if new_h >= 16 and new_w >= 16:  # 确保不会太小
                lr = F.interpolate(lr.unsqueeze(0), size=(new_h, new_w),
                                   mode='bicubic', align_corners=False).squeeze(0)
                hr = F.interpolate(hr.unsqueeze(0), size=(new_h, new_w),
                                   mode='bicubic', align_corners=False).squeeze(0)

        return lr, hr


# ═══════════════════════════════════════════════════════════════════════════════
# 检查点管理
# ═══════════════════════════════════════════════════════════════════════════════

class CheckpointManager:
    """增强版检查点管理器"""

    def __init__(self, checkpoint_dir, keep_last_n=5):
        self.checkpoint_dir = checkpoint_dir
        self.keep_last_n = keep_last_n
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save(self, state, is_best=False, filename='checkpoint.pth'):
        """保存检查点"""
        filepath = os.path.join(self.checkpoint_dir, filename)
        torch.save(state, filepath)

        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best.pth')
            torch.save(state, best_path)

        # 清理旧检查点
        self._cleanup()

    def load(self, filename='checkpoint.pth'):
        """加载检查点"""
        filepath = os.path.join(self.checkpoint_dir, filename)
        if os.path.exists(filepath):
            return torch.load(filepath, map_location='cpu')
        return None

    def _cleanup(self):
        """清理旧检查点"""
        epoch_files = sorted([
            f for f in os.listdir(self.checkpoint_dir)
            if f.startswith('epoch_') and f.endswith('.pth')
        ])
        while len(epoch_files) > self.keep_last_n:
            os.remove(os.path.join(self.checkpoint_dir, epoch_files.pop(0)))