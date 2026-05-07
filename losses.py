"""
精简损失函数 — 只保留经 NIQE 验证有效的组件

设计原则：
- 每个损失都有明确的 NIQE 优化目标
- 没有冗余（FFT 只计算一次）
- 组合权重可直接调参
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════════
# 1. MSCN 统计量损失 — 直接针对 NIQE 核心
# ═══════════════════════════════════════════════════════════════════════════════

class MSCNStatLoss(nn.Module):
    """
    MSCN (Mean Subtracted Contrast Normalized) 统计量损失

    NIQE 的核心是：计算图像 MSCN 系数的统计量（偏度、峰度、方差），
    并与自然场景统计模型比较。此损失直接匹配这些统计量。

    这是唯一一个"专门为 NIQE 设计"的损失，效果最直接。
    """

    def __init__(self, window_size=7):
        super().__init__()
        self.window_size = window_size

    def _compute_mscn(self, x):
        """计算 MSCN 系数 — 数值稳定版

        关键修复：
        1. sigma 的 eps 从 1e-6 提高到 1e-3，避免低方差区域除零
        2. 归一化后的系数 clamp 到 [-5, 5]，防止 pow(3)/pow(4) 溢出
           （原始 MSCN 系数在自然图像上 |值| 通常 < 3，
            超出 5 说明区域方差极低导致归一化异常，应被截断）
        """
        pad = self.window_size // 2
        mu = F.avg_pool2d(x, self.window_size, stride=1, padding=pad)
        sigma = torch.sqrt(
            F.avg_pool2d(x ** 2, self.window_size, stride=1, padding=pad)
            - mu ** 2 + 1e-3  # 修复：1e-6 → 1e-3，防止低方差区域 sigma 过小
        )
        mscn = (x - mu) / (sigma + 1e-2)  # 修复：1e-4 → 1e-2，更稳健的归一化
        mscn = torch.clamp(mscn, -5.0, 5.0)  # 修复：截断极端系数，防止 pow(3/4) 溢出
        return mscn

    def forward(self, pred, target):
        """
        Args:
            pred, target: [B, C, H, W] range [0, 1]
        Returns:
            loss: scalar
        """
        pred_mscn = self._compute_mscn(pred)
        target_mscn = self._compute_mscn(target)

        # 偏度 (Skewness) — 三阶矩，NIQE 最敏感
        pred_skew = pred_mscn.pow(3).mean(dim=[2, 3])
        target_skew = target_mscn.pow(3).mean(dim=[2, 3])
        skew_loss = F.mse_loss(pred_skew, target_skew)

        # 峰度 (Kurtosis) — 四阶矩，控制尾部分布
        pred_kurt = pred_mscn.pow(4).mean(dim=[2, 3])
        target_kurt = target_mscn.pow(4).mean(dim=[2, 3])
        kurt_loss = F.mse_loss(pred_kurt, target_kurt)

        # 方差 — 控制整体对比度
        pred_var = pred_mscn.var(dim=[2, 3])
        target_var = target_mscn.var(dim=[2, 3])
        var_loss = F.mse_loss(pred_var, target_var)

        return skew_loss + 0.5 * kurt_loss + 0.3 * var_loss


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Charbonnier 损失 — 平滑的 L1（减少伪影）
# ═══════════════════════════════════════════════════════════════════════════════

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss = sqrt((x-y)^2 + eps) — 比 L1 更平滑"""

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps))


# ═══════════════════════════════════════════════════════════════════════════════
# 3. 频域损失 — 匹配频域分布
# ═══════════════════════════════════════════════════════════════════════════════

class FFTLoss(nn.Module):
    """频域幅度损失 — NIQE 对频域分布敏感"""

    def __init__(self, loss_type='l1'):
        super().__init__()
        self.loss_type = loss_type

    def forward(self, pred, target):
        # FFT
        pred_fft = torch.fft.rfft2(pred, dim=(-2, -1))
        target_fft = torch.fft.rfft2(target, dim=(-2, -1))

        # 幅度（对数稳定）
        pred_amp = torch.log(torch.abs(pred_fft) + 1e-6)
        target_amp = torch.log(torch.abs(target_fft) + 1e-6)

        # 限制幅度范围，防止极端值
        pred_amp = torch.clamp(pred_amp, -20, 20)
        target_amp = torch.clamp(target_amp, -20, 20)

        if self.loss_type == 'l1':
            return F.l1_loss(pred_amp, target_amp)
        else:
            return F.mse_loss(pred_amp, target_amp)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. 边缘损失 — 保持锐利但不产生伪影
# ═══════════════════════════════════════════════════════════════════════════════

class EdgeLoss(nn.Module):
    """Sobel 边缘损失 — 使用 Charbonnier 形式"""

    def __init__(self):
        super().__init__()
        # Sobel 核
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def _gradients(self, img):
        """计算 Sobel 梯度幅度"""
        b, c, h, w = img.shape
        grad_x = F.conv2d(img.view(b * c, 1, h, w), self.sobel_x.to(img.device),
                          padding=1).view(b, c, h, w)
        grad_y = F.conv2d(img.view(b * c, 1, h, w), self.sobel_y.to(img.device),
                          padding=1).view(b, c, h, w)
        return torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)

    def forward(self, pred, target):
        pred_grad = self._gradients(pred)
        target_grad = self._gradients(target)
        # Charbonnier 形式（平滑）
        return torch.mean(torch.sqrt((pred_grad - target_grad) ** 2 + 1e-6))


# ═══════════════════════════════════════════════════════════════════════════════
# 5. SSIM 损失（单尺度）
# ═══════════════════════════════════════════════════════════════════════════════

class SSIMLoss(nn.Module):
    """1 - SSIM 作为损失"""

    def __init__(self, window_size=11):
        super().__init__()
        self.window_size = window_size
        self.c1 = 0.01 ** 2
        self.c2 = 0.03 ** 2

    def forward(self, pred, target):
        mu1 = F.avg_pool2d(pred, self.window_size, stride=1,
                           padding=self.window_size // 2)
        mu2 = F.avg_pool2d(target, self.window_size, stride=1,
                           padding=self.window_size // 2)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.avg_pool2d(pred ** 2, self.window_size, stride=1,
                                  padding=self.window_size // 2) - mu1_sq
        sigma2_sq = F.avg_pool2d(target ** 2, self.window_size, stride=1,
                                  padding=self.window_size // 2) - mu2_sq
        sigma12 = F.avg_pool2d(pred * target, self.window_size, stride=1,
                                padding=self.window_size // 2) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + self.c1) * (2 * sigma12 + self.c2)) / \
                   ((mu1_sq + mu2_sq + self.c1) * (sigma1_sq + sigma2_sq + self.c2))

        return 1 - ssim_map.mean()


# ═══════════════════════════════════════════════════════════════════════════════
# 组合损失 — 简洁，无冗余
# ═══════════════════════════════════════════════════════════════════════════════

class CompleteLoss(nn.Module):
    """
    组合损失 — 简洁版本

    组成：
    - Charbonnier（像素级基础）
    - MSCNStatLoss（直接针对 NIQE）
    - FFTLoss（频域匹配）
    - EdgeLoss（边缘锐利）
    - SSIMLoss（结构相似）

    没有冗余：FFT 只计算一次，不在其他损失中重复。
    """

    def __init__(self, stage='pixel', device='cuda', **kwargs):
        super().__init__()
        self.stage = stage
        self.device = device

        # 子损失
        self.charbonnier = CharbonnierLoss()
        self.mscn = MSCNStatLoss()
        self.fft = FFTLoss()
        self.edge = EdgeLoss()
        self.ssim = SSIMLoss()

        # LPIPS（可选）
        try:
            import lpips
            self.lpips = lpips.LPIPS(net='alex').to(device)
            self.has_lpips = True
        except ImportError:
            self.has_lpips = False

        self._set_weights(stage, kwargs)

    def _set_weights(self, stage, kwargs):
        """根据阶段设置权重

        关键修复：pixel 阶段禁用 MSCN（w_mscn=0），因为训练初期
        模型输出不稳定，MSCN 的 (x-mu)/sigma + pow(3/4) 极易产生 NaN。
        等 transition 阶段模型输出稳定后再逐步引入 MSCN。
        """
        if stage == 'pixel':
            self.w_charb = kwargs.get('w_charb', 1.0)
            self.w_mscn = kwargs.get('w_mscn', 0.0)   # pixel 阶段禁用 MSCN（NaN 风险）
            self.w_fft = kwargs.get('w_fft', 0.5)     # v3: 0.2→0.5，强化频域监督
            self.w_edge = kwargs.get('w_edge', 0.2)   # v3: 0.1→0.2，更多边缘梯度
            self.w_ssim = kwargs.get('w_ssim', 0.3)
            self.w_lpips = kwargs.get('w_lpips', 0.05)  # v3: 保留用户 LPIPS 权重
        elif stage == 'perceptual':
            self.w_charb = kwargs.get('w_charb', 0.3)   # v3: 0.5→0.3，pixel 让位
            self.w_mscn = kwargs.get('w_mscn', 2.0)     # v3: 1.0→2.0，强化 MSCN 统计量匹配
            self.w_fft = kwargs.get('w_fft', 1.0)       # v3: 0.5→1.0，强化频域匹配
            self.w_edge = kwargs.get('w_edge', 0.5)     # v3: 0.3→0.5，更多边缘梯度
            self.w_ssim = kwargs.get('w_ssim', 0.3)
            self.w_lpips = kwargs.get('w_lpips', 0.3)   # v3: 0.2→0.3，强化感知
        else:
            # 自定义
            self.w_charb = kwargs.get('w_charb', 1.0)
            self.w_mscn = kwargs.get('w_mscn', 1.0)     # v3: 0.5→1.0
            self.w_fft = kwargs.get('w_fft', 0.5)       # v3: 0.3→0.5
            self.w_edge = kwargs.get('w_edge', 0.3)     # v3: 0.2→0.3
            self.w_ssim = kwargs.get('w_ssim', 0.3)
            self.w_lpips = kwargs.get('w_lpips', 0.2)   # v3: 0.1→0.2

    def forward(self, pred, target):
        """
        Returns:
            total_loss, loss_dict
        """
        losses = {}

        # Charbonnier
        if self.w_charb > 0:
            losses['charb'] = self.charbonnier(pred, target)
        else:
            losses['charb'] = torch.tensor(0.0, device=self.device)

        # MSCN（直接针对 NIQE）
        if self.w_mscn > 0:
            losses['mscn'] = self.mscn(pred, target)
        else:
            losses['mscn'] = torch.tensor(0.0, device=self.device)

        # FFT
        if self.w_fft > 0:
            losses['fft'] = self.fft(pred, target)
        else:
            losses['fft'] = torch.tensor(0.0, device=self.device)

        # Edge
        if self.w_edge > 0:
            losses['edge'] = self.edge(pred, target)
        else:
            losses['edge'] = torch.tensor(0.0, device=self.device)

        # SSIM
        if self.w_ssim > 0:
            losses['ssim'] = self.ssim(pred, target)
        else:
            losses['ssim'] = torch.tensor(0.0, device=self.device)

        # LPIPS（双重保障：即使 pred 没被 clamp，这里也强制 clamp）
        if self.w_lpips > 0 and self.has_lpips:
            pred_safe = pred.clamp(0.0, 1.0)
            losses['lpips'] = self.lpips(pred_safe * 2 - 1, target * 2 - 1).mean()
        else:
            losses['lpips'] = torch.tensor(0.0, device=self.device)

        # 加权求和
        total = (
            self.w_charb * losses['charb'] +
            self.w_mscn * losses['mscn'] +
            self.w_fft * losses['fft'] +
            self.w_edge * losses['edge'] +
            self.w_ssim * losses['ssim'] +
            self.w_lpips * losses['lpips']
        )

        losses['total'] = total
        return total, losses

    def set_stage(self, stage, **kwargs):
        """切换训练阶段"""
        self.stage = stage
        self._set_weights(stage, kwargs)

    def update_weights(self, **weights):
        """动态更新权重 — 用于 transition 阶段和命令行覆盖

        Args:
            weights: 任意权重名和值，如 w_charb=0.8, w_mscn=0.5
        """
        for name, value in weights.items():
            if hasattr(self, name) and name.startswith('w_'):
                setattr(self, name, value)

    def get_weights(self):
        """返回当前所有权重字典"""
        return {
            'w_charb': self.w_charb,
            'w_mscn': self.w_mscn,
            'w_fft': self.w_fft,
            'w_edge': self.w_edge,
            'w_ssim': self.w_ssim,
            'w_lpips': self.w_lpips,
        }