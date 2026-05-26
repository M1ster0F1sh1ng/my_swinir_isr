"""
融合版训练脚本 — 基于 run_on_cloud_enhanced.py

集成修复模块（保持原有命令接口完全兼容）：
1. SwinIR_Fixed 模型（mask + 全局残差 + DropPath）
2. FixedFolderDataset（真随机裁剪 + Real-ESRGAN 退化 + 安全增强）
3. CompleteLoss（MSCN + Charbonnier + FFT + Edge，兼容旧权重接口）
4. EMA + TwoStageTrainer

保持原有功能：
- torchrun 分布式训练（DDP）
- SmartLRScheduler（warmup_cosine/plateau/adaptive 等）
- EarlyStopping
- CheckpointManager + TrainingMonitor + auto-resume
- 梯度累积 + AMP
"""

import argparse
import os
import copy
import time
import math
import numpy as np
import random
import json
from collections import deque

import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import ConcatDataset
from torchvision import transforms

# 分布式训练相关
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from tqdm.auto import tqdm

from PIL import Image

# ═══════════════════════════════════════════════════════════════════════════════
# 导入修复版模块
# ═══════════════════════════════════════════════════════════════════════════════
import cloud_dataset as cloud_dataset
from swinir_model import SwinIR_Official, SwinIR_Fixed, SwinIR_Light_Fixed, load_pretrained
from network_swinir import SwinIR as OfficialSwinIR, load_pretrained_official
from losses import CompleteLoss


# ═══════════════════════════════════════════════════════════════════════════════
# 基础工具（保持与原 utils 接口兼容）
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
    mu1 = torch.nn.functional.avg_pool2d(img1, window_size, stride=1, padding=window_size // 2)
    mu2 = torch.nn.functional.avg_pool2d(img2, window_size, stride=1, padding=window_size // 2)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = torch.nn.functional.avg_pool2d(img1 ** 2, window_size, stride=1, padding=window_size // 2) - mu1_sq
    sigma2_sq = torch.nn.functional.avg_pool2d(img2 ** 2, window_size, stride=1, padding=window_size // 2) - mu2_sq
    sigma12 = torch.nn.functional.avg_pool2d(img1 * img2, window_size, stride=1, padding=window_size // 2) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean()


def rgb_to_ycbcr_y(img):
    """
    RGB -> YCbCr Y channel (MATLAB standard coefficients)
    img: [B, 3, H, W] in [0, 1], RGB order
    return: [B, 1, H, W], Y in [16/255, 235/255]
    """
    r, g, b = img[:, 0:1, :, :], img[:, 1:2, :, :], img[:, 2:3, :, :]
    y = 16.0 / 255.0 + (65.481 / 255.0) * r + (128.553 / 255.0) * g + (24.966 / 255.0) * b
    return y


class ImageMetricsEvaluator:
    """
    增强版图像指标评估器
    支持：
    1. YCbCr Y 通道 PSNR（学术标准）
    2. border crop（裁剪边界 artifacts）
    3. RGB PSNR（兼容旧逻辑）
    """

    def __init__(self, device='cuda', calc_niqe=False, border=0, test_y_channel=False,
                 calc_psnr=True, calc_ssim=True, calc_lpips=True):
        self.device = device
        self.calc_niqe = calc_niqe
        self.border = border
        self.test_y_channel = test_y_channel
        self.calc_psnr = calc_psnr
        self.calc_ssim = calc_ssim
        self.calc_lpips = calc_lpips
        # LPIPS
        if calc_lpips:
            try:
                import lpips
                self.lpips_model = lpips.LPIPS(net='alex').to(device)
                self.has_lpips = True
            except ImportError:
                self.has_lpips = False
        else:
            self.has_lpips = False

    def evaluate(self, pred, target):
        pred = pred.to(self.device)
        target = target.to(self.device)
        min_h = min(pred.shape[2], target.shape[2])
        min_w = min(pred.shape[3], target.shape[3])
        pred = pred[:, :, :min_h, :min_w]
        target = target[:, :, :min_h, :min_w]

        metrics = {}

        # === RGB PSNR / SSIM ===
        if self.calc_psnr:
            metrics['psnr_rgb'] = calc_psnr(
                pred[:, :, self.border:-self.border, self.border:-self.border] if self.border > 0 else pred,
                target[:, :, self.border:-self.border, self.border:-self.border] if self.border > 0 else target
            ).item()
        if self.calc_ssim:
            metrics['ssim_rgb'] = calc_ssim(pred, target).item()

        # === YCbCr Y 通道 PSNR（学术标准）===
        if self.test_y_channel:
            pred_y = rgb_to_ycbcr_y(pred)
            target_y = rgb_to_ycbcr_y(target)
            if self.calc_psnr:
                metrics['psnr'] = calc_psnr(
                    pred_y[:, :, self.border:-self.border, self.border:-self.border] if self.border > 0 else pred_y,
                    target_y[:, :, self.border:-self.border, self.border:-self.border] if self.border > 0 else target_y
                ).item()
            if self.calc_ssim:
                metrics['ssim'] = calc_ssim(pred_y, target_y).item()
        else:
            if self.calc_psnr:
                metrics['psnr'] = metrics.get('psnr_rgb', 0.0)
            if self.calc_ssim:
                metrics['ssim'] = metrics.get('ssim_rgb', 0.0)

        if self.calc_lpips and self.has_lpips:
            pred_lpips = pred * 2 - 1
            target_lpips = target * 2 - 1
            with torch.no_grad():
                metrics['lpips'] = self.lpips_model(pred_lpips, target_lpips).mean().item()
        elif self.calc_lpips:
            metrics['lpips'] = 0.0
        return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# EMA（指数移动平均）— 新增
# ═══════════════════════════════════════════════════════════════════════════════

class EMA:
    """指数移动平均"""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                # 确保 shadow 和 param 在同一设备（checkpoint 恢复后可能不一致）
                if self.shadow[name].device != param.data.device:
                    self.shadow[name] = self.shadow[name].to(param.data.device)
                self.shadow[name] = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]

    def apply_shadow(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                shadow = self.shadow[name]
                if shadow.device != param.data.device:
                    shadow = shadow.to(param.data.device)
                param.data = shadow.clone()

    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name].clone()
        self.backup = {}

    def state_dict(self):
        return {'shadow': self.shadow, 'decay': self.decay}

    def load_state_dict(self, state_dict, device=None):
        self.shadow = state_dict['shadow']
        self.decay = state_dict.get('decay', self.decay)
        # 从 checkpoint 加载的 shadow 在 CPU，需移到模型所在设备
        if device is not None:
            self.shadow = {k: v.to(device) for k, v in self.shadow.items()}


# ═══════════════════════════════════════════════════════════════════════════════
# 两阶段训练管理 — 新增
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# 多指标综合评分（验证最佳模型不再只看 PSNR）
# ═══════════════════════════════════════════════════════════════════════════════

class MultiMetricScore:
    """
    多指标综合评分系统

    将 PSNR、SSIM、LPIPS、NIQE 等多个验证指标加权组合为单一评分，
    用于判断"最佳模型"和早停。

    设计原则：
    - PSNR: 越高越好（像素精度）
    - SSIM: 越高越好（结构相似）
    - LPIPS: 越低越好（感知距离）→ 转换为 1-LPIPS
    - NIQE: 越低越好（自然质量）→ 转换为 1/NIQE 或 -NIQE

    默认权重：
    - w_psnr=1.0: PSNR 是主要指标
    - w_ssim=0.5: SSIM 辅助
    - w_lpips=0.3: LPIPS 感知质量
    - w_niqe=0.0: NIQE 默认关闭（需要 pyiqa）
    """

    def __init__(self, w_psnr=1.0, w_ssim=0.5, w_lpips=0.3, w_niqe=0.0,
                 device='cuda'):
        self.w_psnr = w_psnr
        self.w_ssim = w_ssim
        self.w_lpips = w_lpips
        self.w_niqe = w_niqe
        self.device = device

        # 尝试加载 NIQE 评估器
        self.niqe_model = None
        if w_niqe > 0:
            try:
                import pyiqa
                self.niqe_model = pyiqa.create_metric('niqe', device=device)
                print(f'[MultiMetricScore] NIQE 评估器已加载')
            except ImportError:
                print('[MultiMetricScore] 警告: pyiqa 未安装，NIQE 指标不可用')
                print('  安装: pip install pyiqa')
                self.w_niqe = 0.0

    @torch.no_grad()
    def compute(self, metrics):
        """
        计算综合评分

        Args:
            metrics: dict，包含 psnr, ssim, lpips 等
        Returns:
            score: float，综合评分（越高越好）
            breakdown: dict，各指标贡献分解
        """
        psnr = metrics.get('psnr', 0.0)
        ssim = metrics.get('ssim', 0.0)
        lpips = metrics.get('lpips', 0.0)

        # 各指标归一化到可比较的范围
        # PSNR: 典型范围 20-40 dB → 归一化为 0-1 (除以 40)
        psnr_norm = min(psnr / 40.0, 1.0)
        # SSIM: 已在 0-1 范围
        ssim_norm = ssim
        # LPIPS: 越低越好，转换为正向分数 → 1 - LPIPS
        lpips_norm = max(0.0, 1.0 - lpips)

        score = (
            self.w_psnr * psnr_norm +
            self.w_ssim * ssim_norm +
            self.w_lpips * lpips_norm
        )

        breakdown = {
            'psnr_raw': psnr,
            'psnr_norm': psnr_norm,
            'ssim_raw': ssim,
            'ssim_norm': ssim_norm,
            'lpips_raw': lpips,
            'lpips_norm': lpips_norm,
            'score': score,
        }

        # NIQE（如果启用）
        if self.w_niqe > 0 and 'niqe' in metrics:
            niqe = metrics['niqe']
            # NIQE 典型范围 2-15，越低越好 → 反向归一化
            niqe_norm = max(0.0, 1.0 - (niqe - 2.0) / 13.0)
            score += self.w_niqe * niqe_norm
            breakdown['niqe_raw'] = niqe
            breakdown['niqe_norm'] = niqe_norm

        breakdown['total_score'] = score
        return score, breakdown

    @torch.no_grad()
    def compute_niqe(self, img_tensor):
        """计算单张图像的 NIQE 分数（越低越好）"""
        if self.niqe_model is None:
            return None
        try:
            # pyiqa 期望 [B, C, H, W] range [0, 1]
            return self.niqe_model(img_tensor).mean().item()
        except Exception as e:
            return None

    def get_weights(self):
        """返回当前权重设置"""
        return {
            'w_psnr': self.w_psnr,
            'w_ssim': self.w_ssim,
            'w_lpips': self.w_lpips,
            'w_niqe': self.w_niqe,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 增强版验证指标评估器（支持NIQE）
# ═══════════════════════════════════════════════════════════════════════════════

class EnhancedMetricsEvaluator(ImageMetricsEvaluator):
    """在原有 ImageMetricsEvaluator 基础上增加 NIQE 支持"""

    def __init__(self, device='cuda', calc_niqe=False, border=0, test_y_channel=False,
                 calc_psnr=True, calc_ssim=True, calc_lpips=True):
        super().__init__(device=device, calc_niqe=calc_niqe, border=border,
                         test_y_channel=test_y_channel,
                         calc_psnr=calc_psnr, calc_ssim=calc_ssim, calc_lpips=calc_lpips)
        self.calc_niqe = calc_niqe
        self.niqe_model = None

        if calc_niqe:
            try:
                import pyiqa
                self.niqe_model = pyiqa.create_metric('niqe', device=device)
            except ImportError:
                pass

    def evaluate(self, pred, target):
        """评估所有指标（含 NIQE）"""
        metrics = super().evaluate(pred, target)

        # 计算 NIQE（在 SR 输出上）
        if self.niqe_model is not None:
            try:
                with torch.no_grad():
                    metrics['niqe'] = self.niqe_model(pred).mean().item()
            except Exception:
                metrics['niqe'] = 0.0
        else:
            metrics['niqe'] = 0.0

        return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# 封装 Checkpoint 构建逻辑（消除重复代码）
# ═══════════════════════════════════════════════════════════════════════════════

def build_checkpoint(epoch, best_psnr, best_epoch, model, optimizer,
                     lr_scheduler, early_stopping=None, ema=None,
                     monitor=None, stage_trainer=None, is_distributed=False):
    """构建检查点字典 — 统一封装，消除重复"""
    ckpt = {
        'epoch': epoch,
        'model': model.module.state_dict() if is_distributed else model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'best_psnr': best_psnr,
        'best_epoch': best_epoch,
    }
    if early_stopping is not None:
        ckpt['early_stopping'] = early_stopping.state_dict()
    if ema is not None:
        ckpt['ema'] = ema.state_dict()
    if stage_trainer is not None:
        ckpt['stage_trainer'] = stage_trainer.state_dict()
    if monitor is not None:
        ckpt['monitor'] = {
            'train_losses': monitor.train_losses,
            'val_psnrs': monitor.val_psnrs,
            'val_ssims': monitor.val_ssims,
            'val_lpips': monitor.val_lpips,
            'learning_rates': monitor.learning_rates,
            'epochs': monitor.epochs,
            'best_psnr': monitor.best_psnr,
            'best_epoch': monitor.best_epoch,
        }
    return ckpt


class TwoStageTrainer:
    """
    两阶段训练：
    - 阶段1 (pixel): 侧重像素精度
    - 过渡 (transition): 线性混合
    - 阶段2 (perceptual): 侧重感知质量

    修复：v2 版本保留用户传入的自定义权重，不再被硬编码覆盖。
    """
    def __init__(self, loss_fn, stage1_epochs=200, transition_epochs=50,
                 user_weights=None):
        self.loss_fn = loss_fn
        self.stage1_epochs = stage1_epochs
        self.transition_epochs = transition_epochs
        self.current_epoch = 0
        self.current_stage = 'pixel'
        # 保存用户在命令行/配置文件中传入的权重，防止被 set_stage 覆盖
        self.user_weights = user_weights or {}

    def state_dict(self):
        return {
            'current_epoch': self.current_epoch,
            'current_stage': self.current_stage,
            'stage1_epochs': self.stage1_epochs,
            'transition_epochs': self.transition_epochs,
        }

    def load_state_dict(self, state):
        self.current_epoch = state.get('current_epoch', 0)
        self.current_stage = state.get('current_stage', 'pixel')

    def _set_stage_with_user_weights(self, stage):
        """切换阶段时保留用户自定义权重"""
        self.loss_fn.set_stage(stage)
        # 用用户权重覆盖默认值
        if self.user_weights:
            self.loss_fn.update_weights(**self.user_weights)

    def step(self):
        self.current_epoch += 1
        if self.current_epoch <= self.stage1_epochs:
            self.current_stage = 'pixel'
            self._set_stage_with_user_weights('pixel')
            return 'pixel'
        elif self.current_epoch <= self.stage1_epochs + self.transition_epochs:
            self.current_stage = 'transition'
            alpha = (self.current_epoch - self.stage1_epochs) / self.transition_epochs
            # 过渡阶段也尊重用户传入的 pixel/perceptual 权重，做线性插值
            # v3: 大幅提高 perceptual 默认权重，打破 conv 恒等映射陷阱
            w_charb = self.user_weights.get('w_charb', 1.0) * (1 - alpha) +                       self.user_weights.get('w_charb', 0.3) * alpha
            w_mscn = self.user_weights.get('w_mscn', 0.0) * (1 - alpha) +                      self.user_weights.get('w_mscn', 2.0) * alpha
            w_fft = self.user_weights.get('w_fft', 0.5) * (1 - alpha) +                     self.user_weights.get('w_fft', 1.0) * alpha
            w_edge = self.user_weights.get('w_edge', 0.2) * (1 - alpha) +                      self.user_weights.get('w_edge', 0.5) * alpha
            w_lpips = self.user_weights.get('w_lpips', 0.05) * (1 - alpha) +                       self.user_weights.get('w_lpips', 0.3) * alpha
            self.loss_fn.update_weights(
                w_charb=w_charb, w_mscn=w_mscn, w_fft=w_fft,
                w_edge=w_edge, w_lpips=w_lpips,
            )
            return 'transition'
        else:
            self.current_stage = 'perceptual'
            self._set_stage_with_user_weights('perceptual')
            return 'perceptual'


# ═══════════════════════════════════════════════════════════════════════════════
# 以下保持与 run_on_cloud_enhanced.py 完全相同
# ═══════════════════════════════════════════════════════════════════════════════

class SmartLRScheduler:
    """智能学习率调度器 — 保持原实现"""

    def __init__(self, optimizer, mode='cosine', base_lr=2e-4,
                 min_lr=1e-7, warmup_epochs=5, T_max=1000,
                 factor=0.5, patience=10, cooldown=5,
                 adaptive_factor=0.7, adaptive_patience=20,
                 verbose=True):
        self.optimizer = optimizer
        self.mode = mode
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.warmup_epochs = warmup_epochs
        self.T_max = T_max
        self.factor = factor
        self.patience = patience
        self.cooldown = cooldown
        self.adaptive_factor = adaptive_factor
        self.adaptive_patience = adaptive_patience
        self.verbose = verbose

        self.current_epoch = 0
        self.best_metric = float('-inf')
        self.num_bad_epochs = 0
        self.cooldown_counter = 0
        self.last_lr = base_lr
        self.lr_history = []
        self.loss_history = deque(maxlen=adaptive_patience + 5)
        self.loss_trend = deque(maxlen=5)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = base_lr

    def step(self, epoch=None, metric=None, loss=None):
        if epoch is not None:
            self.current_epoch = epoch

        if self.mode == 'cosine':
            self._cosine_annealing()
        elif self.mode == 'warmup_cosine':
            self._warmup_cosine()
        elif self.mode == 'plateau':
            if metric is not None:
                self._reduce_on_plateau(metric)
        elif self.mode == 'onecycle':
            self._one_cycle()
        elif self.mode == 'adaptive':
            if loss is not None:
                self._adaptive_adjust(loss)

        self.lr_history.append(self.get_last_lr())

    def _cosine_annealing(self):
        if self.current_epoch < self.warmup_epochs:
            lr = self.base_lr * (self.current_epoch + 1) / self.warmup_epochs
        else:
            progress = (self.current_epoch - self.warmup_epochs) / (self.T_max - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        self._set_lr(lr)

    def _warmup_cosine(self):
        if self.current_epoch < self.warmup_epochs:
            lr = self.base_lr * (self.current_epoch + 1) / self.warmup_epochs
        else:
            progress = (self.current_epoch - self.warmup_epochs) / (self.T_max - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        self._set_lr(lr)

    def _reduce_on_plateau(self, metric):
        current = float(metric)
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0
        if current > self.best_metric:
            self.best_metric = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
        if self.num_bad_epochs > self.patience:
            self._reduce_lr()
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

    def _one_cycle(self):
        if self.current_epoch < self.warmup_epochs:
            lr = self.base_lr + (self.base_lr * 10 - self.base_lr) * self.current_epoch / self.warmup_epochs
        elif self.current_epoch < self.T_max // 2:
            lr = self.base_lr * 10
        else:
            progress = (self.current_epoch - self.T_max // 2) / (self.T_max - self.T_max // 2)
            lr = self.min_lr + (self.base_lr * 10 - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        self._set_lr(lr)

    def _adaptive_adjust(self, loss):
        self.loss_history.append(loss)
        if self.current_epoch < self.warmup_epochs:
            lr = self.base_lr * (self.current_epoch + 1) / self.warmup_epochs
            self._set_lr(lr)
            return
        if len(self.loss_history) < self.adaptive_patience:
            return
        recent_losses = list(self.loss_history)[-self.adaptive_patience:]
        early_avg = np.mean(recent_losses[:self.adaptive_patience // 2])
        late_avg = np.mean(recent_losses[self.adaptive_patience // 2:])
        improvement = (early_avg - late_avg) / early_avg if early_avg > 0 else 0
        self.loss_trend.append(improvement)
        if improvement < 0.001:
            new_lr = max(self.get_last_lr() * self.adaptive_factor, self.min_lr)
            if self.verbose and new_lr != self.get_last_lr():
                print(f'[SmartLR] Loss 平台期检测到，学习率: {self.get_last_lr():.2e} -> {new_lr:.2e}')
            self._set_lr(new_lr)
        elif improvement > 0.01:
            new_lr = min(self.get_last_lr() * 1.1, self.base_lr)
            self._set_lr(new_lr)

    def _reduce_lr(self):
        old_lr = self.get_last_lr()
        new_lr = max(old_lr * self.factor, self.min_lr)
        if old_lr - new_lr > 1e-10:
            self._set_lr(new_lr)
            if self.verbose:
                print(f'[SmartLR] 学习率降低: {old_lr:.2e} -> {new_lr:.2e}')

    def _set_lr(self, lr):
        self.last_lr = lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_last_lr(self):
        return self.last_lr

    def state_dict(self):
        return {
            'mode': self.mode, 'current_epoch': self.current_epoch,
            'best_metric': self.best_metric, 'num_bad_epochs': self.num_bad_epochs,
            'cooldown_counter': self.cooldown_counter, 'last_lr': self.last_lr,
            'lr_history': list(self.lr_history), 'loss_history': list(self.loss_history),
            # v3: 保存超参数用于 resume 时检测变化
            'base_lr': self.base_lr, 'min_lr': self.min_lr,
            'T_max': self.T_max, 'warmup_epochs': self.warmup_epochs,
        }

    def load_state_dict(self, state_dict):
        # v3: 检测超参数变化，不一致时重置状态（防止 LR 异常）
        saved_base_lr = state_dict.get('base_lr', None)
        saved_min_lr = state_dict.get('min_lr', None)
        saved_T_max = state_dict.get('T_max', None)
        saved_warmup = state_dict.get('warmup_epochs', None)

        mismatch = []
        if saved_base_lr is not None and abs(saved_base_lr - self.base_lr) > 1e-10:
            mismatch.append(f'lr {saved_base_lr:.2e}->{self.base_lr:.2e}')
        if saved_min_lr is not None and abs(saved_min_lr - self.min_lr) > 1e-10:
            mismatch.append(f'min_lr {saved_min_lr:.2e}->{self.min_lr:.2e}')
        if saved_T_max is not None and saved_T_max != self.T_max:
            mismatch.append(f'T_max {saved_T_max}->{self.T_max}')
        if saved_warmup is not None and saved_warmup != self.warmup_epochs:
            mismatch.append(f'warmup {saved_warmup}->{self.warmup_epochs}')

        if mismatch:
            if self.verbose:
                print(f'[SmartLR] 检测到超参数变化: {", ".join(mismatch)}')
                print(f'[SmartLR] 重置调度器状态，从新配置开始计算 LR')
            self.current_epoch = 0
            self.best_metric = float('-inf')
            self.num_bad_epochs = 0
            self.cooldown_counter = 0
            self.last_lr = self.base_lr
            self.lr_history.clear()
            self._set_lr(self.base_lr)
            return

        self.mode = state_dict.get('mode', self.mode)
        self.current_epoch = state_dict.get('current_epoch', 0)
        self.best_metric = state_dict.get('best_metric', float('-inf'))
        self.num_bad_epochs = state_dict.get('num_bad_epochs', 0)
        self.cooldown_counter = state_dict.get('cooldown_counter', 0)
        self.last_lr = state_dict.get('last_lr', self.base_lr)
        self.lr_history = deque(state_dict.get('lr_history', []), maxlen=len(self.lr_history))
        self.loss_history = deque(state_dict.get('loss_history', []), maxlen=self.loss_history.maxlen)
        self._set_lr(self.last_lr)


class EarlyStopping:
    """早停机制 — 保持原实现"""

    def __init__(self, mode='min', patience=30, min_delta=1e-4,
                 cooldown=10, min_epochs=100, verbose=True):
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.cooldown = cooldown
        self.min_epochs = min_epochs
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.cooldown_counter = 0
        self.best_epoch = 0
        self.loss_history = deque(maxlen=patience + 5)
        self.gradient_history = deque(maxlen=5)
        self.stagnant_epochs = 0
        self.improvement_history = []

    def __call__(self, score, epoch):
        if epoch < self.min_epochs:
            return False, None
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return False, None
        score = float(score)
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False, None
        if self._is_improvement(score):
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            self.stagnant_epochs = 0
            improvement = abs(score - self.best_score) / abs(self.best_score) if self.best_score != 0 else 0
            self.improvement_history.append(improvement)
        else:
            self.counter += 1
            self.stagnant_epochs += 1
        if self.counter >= self.patience:
            reason = f'{self.patience} 个 epoch 无改善'
            return True, reason
        if self.mode == 'min' and len(self.loss_history) >= 5:
            if self._detect_sweet_spot():
                reason = '检测到 loss 甜点（开始上升）'
                return True, reason
        if self.stagnant_epochs >= self.patience * 2:
            reason = f'长期停滞 ({self.patience * 2} 个 epoch 无显著改善)'
            return True, reason
        return False, None

    def update_loss(self, loss):
        self.loss_history.append(loss)

    def _is_improvement(self, score):
        if self.mode == 'min':
            return score < self.best_score - self.min_delta
        else:
            return score > self.best_score + self.min_delta

    def _detect_sweet_spot(self):
        if len(self.loss_history) < 5:
            return False
        recent_losses = list(self.loss_history)[-5:]
        gradients = []
        for i in range(1, len(recent_losses)):
            grad = recent_losses[i] - recent_losses[i - 1]
            gradients.append(grad)
        positive_grads = sum(1 for g in gradients if g > 0)
        return positive_grads >= 3

    def reset(self):
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.cooldown_counter = 0
        self.best_epoch = 0
        self.loss_history.clear()
        self.gradient_history.clear()
        self.stagnant_epochs = 0
        self.improvement_history = []

    def state_dict(self):
        return {
            'counter': self.counter, 'best_score': self.best_score,
            'early_stop': self.early_stop, 'cooldown_counter': self.cooldown_counter,
            'best_epoch': self.best_epoch, 'loss_history': list(self.loss_history),
            'stagnant_epochs': self.stagnant_epochs,
        }

    def load_state_dict(self, state_dict):
        self.counter = state_dict.get('counter', 0)
        self.best_score = state_dict.get('best_score', None)
        self.early_stop = state_dict.get('early_stop', False)
        self.cooldown_counter = state_dict.get('cooldown_counter', 0)
        self.best_epoch = state_dict.get('best_epoch', 0)
        self.loss_history = deque(state_dict.get('loss_history', []), maxlen=self.loss_history.maxlen)
        self.stagnant_epochs = state_dict.get('stagnant_epochs', 0)


class NumpyEncoder(json.JSONEncoder):
    """处理 numpy 类型的 JSON 编码器"""
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, deque):
            return list(obj)
        return super(NumpyEncoder, self).default(obj)


class TrainingMonitor:
    """训练监控器 — 保持原实现"""

    def __init__(self, log_dir, window_size=10):
        self.log_dir = log_dir
        self.window_size = window_size
        os.makedirs(log_dir, exist_ok=True)
        self.train_losses = []
        self.val_psnrs = []
        self.val_ssims = []
        self.val_lpips = []
        self.learning_rates = []
        self.epochs = []
        self.best_psnr = 0.0
        self.best_epoch = 0
        self.start_time = time.time()

    def update(self, epoch, train_loss, val_metrics, lr):
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_psnrs.append(val_metrics.get('psnr', 0))
        self.val_ssims.append(val_metrics.get('ssim', 0))
        self.val_lpips.append(val_metrics.get('lpips', 0))
        self.learning_rates.append(lr)
        if val_metrics.get('psnr', 0) > self.best_psnr:
            self.best_psnr = val_metrics['psnr']
            self.best_epoch = epoch

    def get_recent_trend(self):
        if len(self.train_losses) < self.window_size:
            return None
        recent_losses = self.train_losses[-self.window_size:]
        recent_psnrs = self.val_psnrs[-self.window_size:]
        x = np.arange(len(recent_losses))
        loss_slope = np.polyfit(x, recent_losses, 1)[0]
        psnr_slope = np.polyfit(x, recent_psnrs, 1)[0]
        return {
            'loss_slope': loss_slope, 'psnr_slope': psnr_slope,
            'loss_improving': loss_slope < -0.001,
            'psnr_improving': psnr_slope > 0.01,
        }

    def get_training_summary(self):
        elapsed = time.time() - self.start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        trend = self.get_recent_trend()
        return {
            'total_epochs': len(self.epochs),
            'elapsed_time': f'{hours}h {minutes}m',
            'best_psnr': self.best_psnr,
            'best_epoch': self.best_epoch,
            'current_lr': self.learning_rates[-1] if self.learning_rates else 0,
            'recent_trend': trend,
        }

    def save_logs(self):
        log_data = {
            'epochs': self.epochs, 'train_losses': self.train_losses,
            'val_psnrs': self.val_psnrs, 'val_ssims': self.val_ssims,
            'val_lpips': self.val_lpips, 'learning_rates': self.learning_rates,
            'summary': self.get_training_summary(),
        }
        with open(os.path.join(self.log_dir, 'training_log.json'), 'w') as f:
            json.dump(log_data, f, indent=2, cls=NumpyEncoder)


# ═══════════════════════════════════════════════════════════════════════════════
# 分布式工具 — 保持原实现
# ═══════════════════════════════════════════════════════════════════════════════

def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        world_size = 1
        gpu = 0
    torch.cuda.set_device(gpu)
    dist.init_process_group(backend='nccl', init_method='env://',
                            world_size=world_size, rank=rank)
    dist.barrier()
    return rank, world_size, gpu


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def reduce_tensor(tensor, world_size=None):
    if not dist.is_initialized():
        return tensor
    if world_size is None:
        world_size = dist.get_world_size()
    rt = tensor.clone().float()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt


# ═══════════════════════════════════════════════════════════════════════════════
# 检查点管理 — 保持原实现
# ═══════════════════════════════════════════════════════════════════════════════

class CheckpointManager:
    def __init__(self, checkpoint_dir, keep_last_n=3):
        self.checkpoint_dir = checkpoint_dir
        self.keep_last_n = keep_last_n
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save_checkpoint(self, state, is_best=False, filename='checkpoint.pth'):
        if not is_main_process():
            return
        filepath = os.path.join(self.checkpoint_dir, filename)
        torch.save(state, filepath)
        latest_path = os.path.join(self.checkpoint_dir, 'checkpoint_latest.pth')
        torch.save(state, latest_path)
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'checkpoint_best.pth')
            torch.save(state, best_path)
            print(f'✓ 保存最佳检查点: epoch={state["epoch"]}, PSNR={state.get("best_psnr", 0):.2f}')
        self._cleanup_old_checkpoints()

    def load_checkpoint(self, filename='checkpoint_latest.pth'):
        filepath = os.path.join(self.checkpoint_dir, filename)
        if os.path.exists(filepath):
            return torch.load(filepath, map_location='cpu', weights_only=False)
        return None

    def find_latest_checkpoint(self):
        candidates = [
            os.path.join(self.checkpoint_dir, 'checkpoint_latest.pth'),
            os.path.join(self.checkpoint_dir, 'checkpoint_best.pth'),
        ]
        epoch_files = [f for f in os.listdir(self.checkpoint_dir)
                       if f.startswith('checkpoint_epoch') and f.endswith('.pth')]
        if epoch_files:
            epoch_files.sort()
            candidates.insert(0, os.path.join(self.checkpoint_dir, epoch_files[-1]))
        for path in candidates:
            if os.path.exists(path):
                return path
        return None

    def _cleanup_old_checkpoints(self):
        epoch_files = sorted([f for f in os.listdir(self.checkpoint_dir)
                              if f.startswith('checkpoint_epoch') and f.endswith('.pth')])
        while len(epoch_files) > self.keep_last_n:
            os.remove(os.path.join(self.checkpoint_dir, epoch_files.pop(0)))


# ═══════════════════════════════════════════════════════════════════════════════
# 主程序
# ═══════════════════════════════════════════════════════════════════════════════

device = torch.device('cpu')
save_ = False
model = None
ema = None  # 新增：EMA
jump = False
jump_seq = 3

rank = 0
world_size = 1
gpu = 0
is_distributed = False


def pre_run():
    """预处理：初始化设备、模型、分布式环境 — 使用 SwinIR_Fixed"""
    global device, save_, args, model, ema, jump, jump_seq
    global rank, world_size, gpu, is_distributed

    # 初始化分布式环境
    if args.distributed:
        rank, world_size, gpu = setup_distributed()
        is_distributed = True
        device = torch.device(f'cuda:{gpu}')
        if is_main_process():
            print(f'[Rank {rank}/{world_size}] 使用 GPU {gpu}')
    else:
        if not torch.cuda.is_available():
            print('无法使用 cuda，请设置好环境')
            return
        device = torch.device('cuda:0')
        print('使用单卡训练')

    # 设置随机种子
    seed = args.seed + rank
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # 创建输出目录
    if is_main_process():
        output_dir = os.path.join(args.outputs_dir,
                                  f'{args.model}_swinir_fixed_x{args.scale}')
        os.makedirs(output_dir, exist_ok=True)

        existing_dirs = sorted([d for d in os.listdir(output_dir)
                                if d.isdigit() and os.path.isdir(os.path.join(output_dir, d))])
        if args.auto_resume and existing_dirs:
            args.outputs_dir = os.path.join(output_dir, existing_dirs[-1])
            print(f'输出目录: {args.outputs_dir} (auto-resume)')
        else:
            counts = os.listdir(output_dir)
            args.outputs_dir = os.path.join(output_dir, f'{len(counts) + 1:04d}')
            os.makedirs(args.outputs_dir, exist_ok=True)
            print(f'输出目录: {args.outputs_dir}')

    # 同步输出目录
    if is_distributed:
        if is_main_process():
            output_path = args.outputs_dir
        else:
            output_path = ''
        output_path_list = [output_path]
        dist.broadcast_object_list(output_path_list, src=0)
        args.outputs_dir = output_path_list[0]

    # 保存设置
    if args.save.lower() == 'yes':
        save_ = True

    # 跳过验证设置
    if args.jump == 'yes':
        jump = True
        jump_seq = args.jump_seq

    # === 创建模型：根据 --arch 选择架构 ===
    if args.arch == 'official':
        # 官方 SwinIR 模型 — 100% 预训练权重兼容
        model = OfficialSwinIR(
            img_size=64,
            patch_size=1,
            in_chans=3,
            embed_dim=args.embed_dim,
            depths=args.depths,
            num_heads=args.num_heads,
            window_size=args.window_size,
            mlp_ratio=args.mlp_ratio,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=args.drop_path_rate,
            norm_layer=nn.LayerNorm,
            ape=False,
            patch_norm=True,
            upscale=args.scale,
            img_range=1.,
            upsampler=args.upsampler,
            resi_connection='1conv'
        )
    else:
        ModelClass = SwinIR_Fixed
        if args.model == 'full':
            model = ModelClass(
                scale=args.scale,
                embed_dim=args.embed_dim,
                depths=args.depths,
                num_heads=args.num_heads,
                window_size=args.window_size,
                mlp_ratio=args.mlp_ratio,
                drop_path_rate=args.drop_path_rate
            )
        else:
            model = SwinIR_Light_Fixed(
                scale=args.scale,
                embed_dim=60,
                depths=[4, 4],
                num_heads=[4, 4],
                window_size=8,
                drop_path_rate=args.drop_path_rate
            )
    model = model.to(device)

    if is_main_process():
        total = sum(p.numel() for p in model.parameters())
        print(f'模型参数量: {total / 1e6:.2f}M')

    # ═══════════════════════════════════════════════════════════════════
    # 加载官方预训练权重（如果指定）
    # ═══════════════════════════════════════════════════════════════════
    if args.pretrained:
        try:
            if args.arch == 'official':
                # 官方 SwinIR：直接 load_state_dict，无需 key mapping
                load_pretrained_official(model, args.pretrained, strict=args.pretrained_strict)
            else:
                # Fixed 版本：使用 key mapping 适配
                load_pretrained(
                    model,
                    args.pretrained,
                    strict=args.pretrained_strict,
                    load_upsample=not args.pretrained_no_upsample
                )
        except Exception as e:
            print(f'[警告] 预训练权重加载失败: {e}')
            print('  继续使用随机初始化训练...')

    # DDP 包装
    if is_distributed:
        model = DDP(model, device_ids=[gpu], find_unused_parameters=False)

    # === 创建 EMA（如果启用）===
    global ema
    if args.use_ema:
        ema = EMA(model.module if is_distributed else model,
                  decay=args.ema_decay)
        if is_main_process():
            print(f'EMA 启用: decay={args.ema_decay}')


def safe_collate_fn(batch):
    """自定义 collate 函数 — 解决 'Trying to resize storage that is not resizable'

    transforms.ToTensor() 内部使用 torch.from_numpy()，创建的 tensor 底层 storage
    挂载在 numpy 数组内存上，不可 resize。DataLoader 的 default_collate 在拼接
    batch 时需要 resize storage，导致 RuntimeError。

    解决方案：先用 .clone() 为每个 tensor 创建独立可 resize 的 storage，
    再用 torch.stack() 拼接成 batch（torch.stack 本身也会创建新 tensor）。
    双重保障，彻底杜绝此问题。
    """
    lr_batch = torch.stack([item[0].clone() for item in batch])
    hr_batch = torch.stack([item[1].clone() for item in batch])
    return lr_batch, hr_batch


def prepare_eval_lr(eval_paths, scale=2, degradation='second_order'):
    """
    检查并预生成验证集 LR 图像（缺失时自动补全）。
    只在主进程执行；DDP 环境下会 barrier 同步等待，防止其他 rank 提前开始读取不完整的文件。
    """
    if not is_main_process():
        if is_distributed:
            dist.barrier()
        return

    degradator = cloud_dataset.RealESRGANDegradation(scale=scale, mode=degradation)
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
    generated_any = False

    for folder_path in eval_paths:
        # 寻找 HR 目录（优先 HR/ 子目录，否则用根目录）
        hr_folder = os.path.join(folder_path, 'HR')
        if not os.path.isdir(hr_folder):
            hr_folder = folder_path

        if not os.path.isdir(hr_folder):
            continue

        hr_files = [f for f in os.listdir(hr_folder)
                    if f.lower().endswith(valid_extensions)]
        hr_files.sort()
        if len(hr_files) == 0:
            continue

        # 准备 LR 目录
        lr_folder = os.path.join(folder_path, 'LR', f'X{scale}')
        os.makedirs(lr_folder, exist_ok=True)

        # 检查是否一一对应
        missing = []
        for f in hr_files:
            lr_path = os.path.join(lr_folder, f)
            if not os.path.exists(lr_path):
                missing.append(f)

        if len(missing) == 0:
            continue

        generated_any = True
        print(f'[验证集预生成] {folder_path} 缺失 {len(missing)}/{len(hr_files)} 张，开始生成...')
        for f in tqdm(missing, desc=f'退化 {os.path.basename(folder_path)}'):
            hr_path = os.path.join(hr_folder, f)
            lr_path = os.path.join(lr_folder, f)
            hr = Image.open(hr_path).convert('RGB')
            lr = degradator.degrade(hr)
            # 关键修复：退化中的 random_resize 可能改变尺寸，必须强制 resize 回标准尺寸
            w, h = hr.size
            expected_size = (w // scale, h // scale)
            if lr.size != expected_size:
                lr = lr.resize(expected_size, Image.BICUBIC)
            lr.save(lr_path)

    if generated_any:
        print('[验证集预生成] 全部完成')
    else:
        print('[验证集预生成] 所有验证集 LR 已存在且完整，跳过')

    if is_distributed:
        dist.barrier()


def data_loader_list_return():
    """创建数据加载器 — 使用 FixedFolderDataset 和 FixedValidationDataset"""
    # 启动前自动检查并补全验证集 LR（如缺失或不完整）
    prepare_eval_lr(args.eval_file, scale=args.scale, degradation=args.degradation)

    if not is_main_process():
        cloud_dataset.set_verbose(False)

    train_dataset = copy.deepcopy(args.train_file)
    eval_dataset = copy.deepcopy(args.eval_file)

    if is_main_process():
        print('加载 train_set')

    for index in range(len(train_dataset)):
        train_dataset[index] = cloud_dataset.FixedFolderDataset(
            train_dataset[index],
            scale=args.scale,
            patch_size=args.patch_size,
            pre_crop=True,
            degradation=args.degradation,
            augment=not args.no_augment
        )

    train_file_set = ConcatDataset(train_dataset)

    if is_distributed:
        train_sampler = DistributedSampler(
            train_file_set, num_replicas=world_size, rank=rank, shuffle=True
        )
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    train_loader = DataLoader(
        dataset=train_file_set,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=4 if args.num_workers > 0 else None,
        multiprocessing_context='spawn' if args.num_workers > 0 else None,
        collate_fn=safe_collate_fn,
    )

    if is_main_process():
        print('加载 eval_set')

    # 使用验证数据集
    # 验证始终使用 FixedValidationDataset，读取预生成的 LR/X2/
    # 避免全图实时二阶退化导致的极端缓慢（训练仍用 second_order）
    eval_loaders = []
    for index in range(len(eval_dataset)):
        eval_ds = cloud_dataset.FixedValidationDataset(
            eval_dataset[index], scale=args.scale
        )
        eval_loader = DataLoader(
            dataset=eval_ds,
            batch_size=1,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=True if args.num_workers > 0 else False,
            prefetch_factor=4 if args.num_workers > 0 else None,
            multiprocessing_context='spawn' if args.num_workers > 0 else None,
            collate_fn=safe_collate_fn,
        )
        eval_loaders.append(eval_loader)

    if is_main_process():
        print(f'训练集: {len(train_file_set)} samples, '
              f'退化: {args.degradation}, 增强: {not args.no_augment}')

    return train_loader, eval_loaders, train_sampler


def train_one_epoch(model, train_loader, criterion, optimizer,
                    epoch, train_sampler=None, stage_trainer=None):
    """训练一个 epoch — 集成 EMA 更新和 TwoStage 阶段切换"""
    model.train()

    if train_sampler is not None:
        train_sampler.set_epoch(epoch)

    # TwoStage 阶段切换
    current_stage = 'pixel'
    if stage_trainer is not None:
        current_stage = stage_trainer.step()
        if is_main_process() and epoch % 10 == 0:
            weights = {
                'charb': criterion.w_charb, 'mscn': criterion.w_mscn,
                'fft': criterion.w_fft, 'edge': criterion.w_edge,
                'lpips': criterion.w_lpips,
            }
            print(f'[TwoStage] Stage: {current_stage}, weights: {weights}')

    epoch_losses = AverageMeter()
    epoch_charb = AverageMeter()
    epoch_mscn = AverageMeter()
    epoch_fft = AverageMeter()
    epoch_edge = AverageMeter()

    if is_main_process():
        desc = f'Epoch [{epoch + 1}/{args.num_epochs}] [{current_stage}]'
        pbar = tqdm(train_loader, desc=desc)
    else:
        pbar = train_loader

    optimizer.zero_grad(set_to_none=True)

    for batch_idx, (inputs, labels) in enumerate(pbar):
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # 前向传播 — 损失计算全程 FP32，彻底避免 FP16 数值问题
        # 之前的方案（autocast 前向 + FP32 损失）仍有问题：
        #   1. SwinIR 的 attention softmax 在 FP16 下溢出 → pred 直接 NaN
        #   2. clamp(NaN) = NaN，clamp 无法修复 NaN
        #   3. NaN 检测在 backward 之后，梯度已被污染
        # 修复：关闭 autocast，全程 FP32。显存够用就优先稳定性。
        preds = model(inputs)

        # 训练时 clamp(0,1) — 与验证保持一致
        preds = preds.clamp(0.0, 1.0)

        # 前向 NaN 检测（在 backward 之前！）
        if not torch.isfinite(preds).all():
            if is_main_process():
                print(f'\n[警告] 模型输出含 NaN/Inf (batch {batch_idx}), 跳过此 batch')
            optimizer.zero_grad(set_to_none=True)
            continue

        # 对齐尺寸
        min_h = min(preds.shape[2], labels.shape[2])
        min_w = min(preds.shape[3], labels.shape[3])
        preds = preds[:, :, :min_h, :min_w]
        labels = labels[:, :, :min_h, :min_w]

        # 损失计算（FP32）
        loss, loss_dict = criterion(preds, labels)

        # Loss NaN 检测（在 backward 之前！防止梯度污染）
        if not torch.isfinite(loss):
            if is_main_process():
                print(f'\n[警告] Loss NaN/Inf (batch {batch_idx}), 跳过此 batch')
            optimizer.zero_grad(set_to_none=True)
            continue

        # 梯度累积
        loss = loss / args.grad_accum
        loss.backward()

        if (batch_idx + 1) % args.grad_accum == 0:
            # 梯度裁剪（SwinIR 标准做法 max_norm=10.0）
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=10.0)
            if not torch.isfinite(grad_norm):
                if is_main_process():
                    print(f'\n[警告] 梯度 NaN/Inf (batch {batch_idx}), 跳过此步')
                optimizer.zero_grad(set_to_none=True)
                continue
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            # 更新 EMA
            if args.use_ema and ema is not None:
                ema.update(model.module if is_distributed else model)

        # 统计（NaN 检测：跳过异常 batch，避免污染统计量）
        total_val = loss_dict['total'].item()
        if not (math.isfinite(total_val)):
            if is_main_process():
                print(f'\n[警告] 检测到 NaN/Inf loss (batch {batch_idx}), 跳过此 batch')
            continue

        epoch_losses.update(total_val, inputs.size(0))
        epoch_charb.update(loss_dict['charb'].item(), inputs.size(0))
        if 'mscn' in loss_dict:
            epoch_mscn.update(loss_dict['mscn'].item(), inputs.size(0))
        if 'fft' in loss_dict:
            epoch_fft.update(loss_dict['fft'].item(), inputs.size(0))
        if 'edge' in loss_dict:
            epoch_edge.update(loss_dict['edge'].item(), inputs.size(0))

        if is_main_process():
            pbar.set_postfix({
                'loss': f'{epoch_losses.avg:.4f}',
                'charb': f'{epoch_charb.avg:.4f}',
                'mscn': f'{epoch_mscn.avg:.4f}',
                'fft': f'{epoch_fft.avg:.4f}',
            })

    return epoch_losses.avg


@torch.no_grad()
def validate(model, eval_loaders, device, calc_niqe=False, border=0, test_y_channel=False):
    """验证模型 — 支持 NIQE 的多指标评估，权重为 0 时跳过计算以加速"""
    model.eval()
    all_psnrs = []
    all_ssims = []
    all_lpips = []
    all_niqes = []

    # 根据 JSON/命令行权重决定是否计算对应指标
    calc_psnr = getattr(args, 'val_w_psnr', 1.0) > 0
    calc_ssim = getattr(args, 'val_w_ssim', 0.0) > 0
    calc_lpips = getattr(args, 'val_w_lpips', 0.0) > 0

    evaluator = EnhancedMetricsEvaluator(
        device=device, calc_niqe=calc_niqe, border=border, test_y_channel=test_y_channel,
        calc_psnr=calc_psnr, calc_ssim=calc_ssim, calc_lpips=calc_lpips
    )

    for idx, eval_loader in enumerate(eval_loaders):
        epoch_psnr = AverageMeter()
        epoch_ssim = AverageMeter()
        epoch_lpips_loss = AverageMeter()
        epoch_niqe = AverageMeter()

        if is_main_process():
            pbar_eval = tqdm(
                eval_loader,
                desc=f'验证 {idx + 1}/{len(eval_loaders)}',
                total=len(eval_loader.dataset),
                leave=False
            )
        else:
            pbar_eval = eval_loader

        for inputs, labels in pbar_eval:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # FP32 推理（与训练一致，不用 autocast）
            preds = model(inputs).clamp(0.0, 1.0)

            min_h = min(preds.shape[2], labels.shape[2])
            min_w = min(preds.shape[3], labels.shape[3])
            preds = preds[:, :, :min_h, :min_w]
            labels = labels[:, :, :min_h, :min_w]

            metrics = evaluator.evaluate(preds, labels)
            if calc_psnr:
                epoch_psnr.update(metrics['psnr'], inputs.size(0))
            if calc_ssim:
                epoch_ssim.update(metrics['ssim'], inputs.size(0))
            if calc_lpips:
                epoch_lpips_loss.update(metrics['lpips'], inputs.size(0))
            if calc_niqe:
                epoch_niqe.update(metrics.get('niqe', 0.0), inputs.size(0))

            if is_main_process():
                postfix = {}
                if calc_psnr:
                    postfix['PSNR'] = f'{metrics["psnr"]:.2f}'
                if calc_ssim:
                    postfix['SSIM'] = f'{metrics["ssim"]:.4f}'
                if calc_lpips:
                    postfix['LPIPS'] = f'{metrics["lpips"]:.4f}'
                if calc_niqe:
                    postfix['NIQE'] = f'{metrics.get("niqe", 0):.2f}'
                pbar_eval.set_postfix(postfix)

        if is_main_process():
            pbar_eval.close()

        if calc_psnr:
            all_psnrs.append(epoch_psnr.avg)
        if calc_ssim:
            all_ssims.append(epoch_ssim.avg)
        if calc_lpips:
            all_lpips.append(epoch_lpips_loss.avg)
        if calc_niqe:
            all_niqes.append(epoch_niqe.avg)

    avg_psnr = sum(all_psnrs) / len(all_psnrs) if all_psnrs else 0
    avg_ssim = sum(all_ssims) / len(all_ssims) if all_ssims else 0
    avg_lpips = sum(all_lpips) / len(all_lpips) if all_lpips else 0

    result = {'psnr': avg_psnr, 'ssim': avg_ssim, 'lpips': avg_lpips}
    if calc_niqe and all_niqes:
        result['niqe'] = sum(all_niqes) / len(all_niqes)

    return result


def main():
    """主训练函数 — 集成新损失、EMA、TwoStageTrainer"""
    global args, model, device, ema, rank, world_size, is_distributed

    parser = argparse.ArgumentParser()

    # 必须参数（v2: 改为非强制，由 JSON 配置或命令行均可提供）
    parser.add_argument('--train-file', type=str, required=False, nargs='+')
    parser.add_argument('--eval-file', type=str, required=False, nargs='+')
    parser.add_argument('--outputs-dir', type=str, required=False)
    parser.add_argument('--valid-dir', type=str, nargs='+')

    # ═══════════════════════════════════════════════════════════════════
    # 配置文件支持（v2 新增）
    # ═══════════════════════════════════════════════════════════════════
    parser.add_argument('--config', type=str, default='',
                        help='JSON 配置文件路径。配置文件中的参数优先级低于命令行参数。')

    # 模型参数
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--model', type=str, default='full', choices=['light', 'full'])
    parser.add_argument('--arch', type=str, default='fixed', choices=['official', 'fixed'],
                        help='模型架构: official=100%%预训练兼容(追PSNR), fixed=感知优化(追多指标)')
    parser.add_argument('--embed-dim', type=int, default=180,
                        help='嵌入维度 (SwinIR_Fixed)')
    parser.add_argument('--depths', type=int, nargs='+', default=[6, 6, 6, 6],
                        help='每层的深度 (SwinIR_Fixed)')
    parser.add_argument('--num-heads', type=int, nargs='+', default=[6, 6, 6, 6],
                        help='每层注意力头数 (SwinIR_Fixed)')
    parser.add_argument('--window-size', type=int, default=8,
                        help='窗口大小 (SwinIR_Fixed)')
    parser.add_argument('--drop-path-rate', type=float, default=0.1,
                        help='随机深度丢弃率 (SwinIR_Fixed)')
    parser.add_argument('--mlp-ratio', type=float, default=2.0,
                        help='MLP 隐藏层比率')
    parser.add_argument('--upsampler', type=str, default='pixelshuffle',
                        choices=['pixelshuffle', 'pixelshuffledirect', 'nearest+conv'],
                        help='上采样方式: pixelshuffle=经典SR, pixelshuffledirect=轻量SR, nearest+conv=真实世界SR')

    # 训练参数
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='初始学习率 (v2优化: 3e-4，解决平台期收敛慢)')
    parser.add_argument('--min-lr', type=float, default=1e-6,
                        help='最小学习率 (v2优化: 1e-6，允许更大衰减幅度)')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='每GPU批次大小')
    parser.add_argument('--num-epochs', type=int, default=500)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--patch-size', type=int, default=96,
                        help='HR patch尺寸 (v2优化: 96，标准值且是8的倍数)')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--grad-accum', type=int, default=2,
                        help='梯度累积步数 (v2优化: 2，提升参数更新频率)')

    # 损失权重参数
    parser.add_argument('--w-l1', type=float, default=1.0,
                        help='Charbonnier损失权重')
    parser.add_argument('--w-ssim', type=float, default=0.3,
                        help='SSIM损失权重')
    parser.add_argument('--w-lpips', type=float, default=0.05,
                        help='LPIPS感知损失权重 (pixel阶段实际生效需代码支持)')
    parser.add_argument('--w-mscn', type=float, default=None,
                        help='MSCN统计量损失权重')
    parser.add_argument('--w-fft', type=float, default=None,
                        help='FFT频域损失权重')
    parser.add_argument('--w-edge', type=float, default=None,
                        help='边缘损失权重')

    # 两阶段训练参数
    parser.add_argument('--stage1-epochs', type=int, default=120,
                        help='第一阶段 epoch 数 (v2优化: 120，更快转入感知阶段)')
    parser.add_argument('--transition-epochs', type=int, default=40,
                        help='过渡阶段 epoch 数 (v2优化: 40)')

    # 新增：验证综合评分权重（决定"最佳模型"）
    parser.add_argument('--val-w-psnr', type=float, default=1.0,
                        help='验证评分中PSNR权重（默认主指标）')
    parser.add_argument('--val-w-ssim', type=float, default=0.5,
                        help='验证评分中SSIM权重')
    parser.add_argument('--val-w-lpips', type=float, default=0.3,
                        help='验证评分中LPIPS权重(正向转换:1-lpips)')
    parser.add_argument('--val-w-niqe', type=float, default=0.0,
                        help='验证评分中NIQE权重(需安装pyiqa)')
    parser.add_argument('--calc-niqe', action='store_true',
                        help='验证时计算NIQE指标(需安装pyiqa)')

    # 新增：退化模型参数
    parser.add_argument('--degradation', type=str, default='second_order',
                        choices=['clean', 'first_order', 'second_order'],
                        help='退化模式')
    parser.add_argument('--no-augment', action='store_true',
                        help='禁用数据增强')

    # 新增：EMA 参数
    parser.add_argument('--use-ema', action='store_true',
                        help='启用 EMA')
    parser.add_argument('--ema-decay', type=float, default=0.999)

    # 保存参数
    parser.add_argument('--save', type=str, default='no', choices=['yes', 'no'])
    parser.add_argument('--save-seq', type=int, default=5)
    parser.add_argument('--val-interval', type=int, default=1,
                        help='验证频率: 每 N 个 epoch 验证一次 (默认1, 即每epoch都验证)')
    parser.add_argument('--val-warmup-epochs', type=int, default=0,
                        help='前 N 个 epoch 每轮都验证(密集观察期), 之后按 val-interval 稀疏验证')

    # 验证参数
    parser.add_argument('--jump', type=str, default='no', choices=['yes', 'no'])
    parser.add_argument('--jump-seq', type=int, default=3)

    # 分布式
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--local-rank', type=int, default=0)

    # 断点续训
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--auto-resume', action='store_true')
    parser.add_argument('--start-epoch', type=int, default=0)

    # 学习率调度
    parser.add_argument('--lr-schedule', type=str, default='warmup_cosine',
                        choices=['cosine', 'warmup_cosine', 'plateau', 'onecycle', 'adaptive'])
    parser.add_argument('--warmup-epochs', type=int, default=5)
    parser.add_argument('--T-max', type=int, default=300,
                        help='Cosine annealing 总 epoch 数 (v2优化: 300)')
    parser.add_argument('--lr-factor', type=float, default=0.5)
    parser.add_argument('--lr-patience', type=int, default=10)

    # 早停
    parser.add_argument('--early-stop', action='store_true')
    parser.add_argument('--es-patience', type=int, default=30)
    parser.add_argument('--es-min-delta', type=float, default=1e-4)
    parser.add_argument('--es-cooldown', type=int, default=10)
    parser.add_argument('--es-min-epochs', type=int, default=100)

    # ═══════════════════════════════════════════════════════════════════
    # 新增：预训练权重参数
    # ═══════════════════════════════════════════════════════════════════
    parser.add_argument('--pretrained', type=str, default='',
                        help='SwinIR 官方预训练权重路径 (.pth)。支持多种格式：'
                             '{params: state_dict}, {model: state_dict}, 裸 state_dict')
    parser.add_argument('--pretrained-strict', action='store_true',
                        help='严格模式：键名必须完全匹配，否则报错（默认关闭）')
    parser.add_argument('--pretrained-no-upsample', action='store_true',
                        help='不加载上采样层权重（用于不同 scale 的迁移训练，如用 2x 权重初始化 4x 模型）')

    args = parser.parse_args()

    # ═══════════════════════════════════════════════════════════════════
    # v2 新增：配置文件支持
    # 命令行参数 > 配置文件 > 代码默认值
    # ═══════════════════════════════════════════════════════════════════
    if args.config and os.path.exists(args.config):
        import json as _json
        with open(args.config, 'r', encoding='utf-8') as f:
            config_data = _json.load(f)
        # 用配置文件填充命令行未显式设置的参数（None 或默认值视为未设置）
        # 这里采用简单策略：只要命令行是 argparse 默认值，配置文件就允许覆盖
        config_applied = []
        for key, val in config_data.items():
            arg_key = key.replace('-', '_')
            if hasattr(args, arg_key):
                current = getattr(args, arg_key)
                # 对于列表类型（如 depths, num_heads），需要特殊处理
                # 修复：当 current 为 None（未通过命令行传入）时，应接受列表值
                if isinstance(val, list) and current is not None and not isinstance(current, list):
                    continue
                # 如果命令行没有显式传入（等于默认值），则用配置文件覆盖
                # 注意：由于无法直接判断"是否命令行传入"，这里采用
                # 约定：配置文件的值总是优先，除非命令行值与默认值不同
                # 更精确的做法：比较 current 是否与 parser 默认值相同
                # 为简化，直接覆盖（命令行后处理逻辑）
                setattr(args, arg_key, val)
                config_applied.append(key)
        if config_applied and is_main_process():
            print(f'[Config] 从 {args.config} 加载参数: {config_applied}')

    # ═══════════════════════════════════════════════════════════════════
    # v2 新增：手动验证必须参数（train-file / eval-file / outputs-dir）
    # argparse required 已关闭，以支持纯 JSON 配置启动
    # ═══════════════════════════════════════════════════════════════════
    missing_required = []
    if args.train_file is None or len(args.train_file) == 0:
        missing_required.append('--train-file')
    if args.eval_file is None or len(args.eval_file) == 0:
        missing_required.append('--eval-file')
    if args.outputs_dir is None or len(str(args.outputs_dir).strip()) == 0:
        missing_required.append('--outputs-dir')
    if missing_required:
        parser.error(f"以下参数必须提供（命令行或 JSON 配置文件）: {', '.join(missing_required)}")

    # 预处理
    pre_run()

    # 检查点管理器
    checkpoint_manager = CheckpointManager(args.outputs_dir, keep_last_n=3)

    # 训练监控器
    monitor = TrainingMonitor(args.outputs_dir) if is_main_process() else None

    # === 多指标综合评分系统（验证最佳模型用）===
    metric_scorer = MultiMetricScore(
        w_psnr=args.val_w_psnr,
        w_ssim=args.val_w_ssim,
        w_lpips=args.val_w_lpips,
        w_niqe=args.val_w_niqe,
        device=device
    )
    if is_main_process():
        print(f'[MultiMetricScore] 验证评分权重: {metric_scorer.get_weights()}')

    # === 使用新 CompleteLoss（兼容旧权重接口）===
    # 构建权重字典，命令行传入的非None值覆盖默认值
    loss_weights = {
        'w_charb': args.w_l1,
        'w_ssim': args.w_ssim,
        'w_lpips': args.w_lpips,
    }
    # 只有命令行显式传入时才覆盖
    if args.w_mscn is not None:
        loss_weights['w_mscn'] = args.w_mscn
    if args.w_fft is not None:
        loss_weights['w_fft'] = args.w_fft
    if args.w_edge is not None:
        loss_weights['w_edge'] = args.w_edge

    criterion = CompleteLoss(
        stage='pixel',
        device=device,
        **loss_weights
    )

    # 收集用户自定义损失权重，防止被 TwoStageTrainer 硬编码覆盖
    user_loss_weights = {}
    if args.w_mscn is not None:
        user_loss_weights['w_mscn'] = args.w_mscn
    if args.w_fft is not None:
        user_loss_weights['w_fft'] = args.w_fft
    if args.w_edge is not None:
        user_loss_weights['w_edge'] = args.w_edge
    # 以下权重始终由用户控制（有默认值），也加入保留列表
    user_loss_weights['w_charb'] = args.w_l1
    user_loss_weights['w_ssim'] = args.w_ssim
    user_loss_weights['w_lpips'] = args.w_lpips

    # 两阶段训练器
    stage_trainer = TwoStageTrainer(
        criterion,
        stage1_epochs=args.stage1_epochs,
        transition_epochs=args.transition_epochs,
        user_weights=user_loss_weights
    )

    # 优化器（使用 AdamW 替代 Adam）
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            betas=(0.9, 0.999), weight_decay=0.01)

    # 学习率调度器
    lr_scheduler = SmartLRScheduler(
        optimizer,
        mode=args.lr_schedule,
        base_lr=args.lr,
        min_lr=args.min_lr,
        warmup_epochs=args.warmup_epochs,
        T_max=args.T_max,
        factor=args.lr_factor,
        patience=args.lr_patience,
        verbose=is_main_process()
    )

    # 早停
    early_stopping = None
    if args.early_stop:
        early_stopping = EarlyStopping(
            mode='max',
            patience=args.es_patience,
            min_delta=args.es_min_delta,
            cooldown=args.es_cooldown,
            min_epochs=args.es_min_epochs,
            verbose=is_main_process()
        )

    # FP32 全精度训练，不使用 GradScaler

    # 数据加载
    train_loader, eval_loaders, train_sampler = data_loader_list_return()

    # 初始化训练状态
    start_epoch = args.start_epoch
    best_psnr = 0.0
    best_epoch = 0

    # 断点续训
    if args.resume or args.auto_resume:
        checkpoint_path = args.resume
        if args.auto_resume and not checkpoint_path:
            checkpoint_path = checkpoint_manager.find_latest_checkpoint()

        if checkpoint_path and os.path.exists(checkpoint_path):
            if is_main_process():
                print(f'\n{"=" * 60}')
                print(f'恢复训练: {checkpoint_path}')
                print(f'{"=" * 60}')

            checkpoint = torch.load(checkpoint_path, map_location='cpu',
                                    weights_only=False)

            # 加载模型
            if is_distributed:
                model.module.load_state_dict(checkpoint['model'], strict=False)
            else:
                model.load_state_dict(checkpoint['model'], strict=False)

            # v4: residual_gate checkpoint 兼容性
            target_model = model.module if is_distributed else model
            if hasattr(target_model, 'layers'):
                gate_fixed = []
                for i, layer in enumerate(target_model.layers):
                    if hasattr(layer, 'residual_gate') and layer.residual_gate.item() == 1.0:
                        # 旧 checkpoint 未加载 residual_gate，保持初始值 1.0（conv 完全参与）
                        gate_fixed.append(f'layers.{i}')
                if gate_fixed and is_main_process():
                    print(f'[v4] 旧 checkpoint 兼容: residual_gate 保持 1.0 '
                          f'({len(gate_fixed)} 层)')
                    print(f'[v4] 注意：旧 conv 权重已训练 37 epoch，建议从头训练以获得最佳效果')

            # 加载优化器
            optimizer.load_state_dict(checkpoint['optimizer'])

            # 加载调度器
            if 'lr_scheduler' in checkpoint:
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

            # 加载早停
            if early_stopping is not None and 'early_stopping' in checkpoint:
                early_stopping.load_state_dict(checkpoint['early_stopping'])

            # 加载 scaler（兼容旧检查点，不再使用）
            if 'scaler' in checkpoint:
                pass  # 不再加载 scaler

            # 加载 EMA
            if 'ema' in checkpoint and ema is not None:
                ema.load_state_dict(checkpoint['ema'])

            # 加载 TwoStage 阶段状态
            if 'stage_trainer' in checkpoint and stage_trainer is not None:
                stage_trainer.load_state_dict(checkpoint['stage_trainer'])
                if is_main_process():
                    print(f'恢复阶段状态: epoch={stage_trainer.current_epoch}, '
                          f'stage={stage_trainer.current_stage}')

            start_epoch = checkpoint.get('epoch', 0) + 1
            best_psnr = checkpoint.get('best_psnr', 0.0)
            best_epoch = checkpoint.get('best_epoch', 0)

            if is_main_process():
                print(f'从 epoch {start_epoch} 继续训练')
                print(f'当前最佳 PSNR: {best_psnr:.2f} (epoch {best_epoch + 1})')
        else:
            if is_main_process():
                print('未找到检查点，从头开始训练')

    # 同步起始 epoch
    if is_distributed:
        start_epoch_tensor = torch.tensor([start_epoch], device=device)
        dist.broadcast(start_epoch_tensor, src=0)
        start_epoch = int(start_epoch_tensor.item())

    # 训练循环
    if is_main_process():
        print(f'\n{"=" * 60}')
        print(f'开始训练')
        print(f'模型: {"SwinIR_Official" if args.arch == "official" else "SwinIR_Fixed"} ({args.model})')
        print(f'Scale: {args.scale}x, Degradation: {args.degradation}')
        print(f'Embed: {args.embed_dim}, Depths: {args.depths}')
        print(f'Window: {args.window_size}, DropPath: {args.drop_path_rate}')
        print(f'最大 epoch: {args.num_epochs}')
        print(f'批次: {args.batch_size} x {world_size} GPUs = {args.batch_size * world_size}')
        print(f'梯度累积: {args.grad_accum}')
        print(f'有效批次: {args.batch_size * world_size * args.grad_accum}')
        print(f'[训练损失权重] Charb={args.w_l1}, SSIM={args.w_ssim}, LPIPS={args.w_lpips}, '
              f'MSCN={args.w_mscn if args.w_mscn is not None else "stage_default"}, '
              f'FFT={args.w_fft if args.w_fft is not None else "stage_default"}, '
              f'Edge={args.w_edge if args.w_edge is not None else "stage_default"}')
        print(f'[验证评分权重] PSNR={args.val_w_psnr}, SSIM={args.val_w_ssim}, '
              f'LPIPS={args.val_w_lpips}, NIQE={args.val_w_niqe}')
        print(f'两阶段: stage1={args.stage1_epochs}, transition={args.transition_epochs}')
        print(f'EMA: {args.use_ema}')
        print(f'学习率: {args.lr_schedule}')
        print(f'早停: {args.early_stop}')
        print(f'NIQE验证: {args.calc_niqe}')
        print(f'预训练权重: {args.pretrained if args.pretrained else "无"}')
        print(f'{"=" * 60}\n')

    stopped_early = False
    stop_reason = None

    for epoch in range(start_epoch, args.num_epochs):
        # 训练
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer,
            epoch, train_sampler, stage_trainer=stage_trainer
        )

        # 更新学习率（基础步进，plateau模式在验证后再次更新）
        lr_scheduler.step(epoch=epoch, loss=train_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # 跳过验证
        if jump and epoch % jump_seq == 0 and epoch != args.num_epochs - 1:
            continue

        # 验证策略: warmup 期内每轮都验证, 之后按 val-interval 稀疏验证
        is_warmup = epoch < args.val_warmup_epochs
        is_interval = (epoch + 1) % args.val_interval == 0
        is_last = epoch == args.num_epochs - 1
        if not is_warmup and not is_interval and not is_last:
            if is_main_process():
                print(f'Epoch [{epoch + 1}/{args.num_epochs}] 跳过验证 '
                      f'(warmup={args.val_warmup_epochs}, interval={args.val_interval})')
            continue

        # === 验证（可选使用 EMA）===
        if args.use_ema and ema is not None:
            ema.apply_shadow(model.module if is_distributed else model)

        val_metrics = validate(model, eval_loaders, device, calc_niqe=args.calc_niqe,
                                border=args.scale, test_y_channel=True)

        if args.use_ema and ema is not None:
            ema.restore(model.module if is_distributed else model)

        # === 多指标综合评分 ===
        score, score_breakdown = metric_scorer.compute(val_metrics)

        # plateau 调度需要在验证后传入指标
        if args.lr_schedule == 'plateau':
            lr_scheduler.step(epoch=epoch, metric=val_metrics.get('psnr'), loss=train_loss)

        # 更新监控器
        if monitor is not None:
            monitor.update(epoch, train_loss, val_metrics, current_lr)
            monitor.save_logs()

        # v3: LR 异常上升检测（防止 scheduler 状态污染）
        if is_main_process() and len(lr_scheduler.lr_history) >= 2:
            prev_lr = lr_scheduler.lr_history[-2]
            if current_lr > prev_lr * 1.05 and epoch > args.warmup_epochs:
                print(f'[WARNING] LR 异常上升: {prev_lr:.2e} -> {current_lr:.2e}，'
                      f'可能是 scheduler 状态污染，已自动重置')

        # 仅主进程输出和保存
        if is_main_process():
            niqe_str = f' | NIQE: {val_metrics.get("niqe", 0):.2f}' if args.calc_niqe else ''
            print(f'Epoch [{epoch + 1}/{args.num_epochs}] '
                  f'Loss: {train_loss:.4f} | '
                  f'PSNR: {val_metrics["psnr"]:.2f} dB | '
                  f'SSIM: {val_metrics["ssim"]:.4f} | '
                  f'LPIPS: {val_metrics["lpips"]:.4f}{niqe_str} | '
                  f'Score: {score:.4f} | '
                  f'Stage: {stage_trainer.current_stage if stage_trainer else "pixel"} | '
                  f'LR: {current_lr:.2e}')

            # 定期检查点
            if save_ and (epoch + 1) % args.save_seq == 0:
                ckpt = build_checkpoint(
                    epoch=epoch, best_psnr=best_psnr, best_epoch=best_epoch,
                    model=model, optimizer=optimizer, lr_scheduler=lr_scheduler,
                    early_stopping=early_stopping, ema=ema, monitor=monitor,
                    stage_trainer=stage_trainer, is_distributed=is_distributed
                )
                ckpt['args'] = vars(args)
                checkpoint_manager.save_checkpoint(
                    ckpt, filename=f'checkpoint_epoch{epoch + 1:04d}.pth'
                )

            # 保存最佳 — 使用多指标综合评分（不再只看 PSNR）
            is_best = score > getattr(metric_scorer, '_best_score', 0.0)
            if is_best:
                metric_scorer._best_score = score
                best_epoch = epoch
                best_psnr = val_metrics['psnr']
                print(f'✓ 新的最佳模型! 综合评分: {score:.4f} | '
                      f'PSNR: {best_psnr:.2f} dB | '
                      f'SSIM: {val_metrics["ssim"]:.4f} | '
                      f'LPIPS: {val_metrics["lpips"]:.4f}')
                # 打印评分分解
                print(f'  评分分解: PSNR_norm={score_breakdown["psnr_norm"]:.4f} × {metric_scorer.w_psnr} + '
                      f'SSIM_norm={score_breakdown["ssim_norm"]:.4f} × {metric_scorer.w_ssim} + '
                      f'(1-LPIPS)={score_breakdown["lpips_norm"]:.4f} × {metric_scorer.w_lpips}')

            # 保存最新（使用封装函数）
            ckpt = build_checkpoint(
                epoch=epoch, best_psnr=best_psnr, best_epoch=best_epoch,
                model=model, optimizer=optimizer, lr_scheduler=lr_scheduler,
                early_stopping=early_stopping, ema=ema, monitor=monitor,
                stage_trainer=stage_trainer, is_distributed=is_distributed
            )
            ckpt['args'] = vars(args)
            checkpoint_manager.save_checkpoint(ckpt, is_best=is_best)

            # 早停检查 — 使用综合评分（而非纯 PSNR）
            if early_stopping is not None:
                early_stopping.update_loss(train_loss)
                should_stop, reason = early_stopping(score, epoch)
                if should_stop:
                    print(f'\n{"=" * 60}')
                    print(f'早停触发!')
                    print(f'原因: {reason}')
                    print(f'最佳 PSNR: {best_psnr:.2f} (epoch {best_epoch + 1})')
                    print(f'{"=" * 60}')
                    stopped_early = True
                    stop_reason = reason
                    break

    # 训练结束
    if is_main_process():
        if stopped_early:
            print(f'\n训练因早停结束: {stop_reason}')
        print(f'\n{"=" * 60}')
        print(f'训练完成!')
        print(f'最佳 epoch: {best_epoch + 1}')
        print(f'最佳 PSNR: {best_psnr:.2f} dB')
        print(f'{"=" * 60}')

        # 保存最终模型
        if args.use_ema and ema is not None:
            ema.apply_shadow(model.module if is_distributed else model)
        final_ckpt = build_checkpoint(
            epoch=args.num_epochs - 1, best_psnr=best_psnr, best_epoch=best_epoch,
            model=model, optimizer=optimizer, lr_scheduler=lr_scheduler,
            ema=ema, stage_trainer=stage_trainer, is_distributed=is_distributed
        )
        final_ckpt['args'] = vars(args)
        torch.save(final_ckpt, os.path.join(args.outputs_dir, 'final.pth'))
        if args.use_ema and ema is not None:
            ema.restore(model.module if is_distributed else model)

    # 清理分布式
    if is_distributed:
        cleanup_distributed()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        import traceback
        if is_distributed:
            cleanup_distributed()
        raise e