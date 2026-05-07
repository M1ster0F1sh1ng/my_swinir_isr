"""
增强版图像指标评估器
支持：
1. YCbCr Y 通道 PSNR（学术标准）
2. border crop（裁剪边界 artifacts）
3. RGB PSNR（兼容旧逻辑）
"""

import torch
import torch.nn.functional as F


def rgb_to_ycbcr(img):
    """
    RGB -> YCbCr (MATLAB 标准系数)
    img: [B, 3, H, W] in [0, 1], RGB order
    return: [B, 3, H, W], Y in [16/255, 235/255], Cb/Cr in [16/255, 240/255]
    """
    # 系数矩阵 (MATLAB rgb2ycbcr)
    # Y  = 16  + 65.481*R + 128.553*G + 24.966*B
    # Cb = 128 - 37.797*R - 74.203*G + 112.0*B
    # Cr = 128 + 112.0*R  - 93.786*G - 18.214*B
    kr, kg, kb = 65.481, 128.553, 24.966
    
    r, g, b = img[:, 0:1, :, :], img[:, 1:2, :, :], img[:, 2:3, :, :]
    
    y = 16.0/255.0 + (kr/255.0)*r + (kg/255.0)*g + (kb/255.0)*b
    return y


def calc_psnr(img1, img2, max_val=1.0, border=0):
    """计算 PSNR，支持 border crop"""
    if border > 0:
        img1 = img1[:, :, border:-border, border:-border]
        img2 = img2[:, :, border:-border, border:-border]
    
    mse = torch.mean((img1 - img2) ** 2)
    if mse < 1e-10:
        return float('inf')
    return (20 * torch.log10(torch.tensor(max_val, device=mse.device) / torch.sqrt(mse))).item()


def calc_ssim(img1, img2, window_size=11):
    """计算 SSIM"""
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=window_size // 2)
    mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=window_size // 2)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.avg_pool2d(img1 ** 2, window_size, stride=1, padding=window_size // 2) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 ** 2, window_size, stride=1, padding=window_size // 2) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, window_size, stride=1, padding=window_size // 2) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean().item()


class ImageMetricsEvaluator:
    """
    增强版图像指标评估器
    
    Args:
        device: 计算设备
        border: 计算 PSNR/SSIM 前裁剪的边界像素（通常 = scale）
        test_y_channel: 是否在 YCbCr Y 通道上计算 PSNR（学术标准）
    """

    def __init__(self, device='cuda', border=0, test_y_channel=False):
        self.device = device
        self.border = border
        self.test_y_channel = test_y_channel
        
        try:
            import lpips
            self.lpips_model = lpips.LPIPS(net='alex').to(device)
            self.has_lpips = True
        except ImportError:
            self.has_lpips = False

    def evaluate(self, pred, target):
        pred = pred.to(self.device)
        target = target.to(self.device)
        
        # 尺寸对齐
        min_h = min(pred.shape[2], target.shape[2])
        min_w = min(pred.shape[3], target.shape[3])
        pred = pred[:, :, :min_h, :min_w]
        target = target[:, :, :min_h, :min_w]
        
        metrics = {}
        
        # === RGB PSNR / SSIM ===
        metrics['psnr_rgb'] = calc_psnr(pred, target, border=self.border)
        metrics['ssim_rgb'] = calc_ssim(pred, target)
        
        # === YCbCr Y 通道 PSNR（学术标准）===
        if self.test_y_channel:
            pred_y = rgb_to_ycbcr(pred)
            target_y = rgb_to_ycbcr(target)
            metrics['psnr'] = calc_psnr(pred_y, target_y, border=self.border)
            metrics['ssim'] = calc_ssim(pred_y, target_y)
        else:
            metrics['psnr'] = metrics['psnr_rgb']
            metrics['ssim'] = metrics['ssim_rgb']
        
        # LPIPS（始终用 RGB）
        if self.has_lpips:
            pred_lpips = pred * 2 - 1
            target_lpips = target * 2 - 1
            with torch.no_grad():
                metrics['lpips'] = self.lpips_model(pred_lpips, target_lpips).mean().item()
        else:
            metrics['lpips'] = 0.0
            
        return metrics
