"""
超分辨率质量评估脚本
支持有参考图和无参考图两种模式

Usage:
    # 有参考图（标准 SR 评估）
    python evaluate_sr.py --ref reference.png --sr output.png
    
    # 无参考图（真实世界 SR 评估）
    python evaluate_sr.py --sr output.png --no-ref
    
    # 批量对比多个方法
    python evaluate_sr.py --ref reference.png --sr-list bilinear.png srcnn.png swinir.png
"""

import argparse
import os
import numpy as np
from PIL import Image
import torch


def load_image(path):
    """加载图像并转为 numpy [H, W, C] uint8"""
    img = Image.open(path).convert('RGB')
    return np.array(img)


def calc_psnr(img1, img2, max_val=255.0, border=0):
    """计算 PSNR"""
    if border > 0:
        img1 = img1[border:-border, border:-border]
        img2 = img2[border:-border, border:-border]
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse < 1e-10:
        return float('inf')
    return 20 * np.log10(max_val / np.sqrt(mse))


def calc_ssim(img1, img2):
    """简化版 SSIM"""
    import torch.nn.functional as F
    t1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    t2 = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    mu1 = F.avg_pool2d(t1, 11, stride=1, padding=5)
    mu2 = F.avg_pool2d(t2, 11, stride=1, padding=5)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.avg_pool2d(t1 ** 2, 11, stride=1, padding=5) - mu1_sq
    sigma2_sq = F.avg_pool2d(t2 ** 2, 11, stride=1, padding=5) - mu2_sq
    sigma12 = F.avg_pool2d(t1 * t2, 11, stride=1, padding=5) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean().item()


def calc_lpips(img1, img2, device='cpu'):
    """计算 LPIPS"""
    try:
        import lpips
        loss_fn = lpips.LPIPS(net='alex').to(device)
        t1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float() / 255.0 * 2 - 1
        t2 = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float() / 255.0 * 2 - 1
        with torch.no_grad():
            dist = loss_fn(t1.to(device), t2.to(device))
        return dist.item()
    except ImportError:
        return None


def calc_noref_metrics(img):
    """
    无参考质量指标
    """
    metrics = {}
    gray = img.mean(axis=2)
    
    # 1. 局部对比度（使用局部标准差）
    from scipy.ndimage import uniform_filter
    local_mean = uniform_filter(gray.astype(float), size=7)
    local_mean_sq = uniform_filter(gray.astype(float) ** 2, size=7)
    local_std = np.sqrt(np.maximum(local_mean_sq - local_mean ** 2, 0))
    metrics['local_contrast'] = local_std.mean()
    
    # 2. 边缘锐度（Sobel 梯度幅度）
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    from scipy.signal import convolve2d
    gx = convolve2d(gray, sobel_x, mode='same', boundary='symm')
    gy = convolve2d(gray, sobel_y, mode='same', boundary='symm')
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    metrics['edge_sharpness'] = grad_mag.mean()
    
    # 3. 信息熵
    hist, _ = np.histogram(img.flatten(), bins=256, range=(0, 256))
    prob = hist / hist.sum()
    prob = prob[prob > 0]
    metrics['entropy'] = -np.sum(prob * np.log2(prob))
    
    # 4. 色彩丰富度（标准差的均值）
    metrics['colorfulness'] = img.std(axis=2).mean()
    
    # 5. Block artifact 检测（检测 8x8 边界跳跃）
    h, w = gray.shape
    block_jumps_h = []
    block_jumps_v = []
    for i in range(8, h, 8):
        if i < h:
            block_jumps_h.append(np.abs(gray[i] - gray[i-1]).mean())
    for j in range(8, w, 8):
        if j < w:
            block_jumps_v.append(np.abs(gray[:, j] - gray[:, j-1]).mean())
    
    if block_jumps_h and block_jumps_v:
        avg_jump = (np.mean(block_jumps_h) + np.mean(block_jumps_v)) / 2
        # 与整体差分比较
        overall_diff = (np.abs(np.diff(gray, axis=0)).mean() + np.abs(np.diff(gray, axis=1)).mean()) / 2
        metrics['block_artifact_ratio'] = avg_jump / (overall_diff + 1e-8)
    else:
        metrics['block_artifact_ratio'] = 1.0
    
    # 6. 频域分析：高频能量比例
    from numpy.fft import fft2, fftshift
    freq = np.abs(fftshift(fft2(gray)))
    h, w = freq.shape
    low_freq = freq[:h//4, :w//4].mean()
    high_freq = freq[h*3//4:, w*3//4:].mean()
    metrics['high_freq_ratio'] = high_freq / (low_freq + 1e-8)
    
    return metrics


def evaluate_with_ref(ref_path, sr_path, border=0, device='cpu'):
    """有参考图评估"""
    ref = load_image(ref_path)
    sr = load_image(sr_path)
    
    # 尺寸对齐
    min_h = min(ref.shape[0], sr.shape[0])
    min_w = min(ref.shape[1], sr.shape[1])
    ref = ref[:min_h, :min_w]
    sr = sr[:min_h, :min_w]
    
    result = {
        'psnr': calc_psnr(ref, sr, border=border),
        'ssim': calc_ssim(ref, sr),
        'lpips': calc_lpips(ref, sr, device=device),
    }
    
    # 无参考指标也一并计算
    noref = calc_noref_metrics(sr)
    result.update({f'noref_{k}': v for k, v in noref.items()})
    
    return result


def evaluate_no_ref(sr_path):
    """无参考图评估"""
    sr = load_image(sr_path)
    return calc_noref_metrics(sr)


def print_metrics(metrics, name=''):
    """打印指标"""
    prefix = f"[{name}] " if name else ""
    for k, v in metrics.items():
        if v is None:
            print(f"  {prefix}{k}: N/A")
        elif isinstance(v, float):
            if 'psnr' in k.lower():
                print(f"  {prefix}{k}: {v:.2f} dB")
            elif 'ssim' in k.lower():
                print(f"  {prefix}{k}: {v:.4f}")
            elif 'lpips' in k.lower():
                print(f"  {prefix}{k}: {v:.4f}")
            else:
                print(f"  {prefix}{k}: {v:.4f}")
        else:
            print(f"  {prefix}{k}: {v}")


def main():
    parser = argparse.ArgumentParser(description='SR 质量评估')
    parser.add_argument('--ref', type=str, default=None, help='参考 HR 图像路径')
    parser.add_argument('--sr', type=str, default=None, help='SR 输出图像路径')
    parser.add_argument('--sr-list', type=str, nargs='+', help='多个 SR 输出路径（对比模式）')
    parser.add_argument('--border', type=int, default=0, help='裁剪边界像素')
    parser.add_argument('--no-ref', action='store_true', help='无参考模式')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
    args = parser.parse_args()
    
    if args.no_ref or args.ref is None:
        # 无参考模式
        if args.sr_list:
            print("=" * 60)
            print("无参考质量评估（多方法对比）")
            print("=" * 60)
            for sr_path in args.sr_list:
                name = os.path.basename(sr_path)
                metrics = evaluate_no_ref(sr_path)
                print_metrics(metrics, name)
                print()
        elif args.sr:
            print("=" * 60)
            print("无参考质量评估")
            print("=" * 60)
            metrics = evaluate_no_ref(args.sr)
            print_metrics(metrics)
        else:
            print("错误: 无参考模式需要 --sr 或 --sr-list")
            return
    else:
        # 有参考模式
        if args.sr_list:
            print("=" * 60)
            print(f"有参考质量评估（参考图: {args.ref}）")
            print("=" * 60)
            for sr_path in args.sr_list:
                name = os.path.basename(sr_path)
                metrics = evaluate_with_ref(args.ref, sr_path, border=args.border, device=args.device)
                print_metrics(metrics, name)
                print()
        elif args.sr:
            print("=" * 60)
            print(f"有参考质量评估（参考图: {args.ref}）")
            print("=" * 60)
            metrics = evaluate_with_ref(args.ref, args.sr, border=args.border, device=args.device)
            print_metrics(metrics)
        else:
            print("错误: 有参考模式需要 --sr 或 --sr-list")
            return
    
    print("=" * 60)
    print("指标说明:")
    print("  PSNR: 越高越好（像素精度）")
    print("  SSIM: 越高越好（结构相似度）")
    print("  LPIPS: 越低越好（感知距离）")
    print("  local_contrast: 局部对比度，越高通常越清晰")
    print("  edge_sharpness: 边缘锐度，越高说明边缘越锐利")
    print("  entropy: 信息熵，越高信息量越大")
    print("  colorfulness: 色彩丰富度")
    print("  block_artifact_ratio: 块效应比，接近 1.0 说明无明显 block artifact，>1.5 可能有")
    print("  high_freq_ratio: 高频能量比，越高说明细节越丰富")
    print("=" * 60)


if __name__ == '__main__':
    main()
