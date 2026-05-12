"""
PSNR 诊断脚本 — 验证 bilinear/bicubic 基线，排查测试方法问题

Usage:
    python diagnose_psnr.py --hr <HR图片路径> --scale 2
    
如果 HR 图不存在，脚本会尝试从 LR 目录推断。
"""

import argparse
import os
import numpy as np
from PIL import Image


def calc_psnr(img1, img2, max_val=255.0):
    """计算 PSNR"""
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse < 1e-10:
        return float('inf')
    return 20 * np.log10(max_val / np.sqrt(mse))


def downsample_bicubic(img, scale):
    """双三次下采样"""
    w, h = img.size
    return img.resize((w // scale, h // scale), Image.BICUBIC)


def upsample(img, scale, method):
    """上采样"""
    w, h = img.size
    if method == 'bilinear':
        return img.resize((w * scale, h * scale), Image.BILINEAR)
    elif method == 'bicubic':
        return img.resize((w * scale, h * scale), Image.BICUBIC)
    elif method == 'nearest':
        return img.resize((w * scale, h * scale), Image.NEAREST)
    else:
        raise ValueError(f"Unknown method: {method}")


def main():
    parser = argparse.ArgumentParser(description='PSNR 基线诊断')
    parser.add_argument('--hr', type=str, required=True, help='HR 图像路径')
    parser.add_argument('--scale', type=int, default=2, help='放大倍数')
    parser.add_argument('--border', type=int, default=0, help='裁剪边界像素')
    args = parser.parse_args()

    # 加载 HR
    hr = Image.open(args.hr).convert('RGB')
    hr_np = np.array(hr)
    print(f"HR 图像: {hr.size}")

    # 下采样到 LR
    lr = downsample_bicubic(hr, args.scale)
    lr_np = np.array(lr)
    print(f"LR 图像 (bicubic 下采样): {lr.size}")

    # 上采样方法
    methods = ['bilinear', 'bicubic', 'nearest']

    print("\n" + "=" * 60)
    print("基线 PSNR 测试（越低说明退化越强，超分辨率越难）")
    print("=" * 60)

    for method in methods:
        sr = upsample(lr, args.scale, method)
        sr_np = np.array(sr)

        # 裁剪边界
        b = args.border
        if b > 0:
            hr_crop = hr_np[b:-b, b:-b]
            sr_crop = sr_np[b:-b, b:-b]
        else:
            hr_crop = hr_np
            sr_crop = sr_np

        psnr_val = calc_psnr(hr_crop, sr_crop)
        print(f"  {method:10s} 上采样 vs HR: PSNR = {psnr_val:.2f} dB")

    print("\n" + "=" * 60)
    print("关键判断标准：")
    print("  - 如果 bicubic PSNR > 40 dB：HR 和 LR 几乎相同，测试图像有问题")
    print("  - 如果 bicubic PSNR 在 30-36 dB：正常范围")
    print("  - 如果 bicubic PSNR < 25 dB：图像退化非常严重")
    print("=" * 60)

    # 额外检查：HR 和 LR 是否真的是不同的图
    print("\n额外检查：")
    # 将 LR 再次 bicubic 上采样，和 HR 比较结构相似度
    lr_up = upsample(lr, args.scale, 'bicubic')
    lr_up_np = np.array(lr_up)

    # 简单的结构检查：看看高频是否丢失
    def high_freq_energy(img):
        gray = img.mean(axis=2)
        dx = np.abs(np.diff(gray, axis=1)).mean()
        dy = np.abs(np.diff(gray, axis=0)).mean()
        return (dx + dy) / 2

    print(f"  HR 高频能量: {high_freq_energy(hr_np):.2f}")
    print(f"  LR->bicubic 高频能量: {high_freq_energy(lr_up_np):.2f}")
    print(f"  能量比 (SR/HR): {high_freq_energy(lr_up_np) / high_freq_energy(hr_np):.3f}")
    print("  (正常应该 < 0.8，说明上采样确实丢失了高频)")


if __name__ == '__main__':
    main()
