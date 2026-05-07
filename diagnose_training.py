"""
训练诊断脚本 — 验证模型参数是否在更新
Usage:
    python diagnose_training.py --checkpoint ./checkpoints/phase1_pixel/0001/best.pth --pretrained ./pre/001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth
"""

import argparse
import torch
from network_swinir import SwinIR


def load_model(path, patch_norm=True):
    model = SwinIR(
        upscale=2, img_size=64, patch_size=1, in_chans=3,
        embed_dim=180, depths=[6, 6, 6, 6, 6, 6], num_heads=[6, 6, 6, 6, 6, 6],
        window_size=8, mlp_ratio=2., upsampler='pixelshuffle',
        resi_connection='1conv', patch_norm=patch_norm
    )
    ckpt = torch.load(path, map_location='cpu')
    if 'params' in ckpt:
        state_dict = ckpt['params']
    elif 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    elif 'model' in ckpt:
        state_dict = ckpt['model']
    else:
        state_dict = ckpt
    
    # 去除 DDP 的 module. 前缀
    new_state = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]
        new_state[k] = v
    
    model.load_state_dict(new_state, strict=False)
    return model


def compare_models(model_a, model_b, name_a='A', name_b='B'):
    """对比两个模型的参数差异"""
    diff_count = 0
    total_diff = 0.0
    max_diff = 0.0
    
    for (n1, p1), (n2, p2) in zip(model_a.named_parameters(), model_b.named_parameters()):
        assert n1 == n2, f"参数名不匹配: {n1} vs {n2}"
        diff = (p1 - p2).abs()
        max_d = diff.max().item()
        mean_d = diff.mean().item()
        
        if max_d > 1e-6:
            diff_count += 1
            total_diff += mean_d
            max_diff = max(max_diff, max_d)
            if diff_count <= 5:
                print(f"  [{n1}] max_diff={max_d:.6f}, mean_diff={mean_d:.6f}")
    
    print(f"\n总计: {diff_count}/{sum(1 for _ in model_a.named_parameters())} 个参数有变化")
    print(f"平均差异: {total_diff/max(diff_count,1):.8f}")
    print(f"最大差异: {max_diff:.6f}")
    
    if diff_count == 0:
        print("\n⚠️ 警告: 两个模型参数完全相同！训练没有更新权重。")
    elif max_diff < 1e-5:
        print("\n⚠️ 警告: 参数变化极小 (<1e-5)，训练效果极弱。")
    else:
        print("\n✅ 模型参数有正常变化，训练在生效。")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='当前训出来的 best.pth')
    parser.add_argument('--pretrained', type=str, required=True, help='原始预训练权重')
    args = parser.parse_args()
    
    print("加载预训练权重...")
    pretrained = load_model(args.pretrained)
    
    print("加载当前 checkpoint...")
    checkpoint = load_model(args.checkpoint)
    
    print("\n对比预训练权重 vs 当前 best.pth:")
    compare_models(pretrained, checkpoint, 'pretrained', 'checkpoint')


if __name__ == '__main__':
    main()
