"""
SwinIR 超分辨率推理脚本
基于 network_swinir.py (Official SwinIR)

Usage:
    # 单张图片
    python infer_swinir.py \
        --input ./input.png \
        --weights ./checkpoints/phase3_perceptual/full_swinir_fixed_x2/0001/checkpoint_best.pth \
        --scale 2 \
        --output ./output/sr.png

    # 文件夹批量处理
    python infer_swinir.py \
        --input ./input_dir/ \
        --weights ./pre/001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth \
        --scale 2 \
        --output ./output_dir/ \
        --self-ensemble

    # 使用特定设备
    python infer_swinir.py --input ./input.png --weights ./best.pth --device cuda
"""

import argparse
import os
import sys
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import PIL.Image as pil_image
from tqdm import tqdm

from network_swinir import SwinIR


def detect_config_from_state_dict(state_dict, force_scale=None):
    """
    从权重 state_dict 自动检测模型配置
    """
    # embed_dim
    embed_dim = state_dict['conv_first.weight'].shape[0] if 'conv_first.weight' in state_dict else 180

    # window_size
    window_size = 8
    for key in state_dict.keys():
        if 'relative_position_bias_table' in key:
            table_size = state_dict[key].shape[0]
            size = int(math.sqrt(table_size))
            window_size = (size + 1) // 2
            break

    # num_heads
    num_heads = 6
    for key in state_dict.keys():
        if 'relative_position_bias_table' in key:
            num_heads = state_dict[key].shape[1]
            break

    # scale
    scale = force_scale or 2
    if 'upsample.0.weight' in state_dict:
        upsample_out = state_dict['upsample.0.weight'].shape[0]
        scale_squared = upsample_out // 64  # upsample 前是 64 channel (num_feat)
        detected_scale = int(math.sqrt(scale_squared)) if scale_squared > 0 else 2
        if force_scale and force_scale != detected_scale:
            print(f"[警告] 强制使用 scale={force_scale}, 但权重训练时是 scale={detected_scale}")
        scale = detected_scale

    # depths
    layer_indices = set()
    for key in state_dict.keys():
        if key.startswith('layers.') and 'blocks' in key:
            parts = key.split('.')
            if len(parts) >= 4 and parts[0] == 'layers':
                try:
                    layer_indices.add(int(parts[1]))
                except ValueError:
                    pass

    num_layers = len(layer_indices)
    blocks_per_layer = {}
    for key in state_dict.keys():
        if key.startswith('layers.') and 'blocks.' in key:
            parts = key.split('.')
            # 官方格式: layers.X.residual_group.blocks.Y.norm1.weight
            # 或: layers.X.blocks.Y.norm1.weight
            if len(parts) >= 4 and parts[0] == 'layers':
                layer_idx = int(parts[1])
                # 找到 blocks 后面的索引
                for i, p in enumerate(parts):
                    if p == 'blocks' and i + 1 < len(parts):
                        try:
                            block_idx = int(parts[i + 1])
                            if layer_idx not in blocks_per_layer:
                                blocks_per_layer[layer_idx] = set()
                            blocks_per_layer[layer_idx].add(block_idx)
                        except ValueError:
                            pass
                        break

    depths = [max(blocks_per_layer.get(i, [0])) + 1 for i in sorted(layer_indices)] \
        if blocks_per_layer else [6] * num_layers

    return {
        'embed_dim': embed_dim,
        'depths': depths,
        'num_heads': [num_heads] * num_layers,
        'window_size': window_size,
        'scale': scale,
        'num_layers': num_layers,
    }


def create_model(weights_path, force_scale=None):
    """创建模型并加载权重"""
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"权重文件不存在: {weights_path}")

    print(f"[加载权重] {weights_path}")
    checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)

    # 提取 state_dict
    if 'params' in checkpoint:
        state_dict = checkpoint['params']
        print("[格式] params dict")
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
        print("[格式] checkpoint (含 model 键)")
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        print("[格式] state_dict")
    else:
        state_dict = checkpoint
        print("[格式] raw state_dict")

    # 去除 DDP 前缀
    new_state = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]
        new_state[k] = v
    state_dict = new_state

    cfg = detect_config_from_state_dict(state_dict, force_scale=force_scale)

    print(f"[配置] embed_dim={cfg['embed_dim']}, depths={cfg['depths']}, "
          f"window_size={cfg['window_size']}, num_heads={cfg['num_heads'][0]}, "
          f"scale={cfg['scale']}, num_layers={cfg['num_layers']}")

    model = SwinIR(
        upscale=cfg['scale'],
        img_size=64,
        patch_size=1,
        in_chans=3,
        embed_dim=cfg['embed_dim'],
        depths=cfg['depths'],
        num_heads=cfg['num_heads'],
        window_size=cfg['window_size'],
        mlp_ratio=4.,
        upsampler='pixelshuffle',
        resi_connection='1conv',
    )

    # 加载权重
    try:
        missing, unexpected = model.load_state_dict(state_dict, strict=True)
        print("[加载] 严格匹配成功")
    except RuntimeError:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[加载] 缺失参数: {len(missing)} 个")
            for k in missing[:5]:
                print(f"  - {k}")
        if unexpected:
            print(f"[加载] 意外参数: {len(unexpected)} 个")
            for k in unexpected[:5]:
                print(f"  - {k}")
        print("[加载] 非严格模式成功")

    return model, cfg


def augment_img(img, mode=0):
    """图像几何增强"""
    if mode == 0:
        return img
    elif mode == 1:
        return img[:, ::-1, :]
    elif mode == 2:
        return img[::-1, :, :]
    elif mode == 3:
        return img[::-1, ::-1, :]
    elif mode == 4:
        return img.transpose(1, 0, 2)
    elif mode == 5:
        return img[:, ::-1, :].transpose(1, 0, 2)
    elif mode == 6:
        return img[::-1, :, :].transpose(1, 0, 2)
    elif mode == 7:
        return img[::-1, ::-1, :].transpose(1, 0, 2)


def reverse_augment_img(img, mode=0):
    """反变换"""
    if mode == 0:
        return img
    elif mode == 1:
        return img[:, ::-1, :]
    elif mode == 2:
        return img[::-1, :, :]
    elif mode == 3:
        return img[::-1, ::-1, :]
    elif mode == 4:
        return img.transpose(1, 0, 2)
    elif mode == 5:
        return img.transpose(1, 0, 2)[:, ::-1, :]
    elif mode == 6:
        return img.transpose(1, 0, 2)[::-1, :, :]
    elif mode == 7:
        return img.transpose(1, 0, 2)[::-1, ::-1, :]


def preprocess_image(image_path, device):
    """读取并预处理图像"""
    img = pil_image.open(image_path).convert('RGB')
    img = np.array(img).astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
    return img


def postprocess_tensor(tensor):
    """将模型输出转为 PIL Image"""
    tensor = tensor.clamp(0.0, 1.0)
    img = tensor.mul(255.0).byte().permute(0, 2, 3, 1).cpu().numpy()
    return pil_image.fromarray(img[0])


def inference_single(model, img_tensor, window_size=8):
    """单张推理"""
    _, _, h, w = img_tensor.size()

    # pad 到 window_size 倍数
    mod_pad_h = (window_size - h % window_size) % window_size
    mod_pad_w = (window_size - w % window_size) % window_size
    img_tensor = torch.nn.functional.pad(img_tensor, (0, mod_pad_w, 0, mod_pad_h), 'reflect')

    with torch.no_grad():
        output = model(img_tensor)

    # 去除 pad
    output = output[:, :, :h * model.upscale, :w * model.upscale]
    return output


def inference_selfensemble(model, img_tensor, window_size=8):
    """Self-Ensemble x8 推理"""
    outputs = []
    for mode in range(8):
        # numpy augment
        img_np = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        aug_np = augment_img(img_np, mode=mode)
        aug_tensor = torch.from_numpy(aug_np).permute(2, 0, 1).unsqueeze(0).to(img_tensor.device)

        pred = inference_single(model, aug_tensor, window_size=window_size)

        # reverse augment
        pred_np = pred.squeeze(0).permute(1, 2, 0).cpu().numpy()
        rev_np = reverse_augment_img(pred_np, mode=mode)
        rev_tensor = torch.from_numpy(rev_np).permute(2, 0, 1).unsqueeze(0).to(img_tensor.device)
        outputs.append(rev_tensor)

    output = torch.stack(outputs).mean(dim=0)
    return output


def process_image(model, image_path, output_path, device, window_size=8, use_selfensemble=False):
    """处理单张图片"""
    print(f"[处理] {image_path}")
    img_tensor = preprocess_image(image_path, device)
    _, _, h, w = img_tensor.shape
    print(f"  输入尺寸: {w}x{h}")

    start = time.time()
    if use_selfensemble:
        output = inference_selfensemble(model, img_tensor, window_size=window_size)
        print(f"  使用 Self-Ensemble x8")
    else:
        output = inference_single(model, img_tensor, window_size=window_size)

    elapsed = time.time() - start
    _, _, oh, ow = output.shape
    print(f"  输出尺寸: {ow}x{oh} | 耗时: {elapsed:.2f}s")

    result = postprocess_tensor(output)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(out_path)
    print(f"  ✓ 已保存: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description='SwinIR 超分辨率推理')
    parser.add_argument('--input', type=str, required=True, help='输入图像路径或文件夹')
    parser.add_argument('--weights', type=str, required=True, help='模型权重路径 (.pth)')
    parser.add_argument('--scale', type=int, default=None, help='强制指定放大倍数 (2/3/4)')
    parser.add_argument('--output', type=str, required=True, help='输出图像路径或文件夹')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu', 'mps'],
                        help='计算设备')
    parser.add_argument('--self-ensemble', action='store_true', help='启用 Self-Ensemble x8 (提升约 0.1~0.3 dB)')
    parser.add_argument('--window-size', type=int, default=8, help='SwinIR window size')

    args = parser.parse_args()

    # 设备选择
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            print("[设备] CUDA")
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
            print("[设备] Apple Silicon (MPS)")
        else:
            device = torch.device('cpu')
            print("[设备] CPU")
    else:
        device = torch.device(args.device)
        print(f"[设备] {args.device}")

    cudnn.benchmark = True

    # 创建模型
    model, cfg = create_model(args.weights, force_scale=args.scale)
    model = model.to(device)
    model.eval()

    # 判断输入是文件还是文件夹
    input_path = Path(args.input)
    output_path = Path(args.output)

    if input_path.is_file():
        # 单张图片
        if output_path.is_dir() or str(output_path).endswith('/') or str(output_path).endswith('\\'):
            output_path.mkdir(parents=True, exist_ok=True)
            output_file = output_path / f"{input_path.stem}_sr_x{cfg['scale']}.png"
        else:
            output_file = output_path
        process_image(model, input_path, output_file, device,
                      window_size=args.window_size, use_selfensemble=args.self_ensemble)

    elif input_path.is_dir():
        # 文件夹批量处理
        output_path.mkdir(parents=True, exist_ok=True)
        image_files = sorted([
            f for f in input_path.iterdir()
            if f.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        ])

        if not image_files:
            print(f"[错误] 文件夹中没有图片: {input_path}")
            return 1

        print(f"[批量处理] 共 {len(image_files)} 张图片 -> {output_path}")
        for img_file in tqdm(image_files, desc='推理中'):
            out_file = output_path / f"{img_file.stem}_sr_x{cfg['scale']}.png"
            try:
                process_image(model, img_file, out_file, device,
                              window_size=args.window_size, use_selfensemble=args.self_ensemble)
            except Exception as e:
                print(f"[错误] 处理 {img_file.name} 失败: {e}")

    else:
        print(f"[错误] 输入路径不存在: {input_path}")
        return 1

    print("\n✓ 全部完成")
    return 0


if __name__ == '__main__':
    exit(main())
