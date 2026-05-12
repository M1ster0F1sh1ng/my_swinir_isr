"""
修复版数据集 — 基于原始 FolderDataset，修复关键问题

改进点（最小必要变更）：
1. 修复预计算裁剪 Bug：改为 __getitem__ 中真随机裁剪
2. 添加 Real-ESRGAN 退化模型（用 PIL 做真实 JPEG）
3. 只保留安全的数据增强（翻转、90°旋转）
4. 去掉有害的颜色抖动和随机缩放
"""

import os
import re
import torch
import numpy as np
import random
from io import BytesIO
from torch.utils.data import Dataset
from PIL import Image, ImageFilter
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

# ═══════════════════════════════════════════════════════════════════════════════
# 基础工具
# ═══════════════════════════════════════════════════════════════════════════════

VERBOSE = True

def set_verbose(verbose):
    global VERBOSE
    VERBOSE = verbose

rules = r'\d{4}'

def index_get(file_name):
    name_ = re.match(rules, file_name)[0] + '.png'
    return name_


# ═══════════════════════════════════════════════════════════════════════════════
# Real-ESRGAN 退化模型（基于 PIL，JPEG 是真实的）
# ═══════════════════════════════════════════════════════════════════════════════

class RealESRGANDegradation:
    """
    Real-ESRGAN 二阶退化模型

    关键：JPEG 压缩使用 PIL 的 save/load，是真实的 DCT 域量化，
    不是空间域噪声近似。
    """

    def __init__(self, scale=2, mode='second_order'):
        """
        Args:
            scale: 最终下采样倍率
            mode: 'clean'（仅双三次）, 'first_order', 'second_order'
        """
        self.scale = scale
        self.mode = mode

    def _random_blur(self, img_pil):
        """随机高斯模糊"""
        radius = random.uniform(0.5, 3.5)
        return img_pil.filter(ImageFilter.GaussianBlur(radius=radius))

    def _random_motion_blur(self, img_pil):
        """随机运动模糊（模拟真实相机抖动）"""
        size = random.randint(3, 15)
        if size % 2 == 0:
            size += 1
        kernel = np.zeros((size, size))
        # 简单线性运动模糊
        if random.random() < 0.5:
            kernel[size // 2, :] = 1.0 / size  # 水平
        else:
            kernel[:, size // 2] = 1.0 / size  # 垂直
        img_np = np.array(img_pil)
        from scipy.ndimage import convolve
        blurred = convolve(img_np, kernel[:, :, None], mode='reflect')
        return Image.fromarray(np.clip(blurred, 0, 255).astype(np.uint8))

    def _random_sinc_blur(self, img_pil):
        """
        Sinc 滤波（模拟相机抗混叠滤波器的振铃 artifacts）
        产生比高斯模糊更明显的振铃效应
        """
        size = random.choice([7, 9, 11, 13])
        # 生成 sinc 核: h(x) = sin(pi*x) / (pi*x)
        x = np.arange(-size // 2 + 1, size // 2 + 1)
        x = np.where(x == 0, 1e-8, x)
        h = np.sinc(x)  # sin(pi*x)/(pi*x)
        kernel_2d = np.outer(h, h)
        kernel_2d = kernel_2d / kernel_2d.sum()

        img_np = np.array(img_pil).astype(np.float32)
        from scipy.ndimage import convolve
        blurred = convolve(img_np, kernel_2d[:, :, None], mode='reflect')
        return Image.fromarray(np.clip(blurred, 0, 255).astype(np.uint8))

    def _random_color_degradation(self, img_pil):
        """
        颜色退化：亮度、对比度、饱和度、色偏
        模拟不同光照条件和相机白平衡误差
        """
        img_np = np.array(img_pil).astype(np.float32)

        # 亮度偏移
        if random.random() < 0.5:
            brightness = random.uniform(-30, 30)
            img_np = img_np + brightness

        # 对比度变化
        if random.random() < 0.5:
            contrast = random.uniform(0.7, 1.4)
            mean = img_np.mean()
            img_np = (img_np - mean) * contrast + mean

        # 饱和度变化
        if random.random() < 0.5:
            saturation = random.uniform(0.5, 1.5)
            gray = img_np.mean(axis=2, keepdims=True)
            img_np = gray + (img_np - gray) * saturation

        # 色偏（RGB 通道独立增益）
        if random.random() < 0.3:
            r_gain = random.uniform(0.85, 1.15)
            g_gain = random.uniform(0.85, 1.15)
            b_gain = random.uniform(0.85, 1.15)
            img_np[:, :, 0] *= r_gain
            img_np[:, :, 1] *= g_gain
            img_np[:, :, 2] *= b_gain

        return Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8))

    def _random_poisson_noise(self, img_pil):
        """
        泊松噪声（模拟光子计数噪声，低光场景常见）
        信号依赖型噪声：噪声强度与信号强度成正比
        """
        img_np = np.array(img_pil).astype(np.float32)
        # 泊松噪声：对每个像素值进行泊松采样
        # scale 控制噪声水平，越高噪声越强
        scale = random.uniform(5.0, 20.0)
        # 先放大到光子计数域，加泊松噪声，再缩回来
        img_scaled = img_np * scale
        noise = np.random.poisson(img_scaled) / scale - img_np
        img_noisy = img_np + noise * random.uniform(0.5, 1.5)
        return Image.fromarray(np.clip(img_noisy, 0, 255).astype(np.uint8))

    def _random_noise(self, img_pil):
        """随机高斯噪声"""
        img_np = np.array(img_pil).astype(np.float32)
        noise_level = random.uniform(5, 25)  # 5-25 像素级噪声（增强）
        noise = np.random.normal(0, noise_level, img_np.shape)
        img_noisy = np.clip(img_np + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(img_noisy)

    def _random_jpeg(self, img_pil):
        """真实 JPEG 压缩（使用 PIL 的 DCT 域量化）"""
        quality = random.randint(10, 95)  # 下限降到 10（增强）
        buffer = BytesIO()
        img_pil.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        return Image.open(buffer)

    def _random_resize(self, img_pil, min_scale=0.3, max_scale=1.0):
        """随机缩放（用于模拟不同分辨率传感器）"""
        w, h = img_pil.size
        scale = random.uniform(min_scale, max_scale)
        new_w, new_h = max(int(w * scale), 8), max(int(h * scale), 8)
        return img_pil.resize((new_w, new_h), Image.BICUBIC)

    def degrade(self, hr_pil):
        """
        应用退化管道

        Args:
            hr_pil: PIL Image (RGB)
        Returns:
            lr_pil: PIL Image (RGB) — 低分辨率退化图像
        """
        if self.mode == 'clean':
            w, h = hr_pil.size
            return hr_pil.resize((w // self.scale, h // self.scale), Image.BICUBIC)

        elif self.mode == 'first_order':
            # 一阶：模糊 -> 下采样 -> 噪声
            img = self._random_blur(hr_pil)
            w, h = img.size
            img = img.resize((w // self.scale, h // self.scale), Image.BICUBIC)
            img = self._random_noise(img)
            return img

        elif self.mode == 'second_order':
            # 二阶退化（Real-ESRGAN）

            # === 第一阶退化 ===
            if random.random() < 0.9:
                hr_pil = self._random_blur(hr_pil)

            # Sinc 滤波（产生振铃 artifacts，模拟相机 AA 滤波）
            if random.random() < 0.3:
                hr_pil = self._random_sinc_blur(hr_pil)

            # 运动模糊（真实世界常见）
            if random.random() < 0.4:
                hr_pil = self._random_motion_blur(hr_pil)

            # 颜色退化（模拟光照和白平衡变化）
            if random.random() < 0.5:
                hr_pil = self._random_color_degradation(hr_pil)

            # 随机缩放（模拟不同相机传感器尺寸）
            if random.random() < 0.5:
                hr_pil = self._random_resize(hr_pil, min_scale=0.3, max_scale=0.9)

            # 加噪声（高斯 + 泊松混合）
            if random.random() < 0.7:
                hr_pil = self._random_noise(hr_pil)
            if random.random() < 0.3:
                hr_pil = self._random_poisson_noise(hr_pil)

            # === 第二阶退化 ===
            if random.random() < 0.6:
                hr_pil = self._random_blur(hr_pil)

            if random.random() < 0.3:
                hr_pil = self._random_motion_blur(hr_pil)

            # JPEG 压缩（真实的 DCT 域量化）
            if random.random() < 0.7:
                hr_pil = self._random_jpeg(hr_pil)

            # 最终下采样
            w, h = hr_pil.size
            lr_pil = hr_pil.resize((w // self.scale, h // self.scale), Image.BICUBIC)

            # 最终噪声（传感器噪声：高斯 + 泊松）
            if random.random() < 0.5:
                lr_pil = self._random_noise(lr_pil)
            if random.random() < 0.2:
                lr_pil = self._random_poisson_noise(lr_pil)

            return lr_pil

        # 默认
        w, h = hr_pil.size
        return hr_pil.resize((w // self.scale, h // self.scale), Image.BICUBIC)


# ═══════════════════════════════════════════════════════════════════════════════
# 安全的数据增强（不会破坏 LR-HR 空间对应关系）
# ═══════════════════════════════════════════════════════════════════════════════

def safe_augment(lr_tensor, hr_tensor):
    """
    对 LR-HR 对应用安全的数据增强

    只使用空间几何变换，保证 LR 和 HR 的像素对应关系不变。

    注意：torch.flip / torch.rot90 返回的是共享存储的视图(view)，
    DataLoader 的 default_collate 需要 resizable storage，
    因此必须 .contiguous() 生成独立存储的 tensor。

    Args:
        lr_tensor: [C, H, W]
        hr_tensor: [C, H*scale, W*scale]
    Returns:
        aug_lr, aug_hr
    """
    # 随机水平翻转
    if random.random() < 0.5:
        lr_tensor = torch.flip(lr_tensor, dims=[2]).contiguous()
        hr_tensor = torch.flip(hr_tensor, dims=[2]).contiguous()

    # 随机垂直翻转
    if random.random() < 0.5:
        lr_tensor = torch.flip(lr_tensor, dims=[1]).contiguous()
        hr_tensor = torch.flip(hr_tensor, dims=[1]).contiguous()

    # 随机旋转 90°（90/180/270，保持像素网格对齐）
    if random.random() < 0.3:
        k = random.choice([1, 2, 3])
        lr_tensor = torch.rot90(lr_tensor, k, dims=[1, 2]).contiguous()
        hr_tensor = torch.rot90(hr_tensor, k, dims=[1, 2]).contiguous()

    return lr_tensor, hr_tensor


# ═══════════════════════════════════════════════════════════════════════════════
# 修复版训练数据集
# ═══════════════════════════════════════════════════════════════════════════════

class FixedFolderDataset(Dataset):
    """
    修复版 FolderDataset

    修复：
    1. 预计算裁剪 Bug → __getitem__ 中真随机裁剪
    2. 退化模型 → Real-ESRGAN 二阶退化
    3. 增强 → 仅安全的空间变换
    """

    def __init__(self, folder_path, scale=2, patch_size=64, pre_crop=True,
                 degradation='second_order', augment=True):
        super().__init__()
        self.folder_path = folder_path
        self.scale = scale
        self.patch_size = patch_size
        self.pre_crop = pre_crop
        self.augment = augment

        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp')

        # ═══════════════════════════════════════════════════════════════
        # 优先从 HR/ 子目录加载（防止 eval 目录混入预生成的 LR 变体）
        # ═══════════════════════════════════════════════════════════════
        hr_folder = os.path.join(folder_path, 'HR')
        if os.path.isdir(hr_folder):
            search_dir = hr_folder
        else:
            search_dir = folder_path

        self.image_files = [f for f in os.listdir(search_dir)
                            if f.lower().endswith(valid_extensions)]
        self.image_files.sort()

        if len(self.image_files) == 0:
            raise ValueError(f"在 {search_dir} 中没有找到图片文件！")

        if VERBOSE:
            print(f"加载 {len(self.image_files)} 张图片从 {folder_path}")
            if search_dir != folder_path:
                print(f"  (从 HR/ 子目录加载)")
            print(f"  退化: {degradation}, 增强: {augment}")

        self.degradation = RealESRGANDegradation(scale=scale, mode=degradation)
        self.to_tensor = transforms.ToTensor()

    def _get_hr_path(self, img_name):
        """获取 HR 图片路径"""
        hr_folder = os.path.join(self.folder_path, 'HR')
        image_file_name = index_get(img_name)

        if os.path.exists(hr_folder):
            return os.path.join(hr_folder, image_file_name)
        else:
            return os.path.join(self.folder_path, image_file_name)

    def __len__(self):
        # 增强时增加采样次数
        if self.augment:
            return len(self.image_files) * 4
        return len(self.image_files)

    def __getitem__(self, idx):
        """获取训练样本 — 真正的随机裁剪"""
        img_idx = idx % len(self.image_files)
        img_name = self.image_files[img_idx]
        hr_path = self._get_hr_path(img_name)

        # 加载 HR 图像
        hr_image = Image.open(hr_path).convert('RGB')

        if self.pre_crop:
            w, h = hr_image.size

            # === 修复：真正的随机裁剪（每次调用都重新随机）===
            if w > self.patch_size and h > self.patch_size:
                x = np.random.randint(0, w - self.patch_size + 1)
                y = np.random.randint(0, h - self.patch_size + 1)
                hr_patch = hr_image.crop((x, y, x + self.patch_size, y + self.patch_size))
            else:
                hr_patch = hr_image.resize((self.patch_size, self.patch_size), Image.BICUBIC)

            # 应用退化模型生成 LR
            lr_patch = self.degradation.degrade(hr_patch)

            # 关键修复：退化中的随机缩放（_random_resize）会改变图像尺寸，
            # 导致同一个 batch 里 LR tensor 尺寸不一致，无法 stack。
            # 必须将 LR resize 回标准尺寸 patch_size // scale，
            # 退化效果（模糊、噪声、JPEG 伪影）仍然保留。
            lr_w, lr_h = self.patch_size // self.scale, self.patch_size // self.scale
            if lr_patch.size != (lr_w, lr_h):
                lr_patch = lr_patch.resize((lr_w, lr_h), Image.BICUBIC)

            # 转为 Tensor
            lr_tensor = self.to_tensor(lr_patch)
            hr_tensor = self.to_tensor(hr_patch)

            # 安全增强
            if self.augment:
                lr_tensor, hr_tensor = safe_augment(lr_tensor, hr_tensor)

            # 必须克隆！transforms.ToTensor() 内部用 torch.from_numpy() 创建的
            # tensor 底层 storage 挂载在 numpy 数组上，不可 resize，
            # DataLoader 的 default_collate 会 resize storage 来拼接 batch，
            # 不 clone 就会报 RuntimeError: Trying to resize storage that is not resizable
            return lr_tensor.clone(), hr_tensor.clone()

        else:
            # 验证模式
            w, h = hr_image.size
            lr_image = self.degradation.degrade(hr_image)
            return self.to_tensor(lr_image).clone(), self.to_tensor(hr_image).clone()


# ═══════════════════════════════════════════════════════════════════════════════
# 验证数据集（无增强，直接从 LR 文件夹加载）
# ═══════════════════════════════════════════════════════════════════════════════

class FixedValidationDataset(Dataset):
    """验证数据集 — 分别加载 LR 和 HR"""

    def __init__(self, folder_path, scale=2):
        super().__init__()
        self.scale = scale

        # 寻找 HR 文件夹
        hr_folder = os.path.join(folder_path, 'HR')
        if not os.path.exists(hr_folder):
            hr_folder = folder_path

        # 寻找 LR 文件夹
        lr_folder = os.path.join(folder_path, 'LR', f'X{scale}')
        if not os.path.exists(lr_folder):
            lr_folder = os.path.join(folder_path, 'LR')
        if not os.path.exists(lr_folder):
            lr_folder = None  # 将从 HR 下采样

        self.lr_folder = lr_folder
        self.hr_folder = hr_folder

        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
        self.image_files = [f for f in os.listdir(hr_folder)
                            if f.lower().endswith(valid_extensions)]
        self.image_files.sort()

        if VERBOSE:
            print(f"验证集: {len(self.image_files)} 张")

        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]

        # 加载 HR
        hr = Image.open(os.path.join(self.hr_folder, img_name)).convert('RGB')
        hr_tensor = self.to_tensor(hr)

        # 加载 LR 或从 HR 下采样
        if self.lr_folder:
            lr_name = img_name
            lr_path = os.path.join(self.lr_folder, lr_name)
            if os.path.exists(lr_path):
                lr = Image.open(lr_path).convert('RGB')
                lr_tensor = self.to_tensor(lr)
            else:
                w, h = hr.size
                lr = hr.resize((w // self.scale, h // self.scale), Image.BICUBIC)
                lr_tensor = self.to_tensor(lr)
        else:
            w, h = hr.size
            lr = hr.resize((w // self.scale, h // self.scale), Image.BICUBIC)
            lr_tensor = self.to_tensor(lr)

        # 必须克隆！与 FixedFolderDataset 同理，ToTensor() 产生的 tensor
        # 底层 storage 不可 resize，DataLoader collate 会报错
        return lr_tensor.clone(), hr_tensor.clone()


# ═══════════════════════════════════════════════════════════════════════════════
# 保持与原代码兼容的接口
# ═══════════════════════════════════════════════════════════════════════════════

class DegradedValidationDataset(Dataset):
    """
    退化验证数据集 — 对 HR 应用退化生成 LR

    与 FixedValidationDataset 的区别：
    - FixedValidationDataset：从 LR/HR 文件夹加载 clean 对
    - DegradedValidationDataset：从 HR 文件夹加载，实时应用退化生成 LR

    适用于 Phase 2/3 的退化训练验证，确保验证集分布与训练集一致。
    """

    def __init__(self, folder_path, scale=2, degradation='second_order'):
        super().__init__()
        self.scale = scale
        self.degradation = RealESRGANDegradation(scale=scale, mode=degradation)

        # 寻找 HR 文件夹
        hr_folder = os.path.join(folder_path, 'HR')
        if not os.path.exists(hr_folder):
            hr_folder = folder_path

        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
        self.image_files = [f for f in os.listdir(hr_folder)
                            if f.lower().endswith(valid_extensions)]
        self.image_files.sort()

        if len(self.image_files) == 0:
            raise ValueError(f"在 {hr_folder} 中没有找到图片文件！")

        self.hr_folder = hr_folder

        if VERBOSE:
            print(f"退化验证集: {len(self.image_files)} 张, 退化: {degradation}")

        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        hr_path = os.path.join(self.hr_folder, img_name)

        # 加载 HR
        hr = Image.open(hr_path).convert('RGB')

        # 应用退化生成 LR（与训练时一致的退化分布）
        lr = self.degradation.degrade(hr)

        # 确保 LR 尺寸正确（退化中的随机缩放可能改变尺寸）
        w, h = hr.size
        expected_lr_size = (w // self.scale, h // self.scale)
        if lr.size != expected_lr_size:
            lr = lr.resize(expected_lr_size, Image.BICUBIC)

        lr_tensor = self.to_tensor(lr)
        hr_tensor = self.to_tensor(hr)

        return lr_tensor.clone(), hr_tensor.clone()


class FolderDataset(FixedFolderDataset):
    """兼容原接口的别名"""
    pass