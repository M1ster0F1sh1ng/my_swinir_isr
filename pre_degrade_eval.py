"""
预先生成验证集的退化 LR 图像
运行一次即可，之后验证时 degradation 改为 'clean'
"""
import os
import sys
from tqdm import tqdm
from PIL import Image
from cloud_dataset import RealESRGANDegradation


def pre_degrade_eval_set(folder_path, scale=2, degradation='second_order'):
    """
    对验证集进行预退化：
    - 从 folder_path/HR/ 读取原图
    - 应用退化生成 LR
    - 保存到 folder_path/LR/X{scale}/
    """
    hr_folder = os.path.join(folder_path, 'HR')
    if not os.path.isdir(hr_folder):
        print(f"[跳过] {folder_path} 下没有 HR/ 子目录")
        return

    lr_folder = os.path.join(folder_path, 'LR', f'X{scale}')
    os.makedirs(lr_folder, exist_ok=True)

    image_files = [f for f in os.listdir(hr_folder)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    image_files.sort()

    if len(image_files) == 0:
        print(f"[跳过] {hr_folder} 中没有图片")
        return

    degradator = RealESRGANDegradation(scale=scale, mode=degradation)

    print(f"处理: {folder_path} ({len(image_files)} 张)")
    for img_name in tqdm(image_files, desc=f"退化 {os.path.basename(folder_path)}"):
        hr_path = os.path.join(hr_folder, img_name)
        lr_path = os.path.join(lr_folder, img_name)

        # 已存在则跳过
        if os.path.exists(lr_path):
            continue

        hr = Image.open(hr_path).convert('RGB')
        lr = degradator.degrade(hr)
        lr.save(lr_path)

    print(f"完成: LR 保存至 {lr_folder}")


if __name__ == '__main__':
    eval_sets = [
        '/root/autodl-tmp/DIV2K/DIV2K_eval_set/x2',
        '/root/autodl-tmp/Set14',
        '/root/autodl-tmp/urban',
        '/root/autodl-tmp/bsd300/test',
    ]

    for path in eval_sets:
        if os.path.exists(path):
            pre_degrade_eval_set(path, scale=2, degradation='second_order')
        else:
            print(f"[不存在] {path}")

    print("\n全部完成！请将 JSON 中的 degradation 改为 'clean'，并确保 eval-file 路径正确。")
