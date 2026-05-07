import argparse
import os
import copy
import time
import numpy as np
import random

import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import ConcatDataset

from tqdm.auto import tqdm

import cloud_dataset
from swinir_model import SwinIR
from utils import AverageMeter, MultiMetricLoss, ImageMetricsEvaluator

'''
图像文件夹要求：
    ./folder_name   
    ->HR
    ->{0001.png,
        0002.png,
        ...
    }
'''

'''

| Scale  | 建议 HR patch-size | LR 尺寸     | 说明        |
| ------ | ---------------- | --------- | --------- |
| 2×     | 64-128           | 32-64     | 标准配置      |
| 3×     | 96-144           | 32-48     | 需能被3整除    |
| **4×** | **128-256**      | **32-64** | **推荐128** |
| 8×     | 256+             | 32+       | 大patch    |

python run_on_cloud.py \
    --train-file /root/lanyun-tmp/DIV2K/DIV2K_train_set \
     /root/lanyun-tmp/FLickr2K   \
    --eval-file /root/lanyun-tmp/DIV2K/DIV2K_eval_set \
    --outputs-dir ../epoch \
    --scale 2 \
    --num-workers 16\
    --batch-size 64 \
    --patch-size 64\
    --save yes \
    --jump no \
    --save-seq 10 \
    --num-epochs 20  


MD5 (DIV2K_Flickr2K.7z.001) = d338fadb03267b16aa01a17ed8d736eb
MD5 (DIV2K_Flickr2K.7z.002) = b623fcac68a77fe60543fc42f4712e49
MD5 (DIV2K_Flickr2K.7z.003) = 857eaf458bac5fb5d550504ec8ed3222
MD5 (DIV2K_Flickr2K.7z.004) = 68c99636cb2059d2ecf7c53fe8ccda7c
MD5 (DIV2K_Flickr2K.7z.005) = 8daf63724ea559efee42e91dbad81e33
MD5 (DIV2K_Flickr2K.7z.006) = 0cac23d8cb12e74870b98bea7d142af0

    pip install torch h5df numpy tqdm torchvision opencv_python lpips
'''

device = torch.device('cpu')
save_ = False
model = None
jump=False
jump_seq=3
# print(device)
def pre_run():
    global device, save_, args, model, jump,jump_seq


    if not torch.cuda.is_available():
        print(torch.cuda.is_available())
        print("无法使用 cuda，请设置好环境")

    device = torch.device('cuda:0')
    print("using cuda")


    if args.save.lower() == 'yes':
        save_ = True
    else:
        save_ = False

    if args.jump=='yes':
        jump=True
        jump_seq=args.jump_seq
    # 设置随机种子
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    cudnn.benchmark = True  # 自动优化卷积算法
    torch.backends.cuda.matmul.allow_tf32 = True  # 允许 TF32 加速
    torch.backends.cudnn.allow_tf32 = True

    cudnn.benchmark = True
    output_dir = os.path.join(args.outputs_dir, f'{args.model}_swinir_x{args.scale}')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    counts = os.listdir(output_dir)
    args.outputs_dir = os.path.join(output_dir, f'{len(counts) + 1:04d}')
    os.makedirs(args.outputs_dir, exist_ok=True)


    model = SwinIR(scale=args.scale, embed_dim=180, depths=[6, 6, 6, 6]).to(device)
    print("使用完整SwinIR模型")

    # # 编译模型加速（PyTorch 2.0+）
    # if hasattr(torch, 'compile'):
    #     print("使用 torch.compile 加速...")
    #     model = torch.compile(model, mode='max-autotune')

def data_loader_list_return():
    train_dataset = copy.deepcopy(args.train_file)
    eval_dataset = copy.deepcopy(args.eval_file)
    print('加载 train_set')
    for index in range(len(train_dataset)):
        train_dataset[index] = cloud_dataset.FolderDataset(train_dataset[index],
                                                           scale=args.scale,
                                                           patch_size=args.patch_size,
                                                           pre_crop=True)



    train_file_set = ConcatDataset(train_dataset)
    train_loader = DataLoader(
        dataset=train_file_set,
        batch_size=args.batch_size,
        shuffle=True,  # 🔥 关键：所有数据一起打乱
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,  # 丢弃不完整的最后一批，避免BN问题
        persistent_workers=True if args.num_workers > 0 else False  ,# 加速多进程
        prefetch_factor = 4 if args.num_workers > 0 else None,  # 预加载 4 个 batch
        multiprocessing_context = 'spawn' if args.num_workers > 0 else None,
    )


    print('加载 eval_set')
    for index in range(len(eval_dataset)):
        eval_dataset[index] = cloud_dataset.FolderDataset(eval_dataset[index],
                                                          scale=args.scale,
                                                          patch_size=args.patch_size,
                                                          pre_crop=False  # 验证集不裁剪
                                                          )
        eval_dataset[index] = DataLoader(dataset=eval_dataset[index],
                                         batch_size=1,
                                         num_workers=2,  # 验证集也用多进程
                                         pin_memory=True,
                                         persistent_workers=False  # 验证不需要持久化
                                         )

    print(f"使用(FolderDataset)模式")
    return train_loader, eval_dataset


# （有待优化）
# 2.图形化

if __name__ == '__main__':
    # 命令行参数管理
    parser = argparse.ArgumentParser()
    # 必须参数
    parser.add_argument('--train-file', type=str, required=True, nargs='+', help='训练文件夹路径')
    parser.add_argument('--eval-file', type=str, required=True, nargs='+', help='验证数据文件夹路径')
    parser.add_argument('--outputs-dir', type=str, required=True, help='输出文件夹路径')
    parser.add_argument('--valid-dir', type=str, nargs='+', help='测试文件夹路径')
    parser.add_argument('--scale', type=int, default=2, help='缩放倍率')
    parser.add_argument('--model', type=str, default='full', choices=['light', 'full'], help='模型轻量化')
    parser.add_argument('--lr', type=float, default=2e-4, help='学习率')
    parser.add_argument('--batch-size', type=int, default=4, help='批次大小')
    parser.add_argument('--num-epochs', type=int, default=1000, help='epoch 数量，训练次数')
    parser.add_argument('--num-workers', type=int, default=4, help='num of threads')
    parser.add_argument('--patch-size', type=int, default=64, help='训练裁剪块大小')
    parser.add_argument('--seed', type=int, default=123, help='随机种子')
    parser.add_argument('--save', type=str, default='no', choices=['yes', 'no'], help='是否保存每个epoch')
    parser.add_argument('--save-seq', type=int, default=5, help='保存频率')
    parser.add_argument('--jump', type=str, default='no', choices=['yes', 'no'], help='是否跳过一定的验证集')
    parser.add_argument('--jump-seq', type=int, default=3, help='跳过频率')
    parser.add_argument('--grad-accum', type=int, default=1, help='梯度累积步数')
    parser.add_argument('--w-l1', type=float, default=1.0, help='L1损失权重')
    parser.add_argument('--w-ssim', type=float, default=0.5, help='SSIM损失权重')
    parser.add_argument('--w-lpips', type=float, default=0.3, help='LPIPS损失权重')
    args = parser.parse_args()
    # print(args)

    # 预处理
    pre_run()

    # 多指标组合损失函数
    # 权重说明：
    # w_l1=1.0   : L1 损失，基础像素级损失
    # w_ssim=0.5 : SSIM 损失，结构相似性（越高越好，所以用 1-SSIM 作为损失）
    # w_lpips=0.3: LPIPS 损失，感知损失（越低越好）
    criterion = MultiMetricLoss(w_l1=args.w_l1, w_ssim=args.w_ssim, w_lpips=args.w_lpips, device=device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # 学习率调度：预热 + 多步衰减
    warmup_epochs = 5
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 1.0

    scheduler_warmup = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scheduler_step = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200, 400, 600, 800], gamma=0.5)

    train_loader, eval_loaders = data_loader_list_return()

    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_psnr = 0.0

    # 使用自动混合精度
    scaler = torch.amp.GradScaler('cuda', init_scale=2**16, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000)
    autocast = torch.amp.autocast if torch.cuda.is_available() else lambda: torch.enable_grad()

    # 训练
    for epoch in range(args.num_epochs):
        model.train()
        epoch_losses = AverageMeter()

        pbar = tqdm(train_loader, desc=f'Epoch [{epoch + 1}/{args.num_epochs}]')

        for batch_idx, (inputs, labels) in enumerate(pbar):
            # 非阻塞传输到 GPU
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            torch.compiler.cudagraph_mark_step_begin()
            # 自动混合精度训练
            with torch.amp.autocast('cuda'):
                preds = model(inputs)

                # 尺寸对齐
                min_h = min(preds.shape[2], labels.shape[2])
                min_w = min(preds.shape[3], labels.shape[3])
                preds = preds[:, :, :min_h, :min_w]
                labels = labels[:, :, :min_h, :min_w]

                loss, loss_dict = criterion(preds, labels)

            # 梯度累积
            loss = loss / args.grad_accum
            scaler.scale(loss).backward()

            if (batch_idx + 1) % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)  # 更高效地清零梯度

            epoch_losses.update(loss_dict['total'], inputs.size(0))

            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'total': f'{loss_dict["total"]:.4f}',
                'l1': f'{loss_dict["l1"]:.4f}',
                'ssim': f'{loss_dict["ssim"]:.4f}',
                'lpips': f'{loss_dict["lpips"]:.4f}',
                'lr': f'{current_lr:.6f}',
                'gpu': f'{torch.cuda.memory_allocated() / 1024 ** 3:.1f}G'
            })



        scheduler_warmup.step()
        scheduler_step.step()
        if jump and epoch % jump_seq == 0 and epoch != args.num_epochs - 1:
            continue
        # 验证
        model.eval()
        metrics_evaluator = ImageMetricsEvaluator(device=device)
        all_psnrs = []
        all_ssims = []
        all_lpips = []
        with torch.no_grad():
            for idx, eval_loader in enumerate(eval_loaders):
                epoch_psnr = AverageMeter()
                pbar_eval = tqdm(eval_loader,
                                 desc=f'验证集 {idx + 1}/{len(eval_loaders)}',
                                 total=len(eval_loader.dataset),  # 明确总数
                                 leave=False)
                start = time.time()
                for inputs, labels in pbar_eval:
                    load_time = time.time() - start
                    inputs = inputs.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)

                    torch.cuda.synchronize()  # 等待GPU完成
                    infer_start = time.time()

                    with torch.amp.autocast('cuda'):
                        preds = model(inputs).clamp(0.0, 1.0)

                    torch.cuda.synchronize()
                    infer_time = time.time() - infer_start



                    # 尺寸对齐
                    min_h = min(preds.shape[2], labels.shape[2])
                    min_w = min(preds.shape[3], labels.shape[3])
                    preds = preds[:, :, :min_h, :min_w]
                    labels = labels[:, :, :min_h, :min_w]

                    start = time.time()  # 重置计时

                    # 计算多指标
                    metrics = metrics_evaluator.evaluate(preds, labels)
                    epoch_psnr.update(metrics['psnr'], inputs.size(0))

                    pbar_eval.set_postfix({
                        'PSNR': f'{metrics["psnr"]:.2f}',
                        'SSIM': f'{metrics["ssim"]:.4f}',
                        'LPIPS': f'{metrics["lpips"]:.4f}',
                        'inf_time': f'{infer_time:.2f}s',
                        'load_time': f'{load_time:.2f}s'
                    })

                pbar_eval.close()

                avg_psnr = epoch_psnr.avg
                all_psnrs.append(avg_psnr)
                print(f'验证集 {idx + 1} - PSNR: {avg_psnr:.2f} dB, SSIM: {metrics["ssim"]:.4f}, LPIPS: {metrics["lpips"]:.4f}')

        # 计算所有验证集的平均指标
        avg_psnr_all = sum(all_psnrs) / len(all_psnrs) if all_psnrs else 0
        print(f'验证结果 - PSNR: {avg_psnr_all:.2f} dB')

        if save_ and (epoch + 1) % args.save_seq == 0:
            torch.save(best_weights, os.path.join(args.outputs_dir, f'{epoch + 1:04d}.pth'))

        if avg_psnr_all > best_psnr:
            best_epoch = epoch
            best_psnr = avg_psnr_all
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, os.path.join(args.outputs_dir, 'best.pth'))
            print(f'✓ 新的最佳模型! PSNR: {best_psnr:.2f} dB')

    print(f'\n训练完成!')
    print(f'最佳 epoch: {best_epoch + 1}, 最佳 PSNR: {best_psnr:.2f} dB')
    final_path = os.path.join(args.outputs_dir, f'final_epoch{epoch + 1}.pth')
    torch.save(model.state_dict(), final_path)
    print(f'最终模型已保存: {final_path}')