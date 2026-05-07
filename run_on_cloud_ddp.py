import argparse
import os
import copy
import numpy as np
import random

import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import ConcatDataset

# 分布式训练相关
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from tqdm.auto import tqdm

import cloud_dataset
from swinir_model import SwinIR
from utils import AverageMeter, MultiMetricLoss, ImageMetricsEvaluator


# ═══════════════════════════════════════════════════════════════════════════════
# 分布式训练工具函数
# ═══════════════════════════════════════════════════════════════════════════════

def setup_distributed():
    """
    初始化分布式训练环境
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        world_size = 1
        gpu = 0
        print('不使用分布式训练')

    torch.cuda.set_device(gpu)
    dist_backend = 'nccl'
    dist_url = 'env://'

    dist.init_process_group(backend=dist_backend, init_method=dist_url,
                           world_size=world_size, rank=rank)
    dist.barrier()

    return rank, world_size, gpu


def cleanup_distributed():
    """
    清理分布式训练环境
    """
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """检查是否为主进程"""
    return not dist.is_initialized() or dist.get_rank() == 0


def get_rank():
    """获取当前进程 rank"""
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size():
    """获取总进程数"""
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def reduce_tensor(tensor, world_size=None):
    """跨进程聚合 tensor"""
    if not dist.is_initialized():
        return tensor
    if world_size is None:
        world_size = get_world_size()
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt


# ═══════════════════════════════════════════════════════════════════════════════
# 检查点管理
# ═══════════════════════════════════════════════════════════════════════════════

class CheckpointManager:
    """
    检查点管理器：支持保存和加载训练状态
    """

    def __init__(self, checkpoint_dir, keep_last_n=3):
        self.checkpoint_dir = checkpoint_dir
        self.keep_last_n = keep_last_n
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save_checkpoint(self, state, is_best=False, filename='checkpoint.pth'):
        """
        保存检查点

        Args:
            state: 包含 model, optimizer, scheduler, epoch, best_psnr 等的字典
            is_best: 是否为最佳模型
            filename: 保存文件名
        """
        if not is_main_process():
            return

        filepath = os.path.join(self.checkpoint_dir, filename)
        torch.save(state, filepath)

        # 保存为最新检查点
        latest_path = os.path.join(self.checkpoint_dir, 'checkpoint_latest.pth')
        torch.save(state, latest_path)

        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'checkpoint_best.pth')
            torch.save(state, best_path)
            print(f'✓ 保存最佳检查点: epoch={state["epoch"]}, PSNR={state.get("best_psnr", 0):.2f}')

        # 清理旧检查点
        self._cleanup_old_checkpoints()

    def load_checkpoint(self, filename='checkpoint_latest.pth'):
        """
        加载检查点

        Returns:
            state dict 或 None
        """
        filepath = os.path.join(self.checkpoint_dir, filename)
        if os.path.exists(filepath):
            print(f'加载检查点: {filepath}')
            checkpoint = torch.load(filepath, map_location='cpu')
            return checkpoint
        return None

    def find_latest_checkpoint(self):
        """
        自动查找最新的检查点

        Returns:
            检查点路径或 None
        """
        candidates = [
            os.path.join(self.checkpoint_dir, 'checkpoint_latest.pth'),
            os.path.join(self.checkpoint_dir, 'checkpoint_best.pth'),
        ]

        # 查找所有 epoch 检查点
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
        """清理旧的 epoch 检查点，只保留最新的 N 个"""
        epoch_files = [f for f in os.listdir(self.checkpoint_dir) 
                      if f.startswith('checkpoint_epoch') and f.endswith('.pth')]

        if len(epoch_files) > self.keep_last_n:
            epoch_files.sort()
            for old_file in epoch_files[:-self.keep_last_n]:
                os.remove(os.path.join(self.checkpoint_dir, old_file))


# ═══════════════════════════════════════════════════════════════════════════════
# 主程序
# ═══════════════════════════════════════════════════════════════════════════════

device = torch.device('cpu')
save_ = False
model = None
jump = False
jump_seq = 3

# 全局变量用于分布式训练
rank = 0
world_size = 1
gpu = 0
is_distributed = False


def pre_run():
    """
    预处理：初始化设备、模型、分布式环境
    """
    global device, save_, args, model, jump, jump_seq
    global rank, world_size, gpu, is_distributed

    # 初始化分布式环境
    if args.distributed:
        rank, world_size, gpu = setup_distributed()
        is_distributed = True
        device = torch.device(f'cuda:{gpu}')
        print(f'[Rank {rank}/{world_size}] 使用 GPU {gpu}')
    else:
        if not torch.cuda.is_available():
            print("无法使用 cuda，请设置好环境")
            return
        device = torch.device('cuda:0')
        print("使用单卡训练")

    # 设置随机种子（每个进程使用不同的种子）
    seed = args.seed + rank
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # 性能优化设置
    cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # 创建输出目录（仅主进程）
    if is_main_process():
        output_dir = os.path.join(args.outputs_dir, f'{args.model}_swinir_x{args.scale}')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        counts = os.listdir(output_dir)
        args.outputs_dir = os.path.join(output_dir, f'{len(counts) + 1:04d}')
        os.makedirs(args.outputs_dir, exist_ok=True)
        print(f'输出目录: {args.outputs_dir}')

    # 同步输出目录路径
    if is_distributed:
        if is_main_process():
            output_path = args.outputs_dir
        else:
            output_path = ''
        # 广播输出目录路径
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

    # 创建模型
    model = SwinIR(scale=args.scale, embed_dim=180, depths=[6, 6, 6, 6]).to(device)

    if is_main_process():
        print(f'模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M')

    # 包装为 DDP 模型
    if is_distributed:
        model = DDP(model, device_ids=[gpu], find_unused_parameters=False)


def data_loader_list_return():
    """
    创建训练和验证数据加载器
    支持分布式采样
    """
    train_dataset = copy.deepcopy(args.train_file)
    eval_dataset = copy.deepcopy(args.eval_file)

    if is_main_process():
        print('加载 train_set')

    for index in range(len(train_dataset)):
        train_dataset[index] = cloud_dataset.FolderDataset(
            train_dataset[index],
            scale=args.scale,
            patch_size=args.patch_size,
            pre_crop=True
        )

    train_file_set = ConcatDataset(train_dataset)

    # 分布式采样器
    if is_distributed:
        train_sampler = DistributedSampler(
            train_file_set,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        shuffle = False  # 使用 sampler 时不需要 shuffle
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
    )

    if is_main_process():
        print('加载 eval_set')

    eval_loaders = []
    for index in range(len(eval_dataset)):
        eval_ds = cloud_dataset.FolderDataset(
            eval_dataset[index],
            scale=args.scale,
            patch_size=args.patch_size,
            pre_crop=False
        )
        eval_loader = DataLoader(
            dataset=eval_ds,
            batch_size=1,
            num_workers=2,
            pin_memory=True,
            persistent_workers=False
        )
        eval_loaders.append(eval_loader)

    if is_main_process():
        print(f"使用 FolderDataset 模式")

    return train_loader, eval_loaders, train_sampler


def train_one_epoch(model, train_loader, criterion, optimizer, scaler, epoch, train_sampler=None):
    """
    训练一个 epoch

    Args:
        model: 模型
        train_loader: 训练数据加载器
        criterion: 损失函数
        optimizer: 优化器
        scaler: 梯度缩放器（AMP）
        epoch: 当前 epoch
        train_sampler: 分布式采样器

    Returns:
        平均损失
    """
    model.train()

    # 分布式采样器设置 epoch
    if train_sampler is not None:
        train_sampler.set_epoch(epoch)

    epoch_losses = AverageMeter()
    epoch_l1 = AverageMeter()
    epoch_ssim = AverageMeter()
    epoch_lpips = AverageMeter()

    # 仅主进程显示进度条
    if is_main_process():
        pbar = tqdm(train_loader, desc=f'Epoch [{epoch + 1}/{args.num_epochs}]')
    else:
        pbar = train_loader

    optimizer.zero_grad(set_to_none=True)

    for batch_idx, (inputs, labels) in enumerate(pbar):
        # 非阻塞传输到 GPU
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

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
            optimizer.zero_grad(set_to_none=True)

        # 更新统计
        epoch_losses.update(loss_dict['total'], inputs.size(0))
        epoch_l1.update(loss_dict['l1'], inputs.size(0))
        epoch_ssim.update(loss_dict['ssim'], inputs.size(0))
        epoch_lpips.update(loss_dict['lpips'], inputs.size(0))

        # 仅主进程更新进度条
        if is_main_process():
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'total': f'{loss_dict["total"]:.4f}',
                'l1': f'{loss_dict["l1"]:.4f}',
                'ssim': f'{loss_dict["ssim"]:.4f}',
                'lpips': f'{loss_dict["lpips"]:.4f}',
                'lr': f'{current_lr:.6f}',
                'gpu': f'{torch.cuda.memory_allocated() / 1024 ** 3:.1f}G'
            })

    # 同步各进程的损失
    if is_distributed:
        avg_loss = torch.tensor([epoch_losses.avg], device=device)
        avg_loss = reduce_tensor(avg_loss)
        return avg_loss.item()

    return epoch_losses.avg


def validate(model, eval_loaders, device):
    """
    验证模型

    Args:
        model: 模型
        eval_loaders: 验证数据加载器列表
        device: 设备

    Returns:
        平均 PSNR
    """
    model.eval()
    metrics_evaluator = ImageMetricsEvaluator(device=device)

    all_psnrs = []
    all_ssims = []
    all_lpips = []

    with torch.no_grad():
        for idx, eval_loader in enumerate(eval_loaders):
            epoch_psnr = AverageMeter()
            epoch_ssim = AverageMeter()
            epoch_lpips_loss = AverageMeter()

            if is_main_process():
                pbar_eval = tqdm(
                    eval_loader,
                    desc=f'验证集 {idx + 1}/{len(eval_loaders)}',
                    total=len(eval_loader.dataset),
                    leave=False
                )
            else:
                pbar_eval = eval_loader

            for inputs, labels in pbar_eval:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with torch.amp.autocast('cuda'):
                    preds = model(inputs).clamp(0.0, 1.0)

                # 尺寸对齐
                min_h = min(preds.shape[2], labels.shape[2])
                min_w = min(preds.shape[3], labels.shape[3])
                preds = preds[:, :, :min_h, :min_w]
                labels = labels[:, :, :min_h, :min_w]

                # 计算多指标
                metrics = metrics_evaluator.evaluate(preds, labels)
                epoch_psnr.update(metrics['psnr'], inputs.size(0))
                epoch_ssim.update(metrics['ssim'], inputs.size(0))
                epoch_lpips_loss.update(metrics['lpips'], inputs.size(0))

                if is_main_process():
                    pbar_eval.set_postfix({
                        'PSNR': f'{metrics["psnr"]:.2f}',
                        'SSIM': f'{metrics["ssim"]:.4f}',
                        'LPIPS': f'{metrics["lpips"]:.4f}'
                    })

            if is_main_process():
                pbar_eval.close()

            all_psnrs.append(epoch_psnr.avg)
            all_ssims.append(epoch_ssim.avg)
            all_lpips.append(epoch_lpips_loss.avg)

    # 计算所有验证集的平均指标
    avg_psnr = sum(all_psnrs) / len(all_psnrs) if all_psnrs else 0
    avg_ssim = sum(all_ssims) / len(all_ssims) if all_ssims else 0
    avg_lpips = sum(all_lpips) / len(all_lpips) if all_lpips else 0

    # 同步各进程的指标
    if is_distributed:
        metrics_tensor = torch.tensor([avg_psnr, avg_ssim, avg_lpips], device=device)
        metrics_tensor = reduce_tensor(metrics_tensor)
        avg_psnr, avg_ssim, avg_lpips = metrics_tensor.tolist()

    return avg_psnr, avg_ssim, avg_lpips


def main():
    """
    主训练函数
    """
    global args, model, device, rank, world_size, is_distributed

    # 命令行参数管理
    parser = argparse.ArgumentParser()

    # 必须参数
    parser.add_argument('--train-file', type=str, required=True, nargs='+', 
                       help='训练文件夹路径')
    parser.add_argument('--eval-file', type=str, required=True, nargs='+', 
                       help='验证数据文件夹路径')
    parser.add_argument('--outputs-dir', type=str, required=True, 
                       help='输出文件夹路径')
    parser.add_argument('--valid-dir', type=str, nargs='+', 
                       help='测试文件夹路径')

    # 模型参数
    parser.add_argument('--scale', type=int, default=2, 
                       help='缩放倍率')
    parser.add_argument('--model', type=str, default='full', choices=['light', 'full'], 
                       help='模型轻量化')

    # 训练参数
    parser.add_argument('--lr', type=float, default=2e-4, 
                       help='学习率')
    parser.add_argument('--batch-size', type=int, default=4, 
                       help='每个 GPU 的批次大小')
    parser.add_argument('--num-epochs', type=int, default=1000, 
                       help='epoch 数量')
    parser.add_argument('--num-workers', type=int, default=4, 
                       help='数据加载线程数')
    parser.add_argument('--patch-size', type=int, default=64, 
                       help='训练裁剪块大小')
    parser.add_argument('--seed', type=int, default=123, 
                       help='随机种子')

    # 损失权重参数
    parser.add_argument('--w-l1', type=float, default=1.0, 
                       help='L1损失权重')
    parser.add_argument('--w-ssim', type=float, default=0.5, 
                       help='SSIM损失权重')
    parser.add_argument('--w-lpips', type=float, default=0.3, 
                       help='LPIPS损失权重')

    # 梯度累积
    parser.add_argument('--grad-accum', type=int, default=1, 
                       help='梯度累积步数')

    # 保存参数
    parser.add_argument('--save', type=str, default='no', choices=['yes', 'no'], 
                       help='是否保存每个epoch')
    parser.add_argument('--save-seq', type=int, default=5, 
                       help='保存频率')

    # 验证参数
    parser.add_argument('--jump', type=str, default='no', choices=['yes', 'no'], 
                       help='是否跳过一定的验证集')
    parser.add_argument('--jump-seq', type=int, default=3, 
                       help='跳过频率')

    # 分布式训练参数
    parser.add_argument('--distributed', action='store_true', 
                       help='启用分布式训练')
    parser.add_argument('--local_rank', type=int, default=0, 
                       help='本地 rank（由 torchrun 自动设置）')

    # 断点续训参数
    parser.add_argument('--resume', type=str, default='', 
                       help='恢复训练的检查点路径')
    parser.add_argument('--auto-resume', action='store_true', 
                       help='自动查找最新检查点恢复训练')
    parser.add_argument('--start-epoch', type=int, default=0, 
                       help='开始 epoch（用于断点续训）')

    # 学习率调度参数
    parser.add_argument('--warmup-epochs', type=int, default=5, 
                       help='预热 epoch 数')
    parser.add_argument('--milestones', type=int, nargs='+', default=[200, 400, 600, 800], 
                       help='学习率衰减里程碑')
    parser.add_argument('--gamma', type=float, default=0.5, 
                       help='学习率衰减系数')

    args = parser.parse_args()

    # 预处理
    pre_run()

    # 创建检查点管理器
    checkpoint_manager = CheckpointManager(args.outputs_dir, keep_last_n=3)

    # 多指标组合损失函数
    criterion = MultiMetricLoss(
        w_l1=args.w_l1, 
        w_ssim=args.w_ssim, 
        w_lpips=args.w_lpips, 
        device=device
    )

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 学习率调度
    warmup_epochs = args.warmup_epochs
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 1.0

    scheduler_warmup = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scheduler_step = optim.lr_scheduler.MultiStepLR(
        optimizer, 
        milestones=args.milestones, 
        gamma=args.gamma
    )

    # 使用自动混合精度
    scaler = torch.amp.GradScaler('cuda', init_scale=2**16, growth_factor=2.0, 
                                  backoff_factor=0.5, growth_interval=2000)

    # 加载数据
    train_loader, eval_loaders, train_sampler = data_loader_list_return()

    # 初始化训练状态
    start_epoch = args.start_epoch
    best_psnr = 0.0
    best_epoch = 0

    # ═══════════════════════════════════════════════════════════════════════════
    # 断点续训：加载检查点
    # ═══════════════════════════════════════════════════════════════════════════
    if args.resume or args.auto_resume:
        checkpoint_path = args.resume

        # 自动查找最新检查点
        if args.auto_resume and not checkpoint_path:
            checkpoint_path = checkpoint_manager.find_latest_checkpoint()

        if checkpoint_path and os.path.exists(checkpoint_path):
            if is_main_process():
                print(f'\n{"="*60}')
                print(f'恢复训练: {checkpoint_path}')
                print(f'{"="*60}')

            checkpoint = torch.load(checkpoint_path, map_location='cpu')

            # 加载模型权重
            if is_distributed:
                model.module.load_state_dict(checkpoint['model'])
            else:
                model.load_state_dict(checkpoint['model'])

            # 加载优化器状态
            optimizer.load_state_dict(checkpoint['optimizer'])

            # 加载学习率调度器状态
            if 'scheduler_warmup' in checkpoint:
                scheduler_warmup.load_state_dict(checkpoint['scheduler_warmup'])
            if 'scheduler_step' in checkpoint:
                scheduler_step.load_state_dict(checkpoint['scheduler_step'])

            # 加载 AMP scaler 状态
            if 'scaler' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler'])

            # 加载训练状态
            start_epoch = checkpoint.get('epoch', 0) + 1
            best_psnr = checkpoint.get('best_psnr', 0.0)
            best_epoch = checkpoint.get('best_epoch', 0)

            if is_main_process():
                print(f'从 epoch {start_epoch} 继续训练')
                print(f'当前最佳 PSNR: {best_psnr:.2f} (epoch {best_epoch + 1})')
        else:
            if is_main_process():
                print('未找到检查点，从头开始训练')

    # 同步各进程的起始 epoch
    if is_distributed:
        start_epoch_tensor = torch.tensor([start_epoch], device=device)
        dist.broadcast(start_epoch_tensor, src=0)
        start_epoch = int(start_epoch_tensor.item())

    # ═══════════════════════════════════════════════════════════════════════════
    # 训练循环
    # ═══════════════════════════════════════════════════════════════════════════
    if is_main_process():
        print(f'\n{"="*60}')
        print(f'开始训练')
        print(f'总 epoch: {args.num_epochs}')
        print(f'开始 epoch: {start_epoch}')
        print(f'批次大小: {args.batch_size} x {world_size} GPUs = {args.batch_size * world_size}')
        print(f'梯度累积: {args.grad_accum}')
        print(f'有效批次大小: {args.batch_size * world_size * args.grad_accum}')
        print(f'损失权重: L1={args.w_l1}, SSIM={args.w_ssim}, LPIPS={args.w_lpips}')
        print(f'{"="*60}\n')

    for epoch in range(start_epoch, args.num_epochs):
        # 训练一个 epoch
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, 
            epoch, train_sampler
        )

        # 更新学习率
        scheduler_warmup.step()
        scheduler_step.step()

        # 跳过验证（如果启用）
        if jump and epoch % jump_seq == 0 and epoch != args.num_epochs - 1:
            continue

        # 验证
        avg_psnr, avg_ssim, avg_lpips = validate(model, eval_loaders, device)

        # 仅主进程输出和保存
        if is_main_process():
            print(f'Epoch [{epoch + 1}/{args.num_epochs}] '
                  f'Loss: {train_loss:.4f} | '
                  f'PSNR: {avg_psnr:.2f} dB | '
                  f'SSIM: {avg_ssim:.4f} | '
                  f'LPIPS: {avg_lpips:.4f}')

            # 保存定期检查点
            if save_ and (epoch + 1) % args.save_seq == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model': model.module.state_dict() if is_distributed else model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler_warmup': scheduler_warmup.state_dict(),
                    'scheduler_step': scheduler_step.state_dict(),
                    'scaler': scaler.state_dict(),
                    'best_psnr': best_psnr,
                    'best_epoch': best_epoch,
                    'args': vars(args)
                }
                checkpoint_manager.save_checkpoint(
                    checkpoint, 
                    filename=f'checkpoint_epoch{epoch + 1:04d}.pth'
                )

            # 保存最佳模型
            is_best = avg_psnr > best_psnr
            if is_best:
                best_epoch = epoch
                best_psnr = avg_psnr
                if is_main_process():
                    print(f'✓ 新的最佳模型! PSNR: {best_psnr:.2f} dB')

            # 保存最新检查点（包含最佳模型信息）
            checkpoint = {
                'epoch': epoch,
                'model': model.module.state_dict() if is_distributed else model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler_warmup': scheduler_warmup.state_dict(),
                'scheduler_step': scheduler_step.state_dict(),
                'scaler': scaler.state_dict(),
                'best_psnr': best_psnr,
                'best_epoch': best_epoch,
                'args': vars(args)
            }
            checkpoint_manager.save_checkpoint(checkpoint, is_best=is_best)

    # 训练完成
    if is_main_process():
        print(f'\n{"="*60}')
        print(f'训练完成!')
        print(f'最佳 epoch: {best_epoch + 1}')
        print(f'最佳 PSNR: {best_psnr:.2f} dB')
        print(f'{"="*60}')

        # 保存最终模型
        final_path = os.path.join(args.outputs_dir, f'final_epoch{args.num_epochs}.pth')
        torch.save(
            model.module.state_dict() if is_distributed else model.state_dict(),
            final_path
        )
        print(f'最终模型已保存: {final_path}')

    # 清理分布式环境
    if is_distributed:
        cleanup_distributed()


if __name__ == '__main__':
    main()