PbhK64Cm9FD!Xz2

# 超分辨率模型训练：全参数启动配置指南

一键安装包:    pip install torch h5df numpy tqdm torchvision opencv_python lpips timm
或者采用依赖.txt

本配置基于 `run_on_cloud_enhanced.py`，集成了 DDP 多卡训练、断点续训、多指标损失及智能学习率调度功能。

## 1. 基础环境与路径配置
```bash
--train-file /path/to/DIV2K /path/to/Flickr2K  # 训练集路径 (支持多路径)
--eval-file /path/to/DIV2K_eval                # 验证集路径
--outputs-dir ./checkpoints                    # 模型输出目录
--num-workers 8                                # 数据加载线程数 (根据CPU调整)
2. 核心训练策略配置 (必选)



# --- 模型与任务定义 ---
--scale 2                                      # 超分倍率 (2/3/4)
--model light                                  # 模型类型 (light/full)

# --- 分布式训练 (DDP) ---
--distributed                                  # 【多卡必选】启用分布式训练
# 单机多卡启动命令前缀: torchrun --nproc_per_node=N 

# --- 训练效率优化 ---
--batch-size 4                                 # 单卡 Batch Size (多卡需配合梯度累积)
--grad-accum 4                                 # 梯度累积步数 (模拟大 Batch, 有效BS=单卡*卡数*累积)
--amp                                          # 自动混合精度 (通常默认开启)
3. 损失函数配置 (多指标融合)
原理: Total Loss = w_l1*L1 + w_ssim*SSIM + w_lpips*LPIPS
目标: 结合像素级精度与感知级画质。
bash

编辑



--w-l1 1.0                                     # L1 损失权重 (基础像素损失)
--w-ssim 0.5                                   # SSIM 损失权重 (结构相似性)
--w-lpips 0.3                                  # LPIPS 损失权重 (感知距离, 需安装 lpips 库)
4. 智能优化器与学习率调度
推荐策略: warmup_cosine (预热+余弦退火) 或 adaptive (自适应)。
bash

编辑



# --- 调度策略选择 ---
--lr-schedule warmup_cosine                    # 调度器类型 (cosine/warmup_cosine/plateau/onecycle/adaptive)

# --- 通用超参数 ---
--lr 2e-4                                      # 基础学习率
--min-lr 1e-7                                  # 最小学习率 (防止过拟合)

# --- 策略特有参数 (根据 --lr-schedule 选择填写) ---
# 1. 若使用 warmup_cosine / cosine:
--T-max 1000                                   # 余弦退火周期 (总 Epoch 数)

# 2. 若使用 plateau (平台期):
--lr-factor 0.5                                # 平台期触发后, 学习率衰减因子
--lr-patience 10                               # 容忍多少个 Epoch 无改善后衰减

# 3. 若使用 warmup_cosine (通用推荐):
--warmup-epochs 5                              # 前 N 个 Epoch 线性预热
5. 早停机制 (Early Stopping)
作用: 防止无效训练，节省算力资源。
bash

编辑



--early-stop                                   # 【开启】早停功能
--es-patience 30                               # 连续 N 个 Epoch 无改善则停止
--es-min-epochs 100                            # 最少训练 N 个 Epoch 后才开始检测早停
--es-min-delta 1e-4                            # 改善阈值 (PSNR 提升小于该值视为无改善)
6. 容错与断点续训 (高可用)
bash



--auto-resume                                  # 【推荐】自动恢复最近一次训练状态
# 或
--resume /path/to/checkpoint_latest.pth        # 指定特定检查点恢复

# --- 检查点管理 ---
--save yes                                     # 保存模型
--save-seq 10                                  # 每 N 个 Epoch 保存一次定期检查点
7. 完整启动示例 (南昌本地环境参考)
场景: 单机 4卡 V100, 追求高质量, 自动停止



torchrun --nproc_per_node=4 run_on_cloud_enhanced.py \
  --train-file /data/DIV2K /data/Flickr2K \
  --eval-file /data/DIV2K_eval \
  --outputs-dir /data/checkpoints/swinir_x2 \
  --scale 2 \
  --batch-size 4 \
  --grad-accum 4 \
  --distributed \
  --w-l1 1.0 --w-ssim 0.5 --w-lpips 0.3 \
  --lr-schedule warmup_cosine \
  --lr 2e-4 --min-lr 1e-7 --T-max 1000 --warmup-epochs 5 \
  --early-stop --es-patience 30 --es-min-epochs 100 \
  --auto-resume \
  --num-workers 8

如果采用 json 启动方式
torchrun --nproc_per_node=2 run_on_cloud_enhanced.py     --config ./phase1_clean.json     --arch official
根据 gpu 数量选择