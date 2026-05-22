# AGENTS.md — SwinIR 超分辨率训练项目

> 本文档面向 AI 编程助手。项目的主要注释、文档和沟通语言为**中文**。

---

## 项目概述

本项目是一个基于 **SwinIR**（Swin Transformer for Image Restoration）的图像超分辨率（Image Super-Resolution, SR）训练框架。核心目标是通过三阶段训练流水线，从官方预训练权重出发，逐步优化模型在真实退化场景下的重建能力，并兼顾像素精度（PSNR）与感知质量（LPIPS/NIQE）。

- **主要任务**：将低分辨率（LR）图像放大 2x/3x/4x/8x，恢复高分辨率（HR）细节。
- **技术栈**：Python 3、PyTorch 2.11、CUDA/NCCL、PIL、OpenCV、SciPy、timm、lpips。
- **模型架构**：Swin Transformer + PixelShuffle 上采样，支持 `official`（官方预训练兼容）和 `fixed`（感知优化）两种架构变体。
- **训练规模**：支持单机单卡或单机多卡（DDP），典型配置为 2~4 卡 V100。

---

## 项目结构

```
.
├── run_on_cloud_enhanced.py    # 主训练脚本（最完整，推荐使用）
├── run_on_cloud_ddp.py         # DDP 分布式训练（旧版，功能较全）
├── run_on_cloud.py             # 单卡基础训练（最早版本）
├── run_on_cloud_optimized.py   # 优化版训练（较少使用）
├── network_swinir.py           # 官方 SwinIR 模型实现（100% 预训练兼容）
├── swinir_model.py             # 修复版模型（SwinIR_Fixed / SwinIR_Light_Fixed / SwinIR_Official）
├── cloud_dataset.py            # 数据集与 Real-ESRGAN 退化管道
├── losses.py                   # CompleteLoss（精简版多指标损失）
├── utils.py                    # 工具函数：NIQEFriendlyLoss、退化模型、EMA、评估指标等
├── metrics_evaluator.py        # 增强版图像指标评估器（YCbCr Y 通道、边界裁剪）
├── util_calculate_psnr_ssim.py # 学术标准 PSNR/SSIM 计算（含 PSNR-B）
├── infer_swinir.py             # 推理脚本（支持单图/批量、Self-Ensemble x8）
├── evaluate_sr.py              # 质量评估脚本（有参考/无参考）
├── predict.py                  # Cog/Replicate 封装（参考代码，当前未使用）
├── diagnose_psnr.py            # PSNR 诊断工具
├── diagnose_training.py        # 训练诊断工具
├── config_json/                # 三阶段训练 JSON 配置文件
│   ├── phase1_clean.json
│   ├── phase2_degrade.json
│   └── phase3_perceptual.json
│   └── ... (light 变体、x3/x4 变体)
├── pre/                        # 预训练权重存放目录（当前为空，需自行下载）
├── requirements.txt            # Python 依赖
└── readme.md                   # 中文使用说明
```

### 关键模块说明

| 文件 | 职责 |
|------|------|
| `run_on_cloud_enhanced.py` | **唯一推荐的主训练入口**。集成 DDP、断点续训、EMA、TwoStage 训练、SmartLRScheduler、EarlyStopping、JSON 配置、多指标评分。 |
| `network_swinir.py` | 官方 SwinIR 的原始实现（含 `SwinIR` 类）。`infer_swinir.py` 使用此文件加载权重。 |
| `swinir_model.py` | 项目自定义模型。`SwinIR_Official`（兼容预训练）、`SwinIR_Fixed`（添加 residual_gate + 全局 bicubic 残差）、`SwinIR_Light_Fixed`（轻量版）。 |
| `cloud_dataset.py` | `FixedFolderDataset`（训练集，真随机裁剪 + 退化）、`FixedValidationDataset`（验证集，从 LR/HR 文件夹加载）、`DegradedValidationDataset`（退化验证集）、`RealESRGANDegradation`（二阶退化）。 |
| `losses.py` | 精简版 `CompleteLoss`，含 Charbonnier、MSCNStatLoss、FFTLoss、EdgeLoss、SSIMLoss、LPIPS。支持 `pixel`/`perceptual` 阶段自动切换权重。 |
| `utils.py` | 早期工具集合，包含另一个 `CompleteLoss`、`NIQEFriendlyLoss`、`DegradationModel`、`EMA`、`ImageMetricsEvaluator` 等。注意与 `losses.py` 存在功能重叠。 |

---

## 构建与运行

### 环境安装

```bash
pip install -r requirements.txt
```

核心依赖版本锁定：
- `torch==2.11.0`
- `torchvision==0.26.0`
- `lpips==0.1.4`
- `timm==1.0.26`
- `opencv-python==4.13.0.92`
- `scipy==1.17.1`

> 注意：`torch` 需根据本地 CUDA 版本从 PyTorch 官网指定 wheel 安装。CPU 训练不支持。

### 训练命令

#### 三阶段训练流水线（推荐）

```bash
# Phase 1: 纯像素精度（clean SR）
torchrun --nproc_per_node=2 run_on_cloud_enhanced.py --config config_json/phase1_clean.json

# Phase 2: 真实退化学习（Real-ESRGAN 风格）
torchrun --nproc_per_node=2 run_on_cloud_enhanced.py --config config_json/phase2_degrade.json

# Phase 3: 感知质量优化
torchrun --nproc_per_node=2 run_on_cloud_enhanced.py --config config_json/phase3_perceptual.json
```

#### 手动命令行启动（示例）

```bash
torchrun --nproc_per_node=4 run_on_cloud_enhanced.py \
  --train-file /data/DIV2K /data/Flickr2K \
  --eval-file /data/DIV2K_eval /data/Set14 /data/Urban100 /data/bsd300/test \
  --outputs-dir ./checkpoints/swinir_x2 \
  --scale 2 --model full --arch official \
  --batch-size 4 --grad-accum 4 --distributed \
  --w-l1 1.0 --w-ssim 0.5 --w-lpips 0.3 \
  --lr-schedule warmup_cosine --lr 2e-4 --min-lr 1e-7 \
  --T-max 1000 --warmup-epochs 5 \
  --early-stop --es-patience 30 --es-min-epochs 100 \
  --auto-resume --save yes --save-seq 10 \
  --num-workers 8
```

### 推理命令

```bash
# 单图推理
python infer_swinir.py \
  --input ./input.png \
  --weights ./checkpoints/phase3_perceptual/full_swinir_fixed_x2/0001/checkpoint_best.pth \
  --scale 2 --output ./output/sr.png

# 批量推理 + Self-Ensemble（可提升 0.1~0.3 dB）
python infer_swinir.py \
  --input ./input_dir/ --output ./output_dir/ \
  --weights ./pre/001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth \
  --scale 2 --self-ensemble
```

### 评估命令

```bash
# 有参考图评估
python evaluate_sr.py --ref reference.png --sr output.png

# 无参考图评估
python evaluate_sr.py --sr output.png --no-ref
```

---

## 代码风格与约定

### 语言
- 所有注释、文档字符串、日志输出、错误提示均使用**中文**。
- 代码中的类名、函数名、变量名使用英文，遵循 PEP 8。

### 代码组织习惯
- 使用 `═` 符号做区块分隔装饰线（如 `# ═══════════════════════════════════════`）。
- 大量全局变量用于跨函数共享状态（`device`, `model`, `ema`, `args` 等），这在训练脚本中很常见。
- 模型定义与训练逻辑分离：模型在 `swinir_model.py` / `network_swinir.py`，训练循环在 `run_on_cloud_enhanced.py`。
- 存在历史遗留文件：`utils.py` 中的 `CompleteLoss` 与 `losses.py` 中的 `CompleteLoss` 是两个独立实现。主脚本 `run_on_cloud_enhanced.py` 导入的是 `losses.py` 版本。

### 数值与设备约定
- 图像张量范围：`[0, 1]`（`ToTensor()` 默认）。
- LPIPS 输入需转换到 `[-1, 1]`（`pred * 2 - 1`）。
- 默认设备：`cuda`，单卡用 `cuda:0`，DDP 用 `cuda:{local_rank}`。
- 随机种子：`args.seed + rank`，保证分布式下各进程数据不重复。

---

## 训练架构详解

### 模型选择

通过 `--arch` 和 `--model` 控制：

| `--arch` | 说明 | 典型用途 |
|---------|------|---------|
| `official` | 使用 `network_swinir.py` 的 `SwinIR`，100% 兼容官方预训练权重 | 追 PSNR、从官方权重微调 |
| `fixed` | 使用 `swinir_model.py` 的 `SwinIR_Fixed` 或 `SwinIR_Light_Fixed` | 感知优化、自定义残差 |

| `--model` | 说明 |
|----------|------|
| `full` | 完整模型（默认 `embed_dim=180`, `depths=[6,6,6,6,6,6]`） |
| `light` | 轻量模型（`embed_dim=60`, `depths=[4,4]`） |

### 三阶段训练策略

1. **Phase 1 (Pixel)**：干净数据（`degradation=clean`），损失以 Charbonnier/L1 为主，目标最高 PSNR。
2. **Phase 2 (Degrade)**：二阶退化数据（`degradation=second_order`），学习去噪/去模糊/抗 JPEG 能力。验证集也用退化分布。
3. **Phase 3 (Perceptual)**：继续退化训练，引入 LPIPS、FFT、MSCN 等感知损失，牺牲少量 PSNR 换取视觉自然度。

阶段间的预训练权重需手动修改 JSON 中的 `pretrained` 路径。输出目录自动创建子目录 `0001`, `0002`... 以区分不同运行。

### 关键训练特性

- **DDP**：通过 `torchrun --nproc_per_node=N` 启动，`nccl` backend。
- **断点续训**：`--auto-resume` 自动找 `checkpoint_latest.pth`，或 `--resume` 指定路径。
- **EMA**：`--use-ema --ema-decay 0.999`，验证时可选应用 EMA 权重。
- **TwoStageTrainer**：自动在第 `stage1_epochs` 后从 `pixel` 阶段过渡到 `perceptual` 阶段，线性插值损失权重。
- **SmartLRScheduler**：支持 `warmup_cosine`（推荐）、`plateau`、`adaptive`、`onecycle`、`cosine`。
- **EarlyStopping**：基于多指标综合评分（不再是纯 PSNR）。
- **FP32 训练**：出于数值稳定性考虑，**未使用 AMP/GradScaler**。SwinIR 的 attention softmax 在 FP16 下容易溢出导致 NaN。

---

## 数据格式与数据集

### 训练集

- 支持多路径（`--train-file` 传多个文件夹）。
- 每个文件夹下优先读取 `HR/` 子目录中的图片（防止混入预生成的 LR 变体）。
- 支持的格式：`.png`, `.jpg`, `.jpeg`, `.bmp`。
- `FixedFolderDataset` 在 `__getitem__` 中做真随机裁剪（修复了旧版的预计算裁剪 bug）。

### 验证集

- 支持多验证集（`--eval-file` 传多个文件夹）。
- `FixedValidationDataset`：从 `LR/X{scale}/` 和 `HR/` 分别加载（clean 验证）。
- `DegradedValidationDataset`：从 `HR/` 加载并实时应用退化（用于 Phase 2/3，保证验证分布与训练一致）。

### 退化管道（`RealESRGANDegradation`）

`cloud_dataset.py` 中的 `second_order` 模式：
- 第一阶：高斯模糊、Sinc 滤波（振铃）、运动模糊、颜色退化、随机缩放、高斯/泊松噪声。
- 第二阶：再次模糊、运动模糊、JPEG 压缩（真实 PIL DCT 域量化）。
- 最终：双三次下采样 + 最终噪声。

> 关键修复：退化中的随机缩放可能改变图像尺寸，因此每次退化后必须将 LR resize 回标准尺寸 `patch_size // scale`，否则 DataLoader 无法 stack。

---

## 检查点与权重格式

### 保存内容

检查点 `.pth` 是一个字典，通常包含：
- `'model'`：模型 `state_dict`
- `'optimizer'`、`'lr_scheduler'`
- `'epoch'`、`'best_psnr'`、`'best_epoch'`
- `'ema'`（如果启用）
- `'stage_trainer'`（两阶段状态）
- `'early_stopping'`、`'monitor'`（训练日志）
- `'args'`（完整参数快照）

### 文件命名

- `checkpoint_latest.pth`：最新检查点（每次 epoch 保存）
- `checkpoint_best.pth`：最佳模型（基于多指标评分）
- `checkpoint_epoch{XXXX}.pth`：定期保存（由 `--save-seq` 控制，默认保留最近 3 个）
- `final.pth`：训练结束保存的最终模型

### 加载预训练权重

- 官方权重格式支持：`{'params': state_dict}`、`{'model': state_dict}`、`{'state_dict': ...}`、裸 `state_dict`。
- `load_pretrained()`（`swinir_model.py`）会自动做键名映射（如 `residual_group.blocks` → `blocks`、`patch_embed` → `conv_first` 等）。
- 跨 scale 迁移时可用 `--pretrained-no-upsample` 跳过上采样层加载。

---

## 测试策略

本项目**没有使用 pytest/unittest 等自动化测试框架**。质量保障依赖于：

1. **训练时验证**：每个 epoch（或按 `--val-interval`）在多个验证集上计算 PSNR/SSIM/LPIPS/NIQE。
2. **推理验证**：`infer_swinir.py` 支持 Self-Ensemble x8，可作为最终测试增强手段。
3. **诊断脚本**：
   - `diagnose_psnr.py`：检查 PSNR 计算逻辑。
   - `diagnose_training.py`：检查训练流程中的数值异常。
4. **手动评估**：`evaluate_sr.py` 用于对比不同方法输出。

**开发建议**：若修改损失函数或模型结构，务必先在单卡小 epoch 上跑通（观察是否 NaN），再启动完整 DDP 训练。

---

## 安全与稳定性注意事项

- **NaN 防护**：训练脚本在 `forward` 后、`backward` 前、梯度裁剪后均做了 `torch.isfinite()` 检查。一旦检测到 NaN/Inf，会跳过当前 batch 并清空梯度。
- **DataLoader 崩溃**：`torch.flip` / `torch.rot90` 返回 view tensor，其 storage 不可 resize。数据集 `__getitem__` 中必须在空间增强后调用 `.contiguous()`，且 `safe_collate_fn` 中会再 `.clone()` 一次。
- **MSCN NaN**：`MSCNStatLoss` 在像素阶段（`stage=pixel`）默认权重为 0，因为训练初期模型输出不稳定，`(x-mu)/sigma` 易除零。待 `transition` / `perceptual` 阶段再启用。
- **旧 checkpoint 兼容性**：`run_on_cloud_enhanced.py` 恢复时会检测 `residual_gate` 等新增参数，对旧权重做兼容处理。
- **LR 异常检测**：训练监控器会检测学习率异常上升（scheduler 状态污染），自动重置。

---

## 常见开发任务指引

### 添加新的损失组件

1. 在 `losses.py` 中继承 `nn.Module` 实现新损失。
2. 在 `CompleteLoss.__init__` 中实例化，在 `forward` 中计算并加入 `total`。
3. 添加对应的 `w_xxx` 权重参数，支持命令行 `--w-xxx` 和 JSON 配置覆盖。
4. 在 `TwoStageTrainer.step()` 中更新过渡阶段的插值逻辑（如果需要）。

### 修改模型结构

- 若需**保持预训练兼容**：修改 `swinir_model.py` 中的 `SwinIR_Official` 或 `network_swinir.py`，注意参数名匹配。
- 若做**实验性修改**：在 `SwinIR_Fixed` / `RSTB_Fixed` 中修改，不影响官方权重加载。
- 修改后需在 `infer_swinir.py` 的 `detect_config_from_state_dict()` 中更新自动检测逻辑（如果需要）。

### 调整退化强度

编辑 `cloud_dataset.py` 中的 `RealESRGANDegradation`：
- 修改各退化步骤的概率（如 `random.random() < 0.9`）。
- 修改模糊半径、噪声水平、JPEG quality 范围。
- 新增退化步骤（如散焦模糊、压缩伪影）时，注意输出尺寸必须能被 `scale` 整除，或在 `degrade()` 末尾 resize 到标准尺寸。

### 新增 JSON 配置

复制现有 `config_json/phase*.json`，修改对应参数。JSON 中的键名使用连字符（如 `train-file`），程序内会自动替换为下划线（`train_file`）。命令行参数优先级高于 JSON。

---

## 注意事项

- `predict.py` 引用了不存在的 `main_test_swinir.py`，这是从官方 SwinIR 仓库遗留的 Cog 封装代码，**当前未使用**。
- `utils.py` 中的部分类（如 `CompleteLoss`、`ImageMetricsEvaluator`）与 `losses.py`、`metrics_evaluator.py` 功能重复。主脚本 `run_on_cloud_enhanced.py` 明确导入的是 `losses.py` 和 `cloud_dataset.py` 版本。修改时请确认实际被使用的定义位置。
- `degrade_analysis.md` 记录了历史问题分析（退化管道太弱、验证指标缺陷），供参考。
