# 训练效果分析：为什么降噪能力差

## 核心问题：退化管道太弱

当前 RealESRGANDegradation 的参数：
- 高斯模糊 radius: 0.1-2.0 (太轻)
- 噪声: 0-15 (Real-ESRGAN 原版用 5-25)
- JPEG quality: 30-95 (下限太高)
- 缺少运动模糊
- 缺少颜色退化
- 各步骤概率偏低

## 验证指标致命缺陷

Phase 2 训练退化图像，但验证只用 clean 图像：
- 模型学会"平滑策略" → clean PSNR 高
- 但真实退化图像去噪能力差
- best 模型是按 clean PSNR 选的，不是去噪能力

## Phase 3 副作用

LPIPS 感知优化进一步平滑纹理，降低高频恢复能力。
