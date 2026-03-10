# LongVie 2 训练过程伪代码

本文档基于 LongVie 2 官方仓库 (`e:\trae\LongVie`) 的实际实现，结合论文理论框架，详细描述训练过程的完整流程。

---

## 📋 目录

1. [训练流程总览](#1-训练流程总览)
2. [阶段一：控制信号训练](#2-阶段一控制信号训练)
3. [阶段二：历史一致性增强](#3-阶段二历史一致性增强)
4. [核心算法详解](#4-核心算法详解)
5. [完整训练伪代码](#5-完整训练伪代码)

---

## 1. 训练流程总览

### 1.1 两阶段训练策略

```
┌─────────────────────────────────────────────────────────────┐
│                    LongVie 2 训练流程                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  第一阶段：控制信号训练                                      │
│  ├── 目标：训练 Dual Controller                             │
│  ├── 输入：视频 + 深度图 + 轨迹图                           │
│  ├── 输出：control.safetensors                              │
│  └── 时长：~800 steps                                       │
│                                                             │
│  第二阶段：历史一致性增强                                    │
│  ├── 目标：时序一致性训练                                   │
│  ├── 输入：视频 + 深度图 + 轨迹图 + 历史视频                │
│  ├── 输出：final_model.safetensors                          │
│  └── 时长：~800 steps                                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 数据准备流程

```python
# ============================================================
# 数据预处理伪代码
# ============================================================

FUNCTION 准备训练数据 ():
    
    # 步骤 1: 收集原始视频
    # ====================
    原始视频数据集 = [
        {
            "video_path": "video_001.mp4",
            "text_prompt": "一只猫在草地上奔跑"
        },
        {
            "video_path": "video_002.mp4", 
            "text_prompt": "汽车在公路上行驶"
        },
        ...
    ]
    
    # 步骤 2: 提取深度图 (使用 Video Depth Anything)
    # ============================================
    FOR 每个视频 in 原始视频数据集:
        
        # 加载视频帧
        视频帧列表 = 加载视频 (视频 ["video_path"])
        # 形状：[T, H, W, C], T=帧数，例如 81 帧
        
        # 运行深度估计模型
        深度图列表 = VideoDepthAnything.推理 (视频帧列表)
        # 形状：[T, H, W], 单通道深度图
        
        # 归一化深度值到 [0, 1]
        深度图列表 = (深度图列表 - 最小值) / (最大值 - 最小值)
        
        # 保存深度图
        保存为 NPZ(深度图列表，f"depth_{视频 ID}.npz")
        
        # 可选：转换为 MP4 格式用于可视化
        保存为 MP4(深度图列表，f"depth_{视频 ID}.mp4")
    
    # 步骤 3: 提取轨迹图 (使用 SpaTracker)
    # ===================================
    FOR 每个视频 in 原始视频数据集:
        
        # 加载 RGB 视频和深度视频
        RGB 视频 = 加载视频 (视频 ["video_path"])          # [1, T, C, H, W]
        深度视频 = 加载视频 (f"depth_{视频 ID}.mp4")      # [T, 1, H, W]
        
        # 初始化 SpaTracker 模型
        跟踪器 = SpaTrackerPredictor(
            checkpoint="spaT_final.pth",
            seq_length=12  # 序列长度
        )
        
        # 运行空间跟踪
        轨迹点，可见性 = 跟踪器 (
            RGB 视频，
            video_depth=深度视频，
            grid_size=50  # 网格密度
        )
        # 轨迹点形状：[T, N, 2], N=轨迹点数量
        
        # 可视化轨迹为热力图
        轨迹热力图 = 可视化轨迹 (轨迹点，RGB 视频)
        # 形状：[T, H, W, 3], RGB 热力图
        
        # 保存轨迹图
        保存为 MP4(轨迹热力图，f"track_{视频 ID}.mp4")
    
    # 步骤 4: 创建训练元数据
    # ====================
    训练数据列表 = []
    
    FOR 每个视频 in 原始视频数据集:
        数据项 = {
            "video": f"video_{视频 ID}.mp4",
            "depth": f"depth_{视频 ID}.mp4",
            "track": f"track_{视频 ID}.mp4",
            "text": 视频 ["text_prompt"]
        }
        训练数据列表。添加 (数据项)
    
    # 保存为 JSON
    保存为 JSON(训练数据列表，"train_data.json")
    
    RETURN 训练数据列表
```

---

## 2. 阶段一：控制信号训练

### 2.1 模型初始化

```python
# ============================================================
# 阶段一：Dual Controller 训练
# 对应文件：train_longvie_control.py
# ============================================================

FUNCTION 阶段一_控制信号训练 (配置参数):
    
    # --------------------------------------------------------
    # 步骤 1: 加载基础模型
    # --------------------------------------------------------
    
    模型配置 = [
        ModelConfig(
            model_id="Wan-AI/Wan2.1-I2V-14B-480P",
            origin_file_pattern="diffusion_pytorch_model*.safetensors"
        ),  # DiT (14B 参数)
        
        ModelConfig(
            model_id="Wan-AI/Wan2.1-I2V-14B-480P",
            origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"
        ),  # T5 文本编码器
        
        ModelConfig(
            model_id="Wan-AI/Wan2.1-I2V-14B-480P",
            origin_file_pattern="Wan2.1_VAE.pth"
        ),  # VAE
        
        ModelConfig(
            model_id="Wan-AI/Wan2.1-I2V-14B-480P",
            origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
        ),  # CLIP 图像编码器
    ]
    
    # 创建 LongVie Pipeline
    Pipeline = LongViePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cpu",
        model_configs=模型配置，
        skip_download=True,
        redirect_common_files=False,
        
        # 关键：加载初始控制器权重
        control_weight_path="./output/init_controller.safetensors",
        control_layers=12,  # 控制网络层数
        Is_train=True
    )
    
    # --------------------------------------------------------
    # 步骤 2: 设置训练模式
    # --------------------------------------------------------
    
    # 指定可训练模型
    可训练模型 = ["dual_controller"]  # 只训练双重控制器
    
    # 冻结其他模型
    FOR 模型名 in ["dit", "text_encoder", "image_encoder", "vae"]:
        模型 = getattr(Pipeline, 模型名)
        FOR 参数 in 模型.parameters():
            参数.requires_grad = False  # 冻结参数
    
    # 配置 LoRA (可选)
    IF 使用 LoRA:
        加载 LoRA(
            target_modules=["q", "k", "v", "o", "ffn.0", "ffn.2"],
            rank=32,
            alpha=1.0
        )
    
    # 梯度检查点 (节省显存)
    启用梯度检查点 (Pipeline.dual_controller)
    
    # --------------------------------------------------------
    # 步骤 3: 准备数据加载器
    # --------------------------------------------------------
    
    数据集 = LongVieControlVideoDataset(
        dataset_base_path=配置。dataset_base_path,
        dataset_metadata_path=配置。dataset_metadata_path,
        height=配置.height,      # 352
        width=配置.width,        # 640
        data_file_keys=["video", "depth", "track"],
        num_workers=8
    )
    
    数据加载器 = DataLoader(
        数据集，
        batch_size=1,  # 通常为 1，通过梯度累积增加等效 batch_size
        shuffle=True,
        num_workers=8
    )
    
    # --------------------------------------------------------
    # 步骤 4: 配置优化器
    # --------------------------------------------------------
    
    优化器 = AdamW(
        Pipeline.dual_controller.parameters(),
        lr=配置.learning_rate,  # 1e-5
        betas=(0.9, 0.999),
        weight_decay=0.01
    )
    
    学习率调度器 = CosineAnnealingLR(
        优化器，
        T_max=配置.num_epochs * len(数据加载器),
        eta_min=1e-6
    )
    
    # 混合精度训练
    缩放器 = GradScaler()
    
    # 分布式训练配置
    IF 多 GPU 训练:
        Accelerator = accelerate.Accelerator()
        Pipeline, 优化器，数据加载器 = Accelerator.prepare(
            Pipeline, 优化器，数据加载器
        )
    
    # --------------------------------------------------------
    # 步骤 5: 训练循环
    # --------------------------------------------------------
    
    全局步数 = 0
    
    FOR epoch IN RANGE(配置.num_epochs):  # 20 个 epoch
        
        PRINT(f"========== Epoch {epoch+1}/{配置.num_epochs} ==========")
        
        FOR batch IN 数据加载器:
            
            # 前向传播 + 损失计算
            WITH 自动混合精度 ():
                损失 = 前向传播 (Pipeline, batch)
            
            # 反向传播
            缩放器.scale(损失).backward()
            
            # 梯度累积
            IF (全局步数 + 1) % 配置.gradient_accumulation_steps == 0:
                
                # 梯度裁剪
                缩放器.unscale_(优化器)
                torch.nn.utils.clip_grad_norm_(
                    Pipeline.dual_controller.parameters(),
                    max_norm=1.0
                )
                
                # 更新参数
                缩放器.step(优化器)
                缩放器.update()
                优化器.zero_grad()
                
                # 更新学习率
                学习率调度器.step()
            
            # 记录日志
            IF 全局步数 % 10 == 0:
                PRINT(f"Step {全局步数}, Loss: {损失.item():.4f}")
                TensorBoard.add_scalar("Loss", 损失.item(), 全局步数)
            
            # 保存检查点
            IF 全局步数 % 配置.save_steps == 0:
                保存检查点 (
                    Pipeline.dual_controller.state_dict(),
                    f"./output/step-{全局步数}.safetensors"
                )
            
            全局步数 += 1
    
    # 保存最终模型
    保存最终模型 (
        Pipeline.dual_controller.state_dict(),
        "./output/final_control.safetensors"
    )
    
    RETURN Pipeline
```

### 2.2 前向传播与损失计算

```python
# ============================================================
# 前向传播伪代码
# 对应文件：train_longvie_control.py -> forward_preprocess()
# ============================================================

FUNCTION 前向传播 (Pipeline, 批次数据):
    
    # --------------------------------------------------------
    # 步骤 1: 准备输入数据
    # --------------------------------------------------------
    
    # CFG 敏感参数 (正负样本)
    正向输入 = {
        "prompt": 批次数据 ["prompt"]  # 文本提示词
    }
    
    负向输入 = {}  # 空提示词用于 CFG
    
    # CFG 不敏感参数 (共享)
    共享输入 = {
        # 视频数据
        "input_video": 批次数据 ["video"],      # 原始视频 [B, T, H, W, C]
        "dense_video": 批次数据 ["dense_video"], # 深度图 [B, T, H, W]
        "sparse_video": 批次数据 ["sparse_video"], # 轨迹图 [B, T, H, W, C]
        
        # 视频尺寸
        "height": 批次数据 ["video"][0].size[1],  # 352
        "width": 批次数据 ["video"][0].size[0],   # 640
        "num_frames": len(批次数据 ["video"]),    # 81
        
        # 训练参数
        "cfg_scale": 1,  # 训练时不使用 CFG
        "tiled": False,
        "rand_device": Pipeline.device,
        
        # 梯度检查点
        "use_gradient_checkpointing": True,
        "use_gradient_checkpointing_offload": False,
        
        # 时间步边界
        "max_timestep_boundary": 1.0,
        "min_timestep_boundary": 0.0,
    }
    
    # 额外输入 (首帧图像)
    IF "input_image" IN 配置.extra_inputs:
        共享输入 ["input_image"] = 批次数据 ["video"][0]  # 首帧
        共享输入 ["dense_image"] = 批次数据 ["dense_video"][0]  # 首帧深度
        共享输入 ["sparse_image"] = 批次数据 ["sparse_video"][0]  # 首帧轨迹
    
    # --------------------------------------------------------
    # 步骤 2: Pipeline Units 处理
    # --------------------------------------------------------
    
    FOR Unit IN Pipeline.units:
        
        # 执行 Unit
        共享输入，正向输入，负向输入 = Pipeline.unit_runner(
            Unit,
            Pipeline,
            共享输入，
            正向输入，
            负向输入
        )
    
    # --------------------------------------------------------
    # 步骤 3: 合并输入
    # --------------------------------------------------------
    
    最终输入 = {**共享输入，**正向输入}
    
    # --------------------------------------------------------
    # 步骤 4: 计算训练损失
    # --------------------------------------------------------
    
    # 获取迭代中需要的模型
    模型字典 = {
        "dit": Pipeline.dit,
        "dual_controller": Pipeline.dual_controller,
        "motion_controller": Pipeline.motion_controller,
        "vace": Pipeline.vace,
        "animate_adapter": Pipeline.animate_adapter
    }
    
    # 调用 training_loss
    损失 = Pipeline.training_loss(
        **模型字典，
        **最终输入
    )
    
    RETURN 损失
```

### 2.3 基础训练损失

```python
# ============================================================
# 训练损失计算 (阶段一)
# 对应文件：diffsynth/pipelines/wan_video_new_longvie.py
# ============================================================

FUNCTION training_loss(Pipeline, **输入):
    
    # --------------------------------------------------------
    # 步骤 1: 采样时间步
    # --------------------------------------------------------
    
    # 从训练时间步中随机采样
    最大时间步边界 = int(输入.get("max_timestep_boundary", 1.0) * Pipeline.scheduler.num_train_timesteps)
    最小时间步边界 = int(输入.get("min_timestep_boundary", 0.0) * Pipeline.scheduler.num_train_timesteps)
    
    时间步索引 = torch.randint(
        最小时间步边界，
        最大时间步边界，
        (1,)  # 单个时间步
    )
    
    时间步 = Pipeline.scheduler.timesteps[时间步索引]
    # 例如：t=500 (总共 1000 个时间步)
    
    # --------------------------------------------------------
    # 步骤 2: 前向扩散过程 (添加噪声)
    # --------------------------------------------------------
    
    # q(x_t | x_0, t): 在潜在表示上添加噪声
    输入 ["latents"] = Pipeline.scheduler.add_noise(
        输入 ["input_latents"],  # 原始视频的 VAE 编码
        输入 ["noise"],          # 随机噪声
        时间步
    )
    
    # --------------------------------------------------------
    # 步骤 3: 计算训练目标
    # --------------------------------------------------------
    
    # 计算速度场目标 (Flow Matching)
    训练目标 = Pipeline.scheduler.training_target(
        输入 ["input_latents"],
        输入 ["noise"],
        时间步
    )
    # 形状：[B, C, T, H, W] = [1, 16, 21, 44, 80]
    
    # --------------------------------------------------------
    # 步骤 4: 去噪网络前向传播
    # --------------------------------------------------------
    
    # DiT + Dual Controller 预测噪声
    噪声预测 = model_fn_wan_video(
        # 迭代模型
        dit=输入 ["dit"],
        dual_controller=输入 ["dual_controller"],
        motion_controller=输入 ["motion_controller"],
        vace=输入 ["vace"],
        animate_adapter=输入 ["animate_adapter"],
        
        # 输入条件
        latents=输入 ["latents"],           # 带噪声的潜在表示
        timestep=时间步，                    # 当前时间步
        context=输入 ["context"],           # 文本嵌入 (T5)
        image_emb=输入.get("image_emb"),    # 图像嵌入 (CLIP)
        control_emb=输入.get("control_emb"), # 控制信号嵌入 ⭐
        history_emb=输入.get("history_emb"), # 历史嵌入
        
        # 控制信号 (通过 Dual Controller 注入)
        depth_map=输入 ["dense_video"],     # 深度图 ⭐
        track_map=输入 ["sparse_video"],    # 轨迹图 ⭐
    )
    # 形状：[1, 16, 21, 44, 80]
    
    # --------------------------------------------------------
    # 步骤 5: 计算 MSE 损失
    # --------------------------------------------------------
    
    # 均方误差损失
    损失 = torch.nn.functional.mse_loss(
        噪声预测.float(),
        训练目标.float()
    )
    
    # --------------------------------------------------------
    # 步骤 6: 时间步加权
    # --------------------------------------------------------
    
    # 根据时间步调整损失权重 (早期时间步更重要)
    损失 = 损失 * Pipeline.scheduler.training_weight(时间步)
    
    RETURN 损失
```

---

## 3. 阶段二：历史一致性增强

### 3.1 模型初始化差异

```python
# ============================================================
# 阶段二：历史一致性训练
# 对应文件：train_longvie_history_control.py
# ============================================================

FUNCTION 阶段二_历史一致性训练 (配置参数，阶段一模型):
    
    # --------------------------------------------------------
    # 步骤 1: 加载第一阶段训练的模型
    # --------------------------------------------------------
    
    模型配置 = [...]  # 与阶段一相同
    
    Pipeline = LongViePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cpu",
        model_configs=模型配置，
        
        # 关键：加载第一阶段训练的权重
        control_weight_path="./output/step-800.safetensors",  # ⭐ 阶段一输出
        dit_weight_path="",  # 可选：加载 DiT 权重
        control_layers=12,
        Is_train=True
    )
    
    # --------------------------------------------------------
    # 步骤 2: 使用时序增强版 Pipeline
    # --------------------------------------------------------
    
    # 注意：这里导入的是 temporal 版本
    FROM diffsynth.pipelines.wan_video_new_longvie_temporal IMPORT LongViePipeline
    
    # temporal 版本包含：
    # - lowpass/highpass 滤波器
    # - 增强的训练损失
    # - 历史一致性 Unit
    
    # --------------------------------------------------------
    # 步骤 3: 准备历史视频数据集
    # --------------------------------------------------------
    
    数据集 = LongVieHistoryVideoDataset(
        dataset_base_path=配置。dataset_base_path,
        dataset_metadata_path=配置。dataset_metadata_path,
        height=配置.height,
        width=配置.width,
        data_file_keys=["video", "depth", "track"],
        
        # 新增：历史视频相关
        include_history=True,
        include_degraded_image=True,
        
        num_workers=8
    )
    
    # 数据增强：
    # 1. 从长视频中随机采样连续片段
    # 2. 第一个片段作为 history_video
    # 3. 第二个片段作为 input_video
    # 4. 对首帧进行退化处理 (模糊、降采样)
    
    # --------------------------------------------------------
    # 步骤 4: 调整学习率
    # --------------------------------------------------------
    
    优化器 = AdamW(
        Pipeline.dual_controller.parameters(),
        lr=配置.learning_rate,  # 5e-6 (比阶段一小)
        betas=(0.9, 0.999),
        weight_decay=0.01
    )
    
    # ... (其他配置与阶段一类似)
```

### 3.2 历史一致性训练损失

```python
# ============================================================
# 训练损失计算 (阶段二 - 时序增强版)
# 对应文件：diffsynth/pipelines/wan_video_new_longvie_temporal.py
# ============================================================

FUNCTION training_loss_temporal(Pipeline, **输入):
    
    # --------------------------------------------------------
    # 步骤 1-3: 与阶段一相同 (时间步采样、加噪、计算目标)
    # --------------------------------------------------------
    
    时间步 = 采样时间步 ()
    输入 ["latents"] = Pipeline.scheduler.add_noise(...)
    训练目标 = Pipeline.scheduler.training_target(...)
    
    # --------------------------------------------------------
    # 步骤 4: 去噪网络前向传播
    # --------------------------------------------------------
    
    噪声预测 = model_fn_wan_video(**输入，timestep=时间步)
    
    # --------------------------------------------------------
    # 步骤 5: 多损失融合 ⭐
    # --------------------------------------------------------
    
    IF 输入 ["deg_flag"]:  # 使用退化图像训练
        
        # --- 损失 1: 加权主损失 ---
        权重 = torch.ones((1, 1, 21, 1, 1), device=噪声预测.device)
        
        # 首帧权重低 (避免过拟合), 后续帧权重高
        权重 [:, :, 0, :, :] = 0.05   # 第一帧
        权重 [:, :, 1, :, :] = 0.325  # 第二帧
        权重 [:, :, 2, :, :] = 0.757  # 第三帧
        
        损失原始 = (噪声预测.float() - 训练目标.float()) ** 2
        主损失 = (损失原始 * 权重).mean()
        
        # --- 损失 2: 高频细节损失 ---
        预测首帧 = 噪声预测 [:, :, :1]  # 第一帧预测
        真值首帧 = 训练目标 [:, :, :1]  # 第一帧真值
        
        # 高通滤波：提取高频细节
        预测高频 = highpass(预测首帧)
        真值高频 = highpass(真值首帧)
        
        高频细节损失 = torch.nn.functional.mse_loss(
            预测高频，
            真值高频
        )
        # 作用：保持视频细节清晰
        
        # --- 损失 3: 历史一致性损失 ⭐ ---
        IF 输入 ["input_history_image"] IS NOT None:
            
            # 历史视频的目标表示
            历史目标 = Pipeline.scheduler.training_target(
                输入 ["input_history_image"],  # 历史末帧的 VAE 编码
                输入 ["noise"] [:, :, :1],
                时间步
            )
            
            # 强制当前首帧与历史末帧一致
            一致性损失 = torch.nn.functional.mse_loss(
                预测首帧，
                历史目标
            )
            # 作用：解决长视频闪烁问题
        
        # --- 损失 4: 退化图像低频损失 ---
        退化目标 = Pipeline.scheduler.training_target(
            输入 ["input_deg_image"],  # 退化后的首帧
            输入 ["noise"] [:, :, :1],
            时间步
        )
        
        # 低通滤波：提取低频结构
        预测低频 = lowpass(预测首帧)
        退化低频 = lowpass(退化目标)
        
        低频结构损失 = torch.nn.functional.mse_loss(
            预测低频，
            退化低频
        )
        # 作用：保持整体结构稳定
        
        # --- 损失融合 ---
        IF 历史视频存在:
            总损失 = (
                主损失 +
                0.2 * 低频结构损失 +      # deg_loss_lp
                0.15 * 高频细节损失 +      # gt_loss_hp
                0.5 * 一致性损失          # consistency_loss_lp ⭐
            )
        ELSE:
            总损失 = (
                主损失 +
                0.2 * 低频结构损失 +
                0.15 * 高频细节损失
            )
    
    ELSE:  # 标准训练
        总损失 = torch.nn.functional.mse_loss(
            噪声预测.float(),
            训练目标.float()
        )
    
    # --------------------------------------------------------
    # 步骤 6: 时间步加权
    # --------------------------------------------------------
    
    总损失 = 总损失 * Pipeline.scheduler.training_weight(时间步)
    
    RETURN 总损失
```

### 3.3 频域分解函数

```python
# ============================================================
# 频域分解工具函数
# 对应文件：diffsynth/pipelines/wan_video_new_longvie_temporal.py
# ============================================================

FUNCTION lowpass(输入张量，scale=0.5):
    """
    低通滤波器：提取低频成分 (平滑部分)
    
    原理：
    1. 下采样到一半分辨率 (去除高频细节)
    2. 上采样回原始分辨率 (恢复低频结构)
    
    参数:
        输入张量：[B, C, T, H, W]
        scale: 缩放因子 (0.5 表示一半分辨率)
    
    返回:
        低频张量：[B, C, T, H, W]
    """
    
    B, C, T, H, W = 输入张量.shape
    
    # 步骤 1: 下采样
    下采样张量 = F.interpolate(
        输入张量，
        scale_factor=(1, scale, scale),  # 时间维度不变，空间维度减半
        mode="trilinear",
        align_corners=False
    )
    # 形状：[B, C, T, H/2, W/2]
    
    # 步骤 2: 上采样
    低频张量 = F.interpolate(
        下采样张量，
        size=(T, H, W),  # 恢复原始尺寸
        mode="trilinear",
        align_corners=False
    )
    # 形状：[B, C, T, H, W]
    
    RETURN 低频张量


FUNCTION highpass(输入张量，scale=0.5):
    """
    高通滤波器：提取高频成分 (细节部分)
    
    原理:
    高频 = 原始 - 低频
    
    参数:
        输入张量：[B, C, T, H, W]
        scale: 缩放因子
    
    返回:
        高频张量：[B, C, T, H, W]
    """
    
    高频张量 = 输入张量 - lowpass(输入张量，scale=scale)
    
    RETURN 高频张量
```

---

## 4. 核心算法详解

### 4.1 Dual Controller 架构

```python
# ============================================================
# Dual Controller 前向传播
# 对应文件：diffsynth/models/wan_video_dit_dual_control.py
# ============================================================

CLASS WanModelDualControl (nn.Module):
    """
    双重控制器 DiT 模型
    
    在原始 DiT 基础上添加:
    1. 深度图控制分支
    2. 轨迹图控制分支
    3. 自适应特征融合
    """
    
    FUNCTION __init__():
        super().__init__()
        
        # 1. 基础 DiT 结构 (30 层 Transformer)
        self.blocks = nn.ModuleList([...])
        
        # 2. 深度图控制分支 ⭐
        self.depth_control_net = ControlNetBranch(
            in_channels=1,       # 深度图单通道
            out_channels=1024    # 匹配 DiT 维度
        )
        
        # 3. 轨迹图控制分支 ⭐
        self.track_control_net = ControlNetBranch(
            in_channels=3,       # 轨迹图 RGB
            out_channels=1024
        )
        
        # 4. 特征融合模块
        self.fusion_module = AdaptiveFusion(
            dim=1024,
            num_heads=16
        )
    
    FUNCTION forward(x, t, context, depth_map, track_map):
        """
        参数:
            x: 潜在表示 [B, C, T, H, W]
            t: 时间步
            context: 文本嵌入 [B, L, D]
            depth_map: 深度图 [B, T, H, W]
            track_map: 轨迹图 [B, T, H, W, C]
        """
        
        # 步骤 1: 基础 DiT 前向传播
        FOR block IN self.blocks:
            x = block(x, t, context)
        
        # 步骤 2: 提取控制特征
        深度特征 = self.depth_control_net(depth_map)
        # 形状：[B, 1024, T, H, W]
        
        轨迹特征 = self.track_control_net(track_map)
        # 形状：[B, 1024, T, H, W]
        
        # 步骤 3: 自适应融合
        控制特征 = self.fusion_module(深度特征，轨迹特征)
        # 加权融合：α*depth + β*track
        
        # 步骤 4: 注入到主干
        x = x + 控制特征
        
        RETURN x
```

### 4.2 历史一致性机制

```python
# ============================================================
# 历史视频编码
# 对应文件：diffsynth/pipelines/wan_video_new_longvie_temporal.py
# ============================================================

CLASS WanVideoUnit_HistoryVideoEmbedder (PipelineUnit):
    """
    历史视频编码单元
    
    作用:
    1. 编码历史视频的最后一帧
    2. 提供给当前片段作为时序参考
    """
    
    FUNCTION __init__():
        super().__init__(
            input_params=("history_video", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("vae",)
        )
    
    FUNCTION process(Pipeline, history_video, ...):
        
        # 检查是否有历史视频
        IF history_video == []:
            RETURN {"history_latents": None}
        
        # 加载 VAE 到设备
        Pipeline.load_models_to_device(["vae"])
        
        # 预处理历史视频
        history_video = Pipeline.preprocess_video(history_video)
        # 形状：[B, T, C, H, W]
        
        # VAE 编码
        history_latents = Pipeline.vae.encode(
            history_video,
            device=Pipeline.device,
            tiled=False
        )
        # 形状：[B, C_latent, T_latent, H_latent, W_latent]
        # 例如：[1, 16, 2, 44, 80]
        
        # 只取最后一帧
        history_latents = history_latents [:, :, -1:]
        # 形状：[1, 16, 1, 44, 80]
        
        RETURN {"history_latents": history_latents}


CLASS WanVideoUnit_HistoryImageEmbedder (PipelineUnit):
    """
    历史图像编码单元 (阶段二新增)
    
    作用:
    1. 编码历史视频的最后一帧为单帧图像
    2. 用于历史一致性损失计算
    """
    
    FUNCTION __init__():
        super().__init__(
            input_params=("history_video", "noise", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("vae",)
        )
    
    FUNCTION process(Pipeline, history_video, ...):
        
        IF history_video == []:
            RETURN {"input_history_image": None}
        
        # 取历史视频的最后一帧
        最后一帧 = [history_video[-1]]
        
        # 预处理
        最后一帧 = Pipeline.preprocess_video(最后一帧)
        
        # VAE 编码
        history_latents = Pipeline.vae.encode(
            最后一帧，
            device=Pipeline.device
        )
        
        RETURN {"input_history_image": history_latents}
```

### 4.3 自回归长视频生成

```python
# ============================================================
# 自回归生成流程
# 对应文件：inference.py
# ============================================================

FUNCTION 自回归生成 (Pipeline, 提示词，控制信号，历史视频列表):
    """
    生成长视频的自回归流程
    
    原理:
    1. 将长视频切分为多个片段 (每段 81 帧)
    2. 每个片段依赖前一片段的最后几帧
    3. 保持噪声连续性
    
    参数:
        Pipeline: LongViePipeline
        提示词：文本描述
        控制信号：深度图 + 轨迹图序列
        历史视频列表：之前的视频片段
    """
    
    生成的视频片段列表 = []
    历史窗口 = []  # 滑动窗口 (最近 8 帧)
    噪声种子 = None  # 保持连续性
    
    FOR i, 当前控制 IN ENUMERATE(控制信号):
        
        PRINT(f"生成第 {i+1} 个片段...")
        
        # 步骤 1: 准备输入
        输入参数 = {
            "prompt": 提示词，
            "input_image": 历史窗口 [-1] IF 历史窗口 ELSE 首帧图像，
            "dense_video": 当前控制 ["depth"],  # 深度图
            "sparse_video": 当前控制 ["track"], # 轨迹图
            "history": 历史窗口，               # 历史视频
            "noise": 噪声种子，                 # 连续噪声
            "num_frames": 81,
            "height": 352,
            "width": 640,
        }
        
        # 步骤 2: 生成当前片段
        当前片段 = Pipeline(**输入参数)
        # 形状：[81, H, W, C]
        
        # 步骤 3: 更新历史窗口 (滑动窗口)
        历史窗口。extend(当前片段)
        IF len(历史窗口) > 8:
            历史窗口 = 历史窗口 [-8:]  # 保留最近 8 帧
        
        # 步骤 4: 更新噪声种子 (保持连续性)
        噪声种子 = 提取噪声尾部 (当前片段)
        
        # 步骤 5: 保存片段
        保存视频 (当前片段，f"segment_{i}.mp4")
        
        生成的视频片段列表。append(当前片段)
    
    # 拼接所有片段
    完整视频 = CONCATENATE(生成的视频片段列表)
    
    RETURN 完整视频
```

---

## 5. 完整训练伪代码

### 5.1 主训练流程

```python
# ============================================================
# LongVie 2 完整训练流程
# ============================================================

FUNCTION main():
    """
    LongVie 2 训练主函数
    
    执行流程:
    1. 数据准备
    2. 阶段一：控制信号训练
    3. 阶段二：历史一致性增强
    4. 模型评估
    """
    
    PRINT("=" * 60)
    PRINT("LongVie 2 训练开始")
    PRINT("=" * 60)
    
    # ========================================================
    # 阶段 0: 数据准备
    # ========================================================
    
    PRINT("\n[阶段 0] 准备训练数据...")
    
    # 0.1 下载/收集原始视频
    原始视频列表 = 收集视频数据 (
        数据源=["YouTube", "本地数据集"],
        最小分辨率=(640, 352),
        最小帧数=81,
        帧率=16
    )
    
    # 0.2 提取深度图
    PRINT("提取深度图...")
    FOR 视频 IN 原始视频列表:
        深度图 = VideoDepthAnything.推理 (视频)
        保存 (深度图，f"depth_{视频.id}.mp4")
    
    # 0.3 提取轨迹图
    PRINT("提取轨迹图...")
    FOR 视频 IN 原始视频列表:
        轨迹图 = SpaTracker.推理 (视频，深度图)
        保存 (轨迹图，f"track_{视频.id}.mp4")
    
    # 0.4 创建元数据
    训练数据 = [
        {
            "video": f"video_{i}.mp4",
            "depth": f"depth_{i}.mp4",
            "track": f"track_{i}.mp4",
            "text": 视频.prompt
        }
        FOR i, 视频 IN ENUMERATE(原始视频列表)
    ]
    
    保存为 JSON(训练数据，"train_data.json")
    
    PRINT(f"✓ 数据准备完成，共 {len(训练数据)} 个样本")
    
    # ========================================================
    # 阶段一：控制信号训练
    # ========================================================
    
    PRINT("\n" + "=" * 60)
    PRINT("[阶段一] Dual Controller 训练")
    PRINT("=" * 60)
    
    配置 1 = {
        "training_script": "train_longvie_control.py",
        "dataset_base_path": "data/example_video_dataset",
        "dataset_metadata_path": "train_data.json",
        "height": 352,
        "width": 640,
        "learning_rate": 1e-5,
        "num_epochs": 20,
        "gradient_accumulation_steps": 2,
        "trainable_models": "dual_controller",
        "data_file_keys": "video,depth,track",
        "extra_inputs": "input_image",
        "output_path": "./output/train/LongVie_control",
        "save_steps": 100,
    }
    
    # 启动训练
    阶段一模型 = 启动训练 (配置 1)
    
    PRINT(f"✓ 阶段一完成，模型保存到 {配置 1['output_path']}")
    
    # ========================================================
    # 阶段二：历史一致性增强
    # ========================================================
    
    PRINT("\n" + "=" * 60)
    PRINT("[阶段二] 历史一致性训练")
    PRINT("=" * 60)
    
    配置 2 = {
        "training_script": "train_longvie_history_control.py",
        "dataset_base_path": "data/example_video_dataset",
        "dataset_metadata_path": "train_data.json",
        "height": 352,
        "width": 640,
        "learning_rate": 5e-6,  # 更小的学习率
        "num_epochs": 20,
        "gradient_accumulation_steps": 2,
        "trainable_models": "dual_controller",
        "data_file_keys": "video,depth,track",
        "extra_inputs": "input_image",
        "output_path": "./output/train/LongVie_control_history",
        "save_steps": 100,
        
        # 加载阶段一模型
        "control_weight_path": "./output/train/LongVie_control/step-800.safetensors",
    }
    
    # 启动训练
    阶段二模型 = 启动训练 (配置 2)
    
    PRINT(f"✓ 阶段二完成，模型保存到 {配置 2['output_path']}")
    
    # ========================================================
    # 模型评估
    # ========================================================
    
    PRINT("\n" + "=" * 60)
    PRINT("[评估] 测试生成效果")
    PRINT("=" * 60)
    
    # 加载最终模型
    Pipeline = LongViePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        control_weight_path=配置 2["output_path"] + "/final_model.safetensors"
    )
    
    # 测试样本
    测试样本 = [
        {
            "prompt": "一只猫在草地上奔跑",
            "input_image": 加载图像 ("test_cat.png"),
            "dense_video": 加载视频 ("test_depth.mp4"),
            "sparse_video": 加载视频 ("test_track.mp4"),
        }
    ]
    
    # 生成测试视频
    FOR 样本 IN 测试样本:
        生成视频 = Pipeline(
            prompt=样本 ["prompt"],
            input_image=样本 ["input_image"],
            dense_video=样本 ["dense_video"],
            sparse_video=样本 ["sparse_video"],
            num_frames=81,
            height=352,
            width=640,
        )
        
        保存视频 (生成视频，f"test_output_{样本.id}.mp4")
        PRINT(f"✓ 测试样本 {样本.id} 生成完成")
    
    PRINT("\n" + "=" * 60)
    PRINT("LongVie 2 训练全部完成！")
    PRINT("=" * 60)
```

### 5.2 训练配置示例

```yaml
# ============================================================
# Accelerate 配置文件 (accelerate_config_14B.yaml)
# ============================================================

compute_environment: LOCAL_MACHINE
debug: false
distributed_type: MULTI_GPU  # 多 GPU 分布式
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: bf16  # 混合精度训练 (BF16)
num_machines: 1  # 机器数量
num_processes: 8  # GPU 数量 (例如 8×A100)
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false

# 显存优化
gradient_checkpointing: true  # 梯度检查点
deepspeed_config:
  gradient_accumulation_steps: 2
  offload_optimizer_device: none
  offload_param_device: none
  zero3_init_flag: false
  zero_stage: 2  # ZeRO-2 优化
```

### 5.3 训练命令

```bash
# ============================================================
# 训练命令
# ============================================================

# 阶段一：控制信号训练
accelerate launch \
    --main_process_port 21455 \
    --config_file accelerate_config_14B.yaml \
    train_longvie_control.py \
    --dataset_base_path data/example_video_dataset \
    --dataset_metadata_path train_data.json \
    --height 352 \
    --width 640 \
    --learning_rate 1e-5 \
    --num_epochs 20 \
    --gradient_accumulation_steps 2 \
    --trainable_models "dual_controller" \
    --data_file_keys 'video,depth,track' \
    --extra_inputs input_image \
    --output_path "./output/train/LongVie_control" \
    --save_steps 100

# 阶段二：历史一致性训练
accelerate launch \
    --main_process_port 21455 \
    --config_file accelerate_config_14B.yaml \
    train_longvie_history_control.py \
    --dataset_base_path data/example_video_dataset \
    --dataset_metadata_path train_data.json \
    --height 352 \
    --width 640 \
    --learning_rate 5e-6 \
    --num_epochs 20 \
    --gradient_accumulation_steps 2 \
    --trainable_models "dual_controller" \
    --data_file_keys 'video,depth,track' \
    --extra_inputs input_image \
    --output_path "./output/train/LongVie_control_history" \
    --save_steps 100
```

---

## 📊 训练时间估算

| 阶段 | GPU 配置 | 数据集大小 | 训练时间 |
|------|---------|-----------|---------|
| 阶段一 | 8×A100 | 1000 视频 | ~2 天 |
| 阶段二 | 8×A100 | 1000 视频 | ~2 天 |
| **总计** | **8×A100** | **1000 视频** | **~4 天** |

---

## 🎯 关键要点总结

1. **两阶段训练**:
   - 阶段一：学习基础控制能力
   - 阶段二：增强时序一致性

2. **核心创新**:
   - Dual Controller (深度 + 轨迹控制)
   - 历史一致性机制 (滑动窗口)
   - 频域分解损失 (lowpass/highpass)

3. **显存优化**:
   - 梯度检查点
   - 混合精度训练 (BF16)
   - 梯度累积

4. **数据流程**:
   - 原始视频 → 深度图 → 轨迹图 → 训练

---

**文档版本**: v1.0  
**最后更新**: 2026-03-10  
**基于仓库**: `e:\trae\LongVie`
