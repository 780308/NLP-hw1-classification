# RNN-TRAIN-003 冻结实验协议

目标：仅比较学习率策略。所有结论使用三种子内部验证集，不访问 `data/val`。

- 表示：`REP-006-FUSION`，`[190,7,7]`；RGB 水平镜像后重提取特征的既有缓存。
- 架构：`RNN2D-ARCH-BASE`（代码标识 `RNN2D-ARCH-L`），`C=320, H=80, blocks=3`，1,992,642 个可训练参数，双向 tanh `nn.RNN`，无 dropout。
- 划分：复用种子 42、123、2026 的已有划分。每种子 1800 原图 + 1800 镜像训练特征，200 原图内部验证特征。仅从 3600 张训练图拟合归一化参数。
- 通用设置：AdamW、权重衰减 `1e-4`、批量 32、交叉熵、最多 40 轮、验证损失早停耐心 8、梯度裁剪 1.0。检查点以最高验证准确率优先，同分取较低验证损失。
- T0 `RNN-LR-T0-CONST-3E4`：从 `report/rnn_training/experiments/RNN-TRAIN-T1-FLIP/` 导入，恒定 `3e-4`，不重训。
- T1 `RNN-LR-T1-CONST-1P5E4`：恒定 `1.5e-4`。
- T2 `RNN-LR-T2-COSINE`：初始 `3e-4`，`CosineAnnealingLR(T_max=20, eta_min=1e-5)`，每轮训练结束后步进。
- T3 `RNN-LR-T3-PLATEAU`：初始 `3e-4`，`ReduceLROnPlateau(mode="min",factor=0.3,patience=2,min_lr=1e-5)`，每次验证后按验证损失步进。

共新增九次正式运行。每轮记录实际训练学习率；T2/T3 检查点保留调度器配置与状态。T0/T1/T2/T3 结果按照用户给定的均值、最差种子和稳定性门槛选出一种最终学习率政策。当前阶段只生成选定政策的共同轮次诊断，不决定最终轮次，也不训练最终全数据模型。
