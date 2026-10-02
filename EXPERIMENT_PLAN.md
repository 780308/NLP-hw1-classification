# 猫狗图像二分类实验计划

## 1. 目标与现状

按 `synopsis.txt` 的要求，分别训练 DNN、CNN、RNN 完成猫狗分类，并提交可运行代码和基于真实结果的实验报告。500 张测试图全部分类正确（100%）是冲刺目标，不能预先承诺；每错 1 张，总准确率降低 0.2 个百分点。

已核查的数据位于 `data/train` 和 `data/val`：训练集猫、狗各 1000 张，测试集各 250 张。图片为平铺的 `cat.N.jpg`、`dog.N.jpg`，不是类别子目录。2500 张图片的 JPEG 头均可读取，未发现字节级重复；完整解码仍须在实现数据加载器时验证。原始数据保持只读。标签固定为 `cat=0`、`dog=1`。

运行平台为 WSL Ubuntu 24.04、Python 3.12，NVIDIA GeForce RTX 3050 Ti Laptop GPU（4 GB 显存）。实验优先使用 CUDA，必要时降批量，不静默改用 CPU。环境已建于 `/home/dongxianghong/.venvs/hw1`；已验证 PyTorch `2.5.1+cu121`、TorchVision `0.20.1+cu121` 可导入，且 CUDA 张量运算成功。

## 2. 架构调研与选择

本节和第 3–4 节保留最初的候选方案与通用训练设想。已完成实验的实际结构、输入、归一化和命令以各自的冻结协议、README 及下方更新记录为准；CNN-FINAL-001 实际采用 `REP-006-FUSION` 和浅层 CNN-ARCH-C。

| 实验 | 计划结构 | 输入 | 用途与理由 |
|---|---|---|---|
| DNN 主实验 | 展平 → 256 → 64 → 2 的 MLP，ReLU、Dropout 0.3 | RGB 64×64 | 保持纯全连接结构，参数量适中；自然图像的空间关系会在展平时丢失 |
| CNN 主实验 | 四个 VGG 风格卷积块，通道 32/64/128/256，每块两层 3×3 卷积、BatchNorm、ReLU、池化；全局平均池化 → 2 | RGB 128×128 | 从头训练基本卷积网络，满足课程对 CNN 原理的考察 |
| RNN 主实验 | 每张图按行转成 `[B,64,192]`，两层双向 `nn.RNN`，拼接最终双向状态 → 2 | RGB 64×64 | 明确保留循环计算，避免把单个展平向量伪装成序列 |
| 冲分扩展 | ImageNet 预训练 EfficientNet-B0，更换二分类头并微调末端特征块 | RGB 224×224 | 借助迁移学习提高小数据集表现；单独汇报，不并入三模型的公平比较 |

在 TorchVision 官方权重文档中，[EfficientNet-B0](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b0.html) 约 529 万参数、0.39 GFLOPs、ImageNet Top-1 77.692%；[ResNet18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html) 约 1169 万参数、1.81 GFLOPs、69.758%；[MobileNetV3-Small](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html) 约 254 万参数、0.06 GFLOPs、67.668%。因此在 4 GB 显存与准确率优先的约束下，扩展实验选 EfficientNet-B0。上述指标属于 ImageNet，**不能当作本作业的预测准确率**。迁移学习按 [PyTorch 教程](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html) 的冻结特征提取器与局部微调思路执行。

不为 DNN/RNN 借用预训练 CNN 特征，否则三类主模型的输入表示不同，无法清楚检验各自架构。

## 3. 数据、训练与测试协议

1. 仅从 `data/train` 按类别和种子 42 固定划分 1800 张训练、200 张内部验证。`data/val` 作为最终测试集；模型和超参数锁定后，每个最终模型只评估一次。失败样本分析与调参仅使用内部验证集。
2. 三个主实验使用相同的标签映射、归一化（各通道均值/标准差均为 0.5）、训练时随机水平翻转、两类原始 logits、`CrossEntropyLoss` 和评估代码。验证及测试变换完全确定；模型输入尺寸差异在报告中注明。
3. 主实验统一采用 AdamW、学习率 `1e-3`、权重衰减 `1e-4`、批量 32、最多 30 轮、按验证损失连续 5 轮不改善而早停。检查点按验证准确率最高选择，准确率相同则选择验证损失较低的轮次。固定 Python、NumPy、PyTorch 与数据加载器的随机种子。
4. 扩展实验使用 EfficientNet-B0 权重配套的 ImageNet 归一化，训练时温和的随机裁剪和水平翻转。先冻结骨干、训练分类头 3 轮（学习率 `1e-3`）；再解冻末端特征块与分类头，最多 10 轮（骨干 `1e-5`、分类头 `1e-4`）。CUDA 上使用混合精度，批量 16；显存不足时降至 8 并记录。根据内部验证确定训练轮数后，在全部 2000 张训练图上按固定轮数重新训练该扩展模型，再测试一次。
5. 统一记录训练/内部验证损失与准确率、最终测试总体准确率、猫准确率、狗准确率、混淆矩阵、最佳轮次、参数量、实际耗时及完整配置。每类准确率为该类正确数除以该类总数。不得估计或编造未运行的结果。

## 4. 实施顺序与文件约定

1. 在 WSL 用户目录建立虚拟环境，安装 `requirements.txt` 所列依赖，并核查 PyTorch CUDA 可用性。已完成；使用以下命令复现：

   ```bash
   cd /mnt/e/CircuitWizard/2-NLP/hw/hw1
   set -o pipefail
   curl --retry 3 -LsSf https://astral.sh/uv/install.sh | sh
   ~/.local/bin/uv venv ~/.venvs/hw1 --python /usr/bin/python3
   ~/.local/bin/uv pip install --python ~/.venvs/hw1/bin/python torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
   ~/.local/bin/uv pip install --python ~/.venvs/hw1/bin/python -r requirements.txt
   source ~/.venvs/hw1/bin/activate
   python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
   ```
2. 实现基于文件名的 Dataset、共享训练/评估流程与三个主模型。先完整解码图片，再逐模型验证一批数据的形状、前向、损失、反向传播；各跑一次短训练，验证检查点可重载。
3. 在同一内部划分上依次训练三个主模型，再执行一次 EfficientNet-B0 扩展实验。单张 4 GB GPU 上顺序训练，不进行大范围盲目搜索。
4. 固定全部选择后评估 `data/val`，生成准确率比较表、曲线和混淆矩阵，完成 README 与实验报告。报告应说明架构、预处理、训练条件、真实结果、误差和观察。

计划接口：`python train.py --model {dnn,cnn,rnn,efficientnet_b0} --data-dir data`；提供轮数、批量、学习率、种子、设备、工作进程等参数。`python evaluate.py --checkpoint <path> --data-dir data/val` 重载检查点复核指标。最终以实际实现和 README 中可执行的命令为准。

生成文件放在 `outputs/<model>/`，包括最佳检查点、配置、逐轮指标、曲线、最终测试指标与混淆矩阵；报告放在 `report/`。大权重、缓存与临时文件不提交到版本控制。

后续正式实验以 `DNN-001`、`CNN-001` 等唯一编号命名，使用独立输出目录避免覆盖。每次把小型指标产物、配置和划分清单归档到 `report/experiments/<编号>/`，更新 `report/experiment_log.md` 与总报告后，使用包含编号的提交消息 `commit` 并 `push` 到目标仓库 `main`。公开仓库不提交原始 `data/` 图片或 `outputs/` 检查点；详细规则见 README 和实验索引。

## 5. 验收与更新记录

- [x] 核查任务书、目录、类数、文件名和设备。
- [x] 建立 WSL 虚拟环境，安装依赖并确认 CUDA；`uv pip check` 通过。
- [x] 完整解码 2500 张图像，无解码失败。
- [x] 实现并验证 DNN 的前向、训练、检查点重载及最终测试。
- [x] 实现并验证 CNN 的前向、训练、检查点重载及最终测试；RNN 的三种架构、训练与检查点重载已验证，最终测试待完成。
- [ ] 完成三个主实验与单独的预训练扩展实验。
- [ ] 最终测试一次，保存指标、图表和可重载检查点。
- [ ] 更新 README、实验报告和下表；结论只根据实测结果撰写。

| 模型 | 总体准确率 | 猫准确率 | 狗准确率 | 最佳轮次 | 备注 |
|---|---:|---:|---:|---:|---|
| DNN | 62.60% | 70.80% | 54.40% | 11 | 主实验；18 轮早停，详见 `report/experiment_report.md` |
| DNN-FINAL-001 | 73.80% | 65.60% | 82.00% | 固定 3 | 最终 DNN；使用 REP-006-FUSION 与 DNN-ARCH-C |
| CNN-FINAL-001 | 81.20% | 81.20% | 81.20% | 固定 4 | REP-006-FUSION、CNN-ARCH-C、原图镜像 |
| RNN | 待测 | 待测 | 待测 | 待测 | 已完成架构选择，保留测试未测 |
| EfficientNet-B0 | 待测 | 待测 | 待测 | 待测 | 预训练扩展，单独比较 |

更新时记录日期、改动原因、配置和对应输出目录；保留失败或提前终止实验的真实情况。

2026-09-30：创建本计划和 `requirements.txt`；在 WSL 建立虚拟环境。已安装 NumPy `2.5.2`、Pillow `12.3.0`、Matplotlib `3.11.2`；PyTorch 识别 RTX 3050 Ti，GPU 矩阵运算与包依赖检查均通过。

2026-09-30：完成 DNN 主实验。1800/200 内部训练/验证划分，最佳检查点为第 11 轮，内部验证准确率 69.50%；第 18 轮早停。一次最终测试在 500 张图上的总体准确率 62.60%，猫 70.80%，狗 54.40%。原始指标见 `outputs/dnn/`，持续更新报告见 `report/experiment_report.md`。CNN、RNN 尚未运行。

2026-10-01：新增独立的 Stage I 手工空间表示实验协议，详见 `STAGE1-GUIDELINES.md`。S1A-001 已按固定探针比较初始表示，并在规则触发后完成一次融合实验；接下来仅对入选的融合与 HOG+LBP 表示做三种子确认。Stage I 不使用 `data/val`，也不替代原始图像的 CNN/RNN 主实验。

2026-10-01：S1B-001 已完成。融合表示在三个内部划分的固定 MLP 上为 77.50% ± 2.18%，HOG+LBP 为 76.17% ± 0.29%，原始像素为 62.67% ± 1.26%。按预设规则选定融合 `[190,7,7]`；详见 `report/stage1/stage1_summary.md`。纠正预处理对照的混杂因素后，严格几何对照为 66.00%。Stage I 到此停止，不开展 Stage II 或测试集评估。

2026-10-01：`DNN-ARCH-001` 已完成。固定 `REP-006-FUSION`，仅比较 BASE 与三种新纯全连接架构；新架构共九次训练，使用既有三个内部划分和原训练协议。按预设规则选定 C（三种子 78.17% ± 1.26%，最差种子 77.00%），详见 `report/dnn_architecture/architecture_summary.md`。该阶段未执行全量训练或 `data/val` 评估。

2026-10-01：`DNN-FINAL-001` 已完成。根据已有 C 架构三个曲线在相同固定轮次的平均验证准确率，最终测试前将原 6 轮建议修订为 3 轮并提交冻结协议。全部 2000 张训练图训练 3 轮后，在 500 张 `data/val` 上一次评估：总体 73.80%、猫 65.60%、狗 82.00%。完整配置、归一化、预测和审计记录见 `report/dnn_final/`。CNN、RNN 未启动。

2026-10-01：`CNN-ARCH-001` 已完成。固定 Stage I 的 `[190,7,7]` 融合特征图及三份内部划分，仅比较三种浅层 CNN 架构，九次训练均通过审计。按预设均值差规则选定 C：79.83% ± 2.84%，最差种子 77.50%；详见 `report/cnn_architecture/architecture_summary.md`。这只是架构选择，未进行 CNN 最终评估、训练策略搜索或 RNN 实验。

2026-10-01：`CNN-TRAIN-001` 已完成。冻结 CNN-ARCH-C 与 REP-006-FUSION，以历史 C 运行作为 T0，在相同三份 1800/200 内部划分上分别检验原图水平镜像后重提特征、Dropout2d(0.10)、余弦学习率，共九次新训练。T0/T1/T2/T3 的均值分别为 79.83%/80.50%/80.00%/80.00%；T1 因均值提高 0.67 个百分点且类别差距缩小 3.33 个百分点，按预设规则入选。没有两项达到 +1.0 个百分点强改进门槛，组合实验不符合启动条件。汇总与审计见 `report/cnn_training/`。未访问 `data/val`，未执行最终 CNN 训练或 RNN 实验。

2026-10-02：`CNN-FINAL-001` 已完成。评估前提交冻结协议与实现 `3200fba`；复用原始与镜像 REP-006 特征，以全部 4000 张特征图拟合归一化并训练无 Dropout 的 CNN-ARCH-C，种子 42，恰好 4 轮。重载第 4 轮检查点后，对 500 张 `data/val` 原图一次评估：损失 0.4389，总体、猫、狗准确率均 81.20%，Macro F1 81.20%。结果、预测与审计见 `report/cnn_final/`；没有基于保留集继续调整 CNN。RNN 仍待实验。

2026-10-02：`RNN-ARCH-001` 已完成。固定 REP-006 和三份既有内部划分，比较单向行序列、双向行序列、双向 49 单元序列的标准 `nn.RNN`；九次正式运行均通过审计。三种子均值分别为 75.33%、76.00%、75.67%，C 的最差种子 75.00% 比均值第一的 B 高 1.50 个百分点；按预设规则选择 C。详见 `report/rnn_architecture/architecture_summary.md`。本阶段没有访问 `data/val`，没有启动 RNN 最终训练或训练策略搜索。

2026-10-02：`RNN-ARCH-002` 已完成。冻结 REP-006 与原三份划分，比较 S/M/L 三种规模的双轴残差标准 `nn.RNN`，每种子三次共九次运行，审计通过。三种子内部验证均值依次为 80.33% ± 3.01%、79.50% ± 1.32%、81.83% ± 4.31%；按冻结的 0.75 个百分点均值差规则选定 L，最差种子 78.00%，Category A。建议下一阶段仅研究有限的 RNN 训练策略；本任务未访问 `data/val`，未做最终 RNN 训练或修改表示。详见 `report/rnn2d_architecture/architecture_summary.md`。

2026-10-02：`RNN-TRAIN-001` 已完成。固定 RNN2D-L、REP-006 与三份内部划分，导入旧 L 的 T0 作为基线，不重训；T1 RGB 镜像增强、T2 残差路径 Dropout 0.10、T3 权重衰减 `5e-4` 各运行三种子，共九次新运行并通过审计。T0/T1/T2/T3 的内部验证均值依次为 81.83%/84.00%/81.17%/82.17%，选定 T1，最差种子 80.00%，Outcome A。T3 同时满足稳定性改进门槛，故有一次 T1+T3 组合的资格，但本阶段未运行；后续可决定组合或另立受控 RNN 架构扩展。完整记录见 `report/rnn_training/training_strategy_summary.md`。未访问 `data/val`，未做最终 RNN 训练。
