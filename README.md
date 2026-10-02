# 猫狗图像分类：DNN、CNN、RNN 实验

本项目按 `synopsis.txt` 实现猫狗二分类。已完成最终 DNN `DNN-FINAL-001`、CNN 架构比较 `CNN-ARCH-001`、CNN 训练策略比较 `CNN-TRAIN-001` 和最终 CNN `CNN-FINAL-001`；RNN 尚未运行。500 张保留图上的最终 DNN 准确率为 **73.80%**，最终 CNN 为 **81.20%**。实验方案见 [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md)，持续更新的实测报告见 [report/experiment_report.md](report/experiment_report.md)。

每次实验的编号、配置、结果与对应 Git 提交方式记录在 [report/experiment_log.md](report/experiment_log.md)。公开仓库不包含原始 `data/` 图片或 `outputs/` 检查点；运行前需把课程提供的数据放到下述目录。

## 数据与环境

数据保持原样放在 `data/train/` 和 `data/val/`。两个目录均为平铺的 `cat.N.jpg`、`dog.N.jpg` 文件；程序固定 `cat=0`、`dog=1`。`data/train` 有 2000 张，`data/val` 有 500 张。后者曾用于历史 DNN-001 评估，未参与 Stage I 表示选择或 DNN 架构选择；最终 DNN 配置在本次保留集评估前冻结。

以下命令在 WSL Ubuntu 中，从项目目录 `/mnt/e/CircuitWizard/2-NLP/hw/hw1` 执行。已有环境可直接激活；首次安装命令列在实验计划中。

```bash
cd /mnt/e/CircuitWizard/2-NLP/hw/hw1
source ~/.venvs/hw1/bin/activate
python train.py --model dnn --data-dir data
python evaluate.py --checkpoint outputs/dnn/best_model.pt --data-dir data/val
```

训练命令默认使用种子 42、内部训练/验证划分 1800/200、批量 32、最多 30 轮、早停耐心 5 轮。可使用 `python train.py --help` 查看可配置项。优先自动选 CUDA，其次 MPS、CPU。模型选择只依据内部验证集；不要根据 `data/val` 的结果修改模型后仍把它称为一次独立测试。

## 模型与输出

- DNN：64×64 RGB 图像展平后进入 `12288→256→64→2` 的纯全连接网络。
- CNN：在固定 `[190,7,7]` 手工特征图上选定 CNN-ARCH-C 和原图水平镜像策略；最终使用 4000 张原图及镜像特征图训练 4 轮，保留集准确率 81.20%。
- RNN：计划中的按行输入 `nn.RNN` 网络，待实现。

每个模型的运行产物放在 `outputs/<model>/`，包括 `best_model.pt`、`config.json`、`split.json`、`history.csv`、`train_summary.json`、`test_metrics.json` 和图表。`outputs/` 不纳入版本控制；报告引用的图另存于 `report/figures/`。检查点包含结构配置、类别映射、最佳轮次和内部验证指标，可由 `evaluate.py` 重载。

完整依赖列于 `requirements.txt`。当前 WSL 环境已验证 PyTorch `2.5.1+cu121`、TorchVision `0.20.1+cu121` 和 RTX 3050 Ti 的 CUDA 运算。

## 后续实验的 Git 工作流

新实验使用独立编号和输出目录，例如 `python train.py --model dnn --data-dir data --output-dir outputs/dnn/dnn-002`，避免覆盖 DNN-001。实验完成后，将配置、划分、逐轮记录、汇总和评估 JSON 复制到 `report/experiments/dnn-002/`，更新实验索引、总报告和计划，再执行 `git add`、带编号的 `git commit` 和 `git push origin main`。由 `git log --grep=dnn-002` 可定位当次代码和报告。正式实验未完成或失败也应记录状态；不使用已看过的测试结果选择后续模型。

## 第一阶段：手工空间表示

实验协议见 [STAGE1-GUIDELINES.md](STAGE1-GUIDELINES.md)。第一阶段只使用 `data/train`，共用固定的 90/10 内部划分；`data/val` 不参与表示选择。运行环境仍为 WSL 的 `~/.venvs/hw1`。

```bash
source ~/.venvs/hw1/bin/activate
python handcrafted_feature_experiments.py smoke
python handcrafted_feature_experiments.py build-cache
python handcrafted_feature_experiments.py validate-cache
python handcrafted_feature_experiments.py tiny-overfit
python handcrafted_feature_experiments.py run-control
python handcrafted_feature_experiments.py run-screen
python handcrafted_feature_experiments.py summarize-screen
python handcrafted_feature_experiments.py run-fusion       # 仅当 Phase A 决策触发融合
python handcrafted_feature_experiments.py run-confirm
python handcrafted_feature_experiments.py summarize-confirm
python handcrafted_feature_experiments.py audit-results
```

特征缓存和检查点保存在 `outputs/stage1/`，不会提交。小型配置、逐轮指标和汇总保存在 `report/stage1/`；脚本会由各次运行的 JSON 重新生成总表。阶段 I 的缓存表示保持 `[C,7,7]` 空间结构，可供后续 DNN、浅层 CNN 和按行输入的 RNN 使用。所需新依赖已经列入 `requirements.txt`。

Stage I 已按三种子内部验证选择 `REP-006-FUSION`（`[190,7,7]`，展平 9310 维）。DNN 架构实验固定这一表示、已有特征缓存、划分及训练集归一化，只比较纯全连接网络结构。Stage I 的固定 MLP 结果直接用作 `DNN-ARCH-BASE`，不会重新训练。运行以下命令将新增 A、B、C 各三个种子的实验，结果写入 `report/dnn_architecture/`；本阶段不读取 `data/val`。

```bash
python dnn_architecture_experiments.py smoke
python dnn_architecture_experiments.py run-all
python dnn_architecture_experiments.py summarize
python dnn_architecture_experiments.py audit
```

运行顺序为先提交固定代码版本，再执行 `run-all`，以便每次运行记录对应的 Git 提交。候选结构、选择规则与最终 DNN 的后续评估协议见 `report/dnn_architecture/architecture_summary.md`（运行 `summarize` 后生成）。

`DNN-ARCH-001` 已完成九次新训练；四模型比较选定 **DNN-ARCH-C**（三种子内部验证 78.17% ± 1.26%，最差种子 77.00%）。完整记录见 [架构汇总](report/dnn_architecture/architecture_summary.md)。其最初的 6 轮建议在最终评估前经固定轮次比较修订为 3 轮，详见下一节。

## 最终 DNN：DNN-FINAL-001

测试前的[冻结协议](report/dnn_final/frozen_protocol.md)将训练轮数由架构报告原建议的 6 轮修订为匹配固定轮次比较支持的 **3 轮**。最终训练复用 `REP-006-FUSION`，以全部 2000 张训练图拟合归一化并训练 DNN-ARCH-C；随后对 `data/val` 的 500 张图评估一次。结果为总体 **73.80%**、猫 **65.60%**、狗 **82.00%**、Macro F1 **73.62%**。完整指标、混淆矩阵与历史对照见[最终汇总](report/dnn_final/final_summary.md)。

以下命令用于复现全流程；现有 `DNN-FINAL-001` 产物已经生成，脚本会拒绝覆盖最终训练或再次评估。原始数据和 Stage I 特征缓存需在本机可用。

```bash
python dnn_final_experiment.py smoke
python dnn_final_experiment.py train
python dnn_final_experiment.py evaluate
python dnn_final_experiment.py archive
```

## CNN 架构比较：CNN-ARCH-001

仅使用 `data/train` 的三份既有 1800/200 内部划分，冻结 `REP-006-FUSION`、缓存及逐划分归一化；只比较浅层普通 CNN、浅层残差 CNN 和三尺度 CNN。`data/val` 不参与本阶段。

```bash
python cnn_architecture_experiments.py smoke
python cnn_architecture_experiments.py tiny-overfit
python cnn_architecture_experiments.py run-all
python cnn_architecture_experiments.py summarize
python cnn_architecture_experiments.py audit
```

正式运行的检查点保存在忽略的 `outputs/cnn_architecture/CNN-ARCH-001/`；配置、逐轮数据和预测归档到 `report/cnn_architecture/`。

九次架构训练已完成，按三种子内部验证选定 **CNN-ARCH-C**：79.83% ± 2.84%，最差种子 77.50%。详见[架构汇总与曲线](report/cnn_architecture/architecture_summary.md)。本阶段没有运行 CNN 最终测试。

## CNN 训练策略比较：CNN-TRAIN-001

固定 CNN-ARCH-C 与 `REP-006-FUSION`，直接引用既有 CNN-ARCH-C 运行作为 T0；新增 T1 原图水平镜像后重新提取手工特征、T2 融合层后的 Dropout2d(0.10)、T3 余弦学习率三种单独干预。仅用 `data/train` 的既有内部划分，`data/val` 不参与。

```bash
python cnn_training_strategy_experiments.py smoke
python cnn_training_strategy_experiments.py build-flip-cache
python cnn_training_strategy_experiments.py validate-flip-cache
python cnn_training_strategy_experiments.py run-all
python cnn_training_strategy_experiments.py summarize
python cnn_training_strategy_experiments.py audit
```

镜像特征缓存保存在忽略的 `outputs/stage1/cache/handcrafted128_flipped/`；正式运行检查点保存在 `outputs/cnn_training/CNN-TRAIN-001/`；小型记录归档在 `report/cnn_training/`。命令中的 `run-all` 用于首次执行九次训练；复核已有结果时只运行 `summarize` 和 `audit`。

九次新训练和三次历史 T0 记录已通过审计。T0/T1/T2/T3 的三种子内部验证均值分别为 79.83%/80.50%/80.00%/80.00%；按预设规则选定 **T1 原图水平镜像**。没有单策略达到相对 T0 至少 +1.0 个百分点，因此不运行组合实验。详见[策略汇总](report/cnn_training/training_strategy_summary.md)。本阶段没有执行 CNN 的 500 张保留图测试。

## 最终 CNN：CNN-FINAL-001

按[评估前冻结协议](report/cnn_final/frozen_protocol.md)，复用原始与镜像 REP-006 缓存，全部 4000 张训练图拟合归一化，固定种子 42 和 4 轮训练。重载最终检查点后对 `data/val` 原图评估一次：损失 0.4389，总体、猫和狗准确率均为 **81.20%**，Macro F1 为 81.20%。详见[最终汇总](report/cnn_final/final_summary.md)。这 500 张图曾用于历史 DNN 评估，但没有参与 CNN 架构或训练策略选择。

在 WSL Ubuntu 的项目根目录依次运行以下命令。`train` 会拒绝覆盖已有最终运行；`evaluate` 会拒绝第二次保留集评估。已有结果可用 `audit` 复核。

```bash
python cnn_final_experiment.py smoke
python cnn_final_experiment.py train
python cnn_final_experiment.py evaluate
python cnn_final_experiment.py archive
python cnn_final_experiment.py audit
```

大检查点保存在忽略的 `outputs/cnn_final/CNN-FINAL-001/`，小型配置、逐轮指标和 500 行预测保存在 `report/cnn_final/`。
