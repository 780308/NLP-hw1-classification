# 猫狗图像分类：DNN、CNN、RNN 实验

本项目按 `synopsis.txt` 实现猫狗二分类。当前已完成 **DNN 主实验**；CNN、RNN 和单独的预训练扩展实验尚未运行。实验方案见 [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md)，逐步更新的实测报告见 [report/experiment_report.md](report/experiment_report.md)。

每次实验的编号、配置、结果与对应 Git 提交方式记录在 [report/experiment_log.md](report/experiment_log.md)。公开仓库不包含原始 `data/` 图片或 `outputs/` 检查点；运行前需把课程提供的数据放到下述目录。

## 数据与环境

数据保持原样放在 `data/train/` 和 `data/val/`。两个目录均为平铺的 `cat.N.jpg`、`dog.N.jpg` 文件；程序固定 `cat=0`、`dog=1`。`data/train` 有 2000 张，`data/val` 有 500 张，后者仅用于最终测试。

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
- CNN：计划中的自建 VGG 风格卷积网络，待实现。
- RNN：计划中的按行输入 `nn.RNN` 网络，待实现。

每个模型的运行产物放在 `outputs/<model>/`，包括 `best_model.pt`、`config.json`、`split.json`、`history.csv`、`train_summary.json`、`test_metrics.json` 和图表。`outputs/` 不纳入版本控制；报告引用的图另存于 `report/figures/`。检查点包含结构配置、类别映射、最佳轮次和内部验证指标，可由 `evaluate.py` 重载。

完整依赖列于 `requirements.txt`。当前 WSL 环境已验证 PyTorch `2.5.1+cu121`、TorchVision `0.20.1+cu121` 和 RTX 3050 Ti 的 CUDA 运算。

## 后续实验的 Git 工作流

新实验使用独立编号和输出目录，例如 `python train.py --model dnn --data-dir data --output-dir outputs/dnn/dnn-002`，避免覆盖 DNN-001。实验完成后，将配置、划分、逐轮记录、汇总和评估 JSON 复制到 `report/experiments/dnn-002/`，更新实验索引、总报告和计划，再执行 `git add`、带编号的 `git commit` 和 `git push origin main`。由 `git log --grep=dnn-002` 可定位当次代码和报告。正式实验未完成或失败也应记录状态；不使用已看过的测试结果选择后续模型。

## 第一阶段：手工空间表示

实验协议见 [STAGE1-GUIDELINES.md](STAGE1-GUIDELINES.md)。第一阶段只使用 `data/train`，共用固定的 90/10 内部划分；`data/val` 不参与表示选择。运行环境仍为 WSL 的 `~/.venvs/hw1`。

```bash
source ~/.venvs/hw1/bin/activate
python stage1.py smoke
python stage1.py build-cache
python stage1.py validate-cache
python stage1.py tiny-overfit
python stage1.py run-control
python stage1.py run-screen
python stage1.py summarize-screen
python stage1.py run-fusion       # 仅当 Phase A 决策触发融合
python stage1.py run-confirm
python stage1.py summarize-confirm
```

特征缓存和检查点保存在 `outputs/stage1/`，不会提交。小型配置、逐轮指标和汇总保存在 `report/stage1/`；脚本会由各次运行的 JSON 重新生成总表。阶段 I 的缓存表示保持 `[C,7,7]` 空间结构，可供后续 DNN、浅层 CNN 和按行输入的 RNN 使用。所需新依赖已经列入 `requirements.txt`。
