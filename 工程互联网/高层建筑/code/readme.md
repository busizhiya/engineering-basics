# 高层建筑振动响应预测（Conv1D + BiLSTM + 双头 MLP）

## 我做了什么
我已按你的背景说明，完成了一套可直接运行的端到端代码，并把训练与推理分开：

1. 新增数据预处理脚本 `preprocess.py`
- 自动读取 `raw_data/*.csv`
- 只保留所有传感器都完整的时序帧（去掉开头不完整段）
- 对每个通道做线性去趋势
- 做带通滤波（默认 0.1~20Hz，采样率按时间戳估计，异常时回退到 50Hz）
- 导出清洗后的 `csv` 文件到 `processed/cleaned/`
- 输出驱动通道（加速度06的 Z 轴）FFT 图到 `plots/fft_driver_z.png`
- 在 `processed/meta.json` 记录主频和文件信息

2. 新增模型定义 `src/models.py`
- 结构为 `Conv1D + BiLSTM + 双头 MLP`
- 适配两个任务：
  - 正向模型（驱动 -> 响应）：3 -> 17
  - 反向模型（响应 -> 驱动）：17 -> 3

3. 新增训练脚本 `train.py`
- 与推理完全分离
- 支持 `--task forward` 和 `--task inverse`
- 自动按文件划分 train/val/test
- 滑窗 many-to-many 训练
- 不做标准化，直接在物理量尺度训练
- 每次训练会生成 loss 曲线图 `outputs/<task>/loss_curve.png`
- 保存最优模型 `outputs/<task>/best.pt`
- 输出测试集 RMS 指标 `outputs/<task>/metrics.json`

4. 新增推理脚本 `infer.py`
- 载入 `best.pt` 对单个 cleaned csv 做整段推理
- 不做标准化与反标准化，直接输出物理量尺度预测
- 生成真实值 vs 预测值对比图 `prediction_vs_true.png`
- 生成 RMS 报告 `rms_report.csv`
- 导出逐时刻预测结果 `prediction_full.csv`

5. 新增工具模块
- `src/data_utils.py`：列映射、清洗、滤波、滑窗
- `src/train_utils.py`：数据集封装、划分、损失曲线、JSON 输出

6. 新增依赖文件
- `requirements.txt`

## 目录说明

```text
v2/
  raw_data/
  src/
    data_utils.py
    models.py
    train_utils.py
  preprocess.py
  train.py
  infer.py
  requirements.txt
  readme.md
```

## 环境准备

建议 Python 3.10+。

```powershell
pip install -r requirements.txt
```

## 运行步骤

### 1) 数据预处理

```powershell
python preprocess.py --raw_dir raw_data --out_dir processed/cleaned --meta_path processed/meta.json --fft_plot plots/fft_driver_z.png
```

运行后你会得到：
- 清洗后样本：`processed/cleaned/*.csv`
- 频域图：`plots/fft_driver_z.png`
- 元信息：`processed/meta.json`

### 2) 训练模型一（驱动 -> 响应）

说明：`--data_dir` 可以传两类目录。
- `processed/cleaned`：预处理后 csv。
- `raw_data`：原始采集 csv（脚本会自动完成清洗）。

```powershell
python train.py --task forward --data_dir processed/cleaned --out_dir outputs --seq_len 200 --stride 50 --epochs 80 --batch_size 64 --lr 1e-3
```

输出：
- 最优模型：`outputs/forward/best.pt`
- loss 图：`outputs/forward/loss_curve.png`
- 指标：`outputs/forward/metrics.json`

### 3) 训练模型二（响应 -> 驱动）

```powershell
python train.py --task inverse --data_dir processed/cleaned --out_dir outputs --seq_len 200 --stride 50 --epochs 80 --batch_size 64 --lr 1e-3
```

输出：
- 最优模型：`outputs/inverse/best.pt`
- loss 图：`outputs/inverse/loss_curve.png`
- 指标：`outputs/inverse/metrics.json`

### 4) 推理与可视化（以模型一为例）

说明：`--csv` 也支持两类文件。
- 预处理后的 cleaned csv（推荐）。
- 原始 raw_data 下 csv（脚本会自动清洗后推理）。

```powershell
python infer.py --checkpoint outputs/forward/best.pt --csv processed/cleaned/简谐波_5mm_3.4Hz_补测.csv --out_dir outputs/infer --stride 50
python infer.py --checkpoint outputs/inverse/best.pt --csv processed/cleaned/简谐波_5mm_3.4Hz_补测.csv --out_dir outputs/infer --stride 50

```

输出：
- 对比图：`outputs/infer/forward/<文件名>/prediction_vs_true.png`
- RMS：`outputs/infer/forward/<文件名>/rms_report.csv`
- 全时序预测：`outputs/infer/forward/<文件名>/prediction_full.csv`

模型二推理同理，把 checkpoint 换成 `outputs/inverse/best.pt`。

## 输入输出维度定义

- 模型一（forward）
  - 输入（3）：`WSMS00012.AccX/Y/Z`（加速度06）
  - 输出（17）：`WSMS00007~11.AccX/Y/Z` + `WSGD00003/00004.chdata`

- 模型二（inverse）
  - 输入（17）：`WSMS00007~11.AccX/Y/Z` + `WSGD00003/00004.chdata`
  - 输出（3）：`WSMS00012.AccX/Y/Z`

## 指标说明

- 训练损失：MSE（物理量尺度）
- 汇报指标：RMSE（物理量尺度）
- `metrics.json` 给出测试集整体 RMSE 与各通道 RMSE
- `rms_report.csv` 给出单次推理样本的整体与分通道 RMSE

## 注意事项

1. 本代码已实现“训练和推理分开”，符合你的要求。
2. 我没有替你执行训练命令（按你的要求）。
3. 如显存不足，可降低 `--batch_size`，如窗口过长可降低 `--seq_len`。
4. 如果某些文件有效长度太短，可能无法切出窗口，请调整 `--seq_len` 或 `--stride`。
