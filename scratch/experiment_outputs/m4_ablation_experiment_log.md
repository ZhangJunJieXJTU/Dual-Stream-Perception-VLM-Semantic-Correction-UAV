# M4 本地补充实验记录（2026-04-30）

## 1. 实验目标

基于 `YOLOFuse` 在本地 Apple M4（CPU）完成可复现实验，补充论文中“融合策略对比”部分的定量结果、脚本与流程。

## 2. 运行环境

- 设备：Apple M4
- 系统：macOS
- Python：3.12（项目 `.venv`）
- 框架：`ultralytics==8.3.74`（仓库本地可编辑安装）

## 3. 数据说明

由于完整 LLVIP 官方压缩包下载链路不稳定（Google Drive 连接中断），本轮先使用仓库内已有样例数据进行补充实验：

- 来源目录：`datasets/LLVIP_raw/FusionGAN`
- 可见光训练：`Train_LLVIP_vi`（32 对）
- 红外训练：`Train_LLVIP_ir`（32 对）
- 可见光验证：`Test_LLVIP_vi`（12 对）
- 红外验证：`Test_LLVIP_ir`（12 对）

构建后的实验数据目录：

- `datasets/llvip_sample_pseudo/images/{train,val}`
- `datasets/llvip_sample_pseudo/imagesIR/{train,val}`
- `datasets/llvip_sample_pseudo/labels/{train,val}`

> 标注采用离线启发式伪标注（阈值+连通域框），用于本地流程验证与补充实验，不等同于官方人工标注精度。

## 4. 实验脚本

### 4.1 数据构建与伪标注

- 脚本：`scratch/prepare_llvip_sample_dataset.py`
- 功能：复制可见光/红外同名图像对并生成 YOLO 格式伪标签，同时输出 `scratch/llvip_sample_pseudo.yaml`

运行命令：

```bash
PYTHONPATH=. MPLCONFIGDIR=/private/tmp/mpl .venv/bin/python scratch/prepare_llvip_sample_dataset.py
```

### 4.2 融合策略对比训练

- 脚本：`scratch/run_fusion_ablation_m4.py`
- 对比模型：
  - `Easy-level-Feature-Fusion`
  - `Mid-level-Feature-Fusion`
  - `Decision-level-Fusion`
- 统一配置：`epochs=3, batch=4, imgsz=640, device=cpu, optimizer=AdamW`

运行命令：

```bash
PYTHONPATH=. MPLCONFIGDIR=/private/tmp/mpl YOLO_CONFIG_DIR=/private/tmp/ultra .venv/bin/python scratch/run_fusion_ablation_m4.py
```

### 4.3 指标汇总

- 脚本：`scratch/summarize_m4_ablation.py`
- 输出：`scratch/experiment_outputs/m4_ablation_summary.csv`

运行命令：

```bash
PYTHONPATH=. .venv/bin/python scratch/summarize_m4_ablation.py
```

## 5. 结果汇总

| 模型 | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| easy_fusion_n_3ep | 0.00952 | 0.10000 | 0.00633 | 0.00371 |
| mid_fusion_n_3ep | 0.00472 | 0.28333 | 0.01330 | 0.00409 |
| decision_fusion_n_3ep | 0.00333 | 0.20000 | 0.00202 | 0.00094 |

## 6. 原始产物路径

- `runs/m4_ablation/easy_fusion_n_3ep`
- `runs/m4_ablation/mid_fusion_n_3ep`
- `runs/m4_ablation/decision_fusion_n_3ep`
- `scratch/experiment_outputs/m4_ablation_summary.csv`

## 7. 可用于论文补充实验的表述建议

1. 本地硬件受限（Apple M4 CPU）下，采用轻量配置完成多融合策略快速对比，验证了实验流程可复现。  
2. 在本轮样例伪标注数据上，`Mid-level-Feature-Fusion` 的 `mAP50` 最优。  
3. 由于数据规模较小且标签为自动伪标注，绝对指标仅用于补充对比，不作为最终主结果。  
4. 后续在完整 LLVIP 人工标注集上复现实验，可保持相同脚本，仅替换数据路径与训练轮次。
