<p align="center"><a href="README_en.md">English</a></p>

<p align="center">
  <a href="https://github.com/ZhangJunJieXJTU/Dual-Stream-Perception-VLM-Semantic-Correction-UAV">
    <img src="https://img.shields.io/badge/project-UAV%20Multimodal%20Detection-blue" alt="Project">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/license-AGPL--3.0-green" alt="License">
  </a>
  <a href="https://github.com/ZhangJunJieXJTU/Dual-Stream-Perception-VLM-Semantic-Correction-UAV">
    <img src="https://img.shields.io/badge/VLM-semantic%20correction-purple" alt="VLM semantic correction">
  </a>
</p>

# 基于双流感知与 VLM 语义修正的无人机目标检测项目

本仓库用于支撑论文《基于双流感知与 VLM 语义修正的无人机泛化目标检测方法》的代码实验与后续工程实现。项目面向无人机红外-可见光双模态目标检测任务，重点关注复杂光照、遮挡、低对比度、小目标和长尾异常目标场景下的检测鲁棒性。

当前代码基线提供 RGB/IR 双流输入、融合检测、训练、验证和推理脚本，后续可在此基础上继续接入 ByteTrack 时序关联、注意力热力图筛选和 VLM 难例语义修正模块。

<p align="center">
  <img src="examples/Images/rgbir.png" alt="RGB-IR 双模态融合示意图" width="600"/>
</p>

## 项目定位

- **论文方向**：无人机多模态感知、红外-可见光融合、泛化目标检测、VLM 语义修正。
- **基础检测框架**：支持 YOLO 系列检测网络的训练、验证与推理。
- **双模态输入**：支持同名 RGB 图像与红外图像自动配对。
- **融合策略**：包含数据级融合、决策级融合、早期特征融合、中期特征融合、极简融合与 DEYOLO 等配置。
- **扩展目标**：为论文中的“感知流 + 注意流 + VLM 修正 + 时序平滑”整体方法提供代码基础。

## 目录结构

```text
.
├── train_dual.py              # 双模态训练入口
├── val_dual.py                # 双模态验证入口
├── infer_dual.py              # 双模态推理入口
├── ultralytics/               # 检测框架核心代码
├── ultralytics/cfg/models/fuse/
│   ├── Data-level-Fusion.yaml
│   ├── Decision-level-Fusion.yaml
│   ├── Early-level-Feature-Fusion.yaml
│   ├── Mid-level-Feature-Fusion.yaml
│   ├── Easy-level-Feature-Fusion.yaml
│   └── DEYOLO.yaml
└── examples/
```

## 数据集格式

项目通过文件名自动关联 RGB 与红外图像。RGB 图像和红外图像需要保持同名，并放在平级目录中。

```text
datasets/
├── images/
│   ├── train/
│   │   └── 000001.jpg
│   └── val/
│       └── 000002.jpg
├── imagesIR/
│   ├── train/
│   │   └── 000001.jpg
│   └── val/
│       └── 000002.jpg
└── labels/
    ├── train/
    │   └── 000001.txt
    └── val/
        └── 000002.txt
```

标注文件采用标准 YOLO 格式，通常只需要为 RGB 图像准备一套标注，红外模态会按同名文件自动匹配。

## 环境安装

```bash
git clone https://github.com/ZhangJunJieXJTU/Dual-Stream-Perception-VLM-Semantic-Correction-UAV.git
cd Dual-Stream-Perception-VLM-Semantic-Correction-UAV
pip install -e .
```

建议使用独立 Python/Conda 环境安装，避免与系统环境中的 PyTorch、CUDA 或 OpenCV 版本冲突。

## 训练

```bash
python train_dual.py
```

训练前需要根据实际数据集路径修改数据配置文件，例如：

```text
ultralytics/cfg/datasets/LLVIP.yaml
```

也可以基于 `ultralytics/cfg/models/fuse/` 下的配置文件选择不同融合策略。

## 验证

```bash
python val_dual.py
```

建议在论文实验中至少记录以下指标：

- Precision
- Recall
- mAP@0.5
- mAP@0.5:0.95
- 参数量
- FLOPs
- 单帧推理延迟

## 推理

```bash
python infer_dual.py
```

推理输入需要同时准备 RGB 图像和红外图像，并确保文件名一致。

## 与论文方法的对应关系

| 论文模块 | 当前代码基础 | 后续扩展方向 |
| --- | --- | --- |
| 感知流 | RGB/IR 双模态 YOLO 检测 | 接入无人机场景数据集训练与多尺度检测配置 |
| 双流融合 | 多种 RGB-IR 融合配置 | 按实验设计统一对比不同融合层级 |
| 时序关联 | 可接入 ByteTrack 配置 | 输出带 track id 的候选目标序列 |
| 注意流 | 待扩展 | 增加任务驱动热力图生成与 ROI 注意力积分 |
| VLM 语义修正 | 待扩展 | 对低置信度/低质量检测框进行语义打分与坐标修正 |
| 轨迹平滑 | 待扩展 | 增加滑动窗口中值滤波后处理 |

## 实验建议

论文实验建议至少覆盖以下部分：

1. RGB 单模态、IR 单模态、RGB-IR 融合检测对比。
2. 不同融合策略的消融实验。
3. 引入 ByteTrack 后的跟踪稳定性对比。
4. VLM 修正前后的难例检测效果对比。
5. 端到端推理耗时与 VLM 调用比例分析。

## 基础与许可

本项目代码以开源多模态检测实现和 Ultralytics YOLO 框架为基础进行论文项目整理与后续扩展。仓库遵循 `LICENSE` 中的 AGPL-3.0 许可证要求。使用、修改或发布本项目时，请遵守相应开源许可证条款。
