# Smoke Experiment Report

## Environment

- Python: `.venv/bin/python`
- PyTorch: 2.11.0
- OpenCV: 4.13.0.92
- Ultralytics/YOLOFuse package: 8.3.74 editable install
- Device: CPU only; CUDA unavailable, MPS unavailable
- Extra dependency added: `einops`

## Local Data Availability

The repository only contains one paired LLVIP sample:

- RGB: `ultralytics/assets/LLVIP/images/120270.jpg`
- IR: `ultralytics/assets/LLVIP/imagesIR/120270.jpg`

No complete LLVIP, DroneVehicle, VisDrone-VID, or UAVDT dataset was found locally. Therefore, the current run is a code-path smoke test only, not a paper-valid benchmark.

## Toy Dataset

Created a minimal paired dataset under:

`scratch/toy_llvip`

The same RGB/IR pair was used for train and validation, with two manually drafted `person` boxes. This is only to verify that:

1. RGB/IR paired loading works.
2. The dual-modal model can train.
3. Validation metrics are produced.
4. Prediction images and label files are produced.

## Training Command

```bash
MPLCONFIGDIR=.cache/matplotlib YOLO_CONFIG_DIR=.cache/ultralytics XDG_CACHE_HOME=.cache \
.venv/bin/python - <<'PY'
from ultralytics import YOLO
model = YOLO('ultralytics/cfg/models/fuse/Easy-level-Feature-Fusion.yaml')
model.train(
    data='scratch/toy_llvip.yaml',
    ch=6,
    imgsz=320,
    epochs=1,
    batch=1,
    workers=0,
    device='cpu',
    project='runs/smoke_train',
    name='easy_fuse_toy',
    exist_ok=True,
    amp=False,
    cache=False,
    plots=True,
)
PY
```

## Smoke Training Result

Output directory:

`runs/smoke_train/easy_fuse_toy`

Training completed successfully. The loader confirmed dual modality:

`Loaded dual-modality (RGB+IR) dataset`

| Metric | Value |
| --- | ---: |
| Train images | 1 |
| Val images | 1 |
| Val instances | 2 |
| Epochs | 1 |
| train/box_loss | 0.82017 |
| train/cls_loss | 2.58292 |
| train/dfl_loss | 0.47630 |
| Precision | 0 |
| Recall | 0 |
| mAP@0.5 | 0 |
| mAP@0.5:0.95 | 0 |
| Inference speed | 60.9 ms/image on CPU |

The zero mAP is expected for a one-image, one-epoch, from-scratch toy run and must not be used as a paper result.

## Prediction Smoke Test

Output directory:

`runs/smoke_predict/easy_fuse_toy`

Prediction completed successfully and produced:

- `120270_rgb.jpg`
- `120270_ir.jpg`
- `labels/120270_rgb.txt`
- `labels/120270_ir.txt`

The output contains many low-confidence boxes because the model was not meaningfully trained.

## Model Complexity

Generated at:

`scratch/experiment_outputs/model_complexity.csv`

| Model | Params (M) | GFLOPs @320 | GFLOPs @640 |
| --- | ---: | ---: | ---: |
| Easy-level-Feature-Fusion | 3.842 | 2.247 | 8.987 |
| Early-level-Feature-Fusion | 2.587 | 1.859 | 7.435 |
| Mid-level-Feature-Fusion | 1.233 | 0.887 | 3.547 |
| Decision-level-Fusion | 4.271 | 2.736 | 10.943 |
| DEYOLO | 6.154 | 4.376 | 17.503 |

`Data-level-Fusion.yaml` did not instantiate successfully in the current codebase and raised:

`TypeError: list indices must be integers or slices, not tuple`

## Requirements for Paper-Valid Experiments

To produce values that can be written into the paper, the next step is to obtain a real paired dataset and run controlled experiments:

1. Download LLVIP or prepare the target UAV RGB/IR dataset.
2. Convert VOC XML annotations to YOLO format if using LLVIP.
3. Train all selected fusion baselines with the same split and hyperparameters.
4. Run validation to collect Precision, Recall, mAP@0.5, and mAP@0.5:0.95.
5. Run inference speed tests on the target hardware.
6. Add ByteTrack and VLM correction experiments after the detector baseline is stable.
