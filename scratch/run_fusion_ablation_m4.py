from pathlib import Path

from ultralytics import YOLO


def run_once(cfg: str, exp_name: str) -> None:
    root = Path(__file__).resolve().parents[1]
    model = YOLO(str(root / cfg))
    model.train(
        data=str(root / "scratch/llvip_sample_pseudo.yaml"),
        ch=6,
        imgsz=640,
        epochs=3,
        batch=4,
        workers=2,
        device="cpu",
        optimizer="AdamW",
        amp=False,
        cache=False,
        project="runs/m4_ablation",
        name=exp_name,
        pretrained=False,
        patience=0,
        close_mosaic=0,
        fraction=1.0,
        exist_ok=True,
    )


def main() -> None:
    runs = [
        ("ultralytics/cfg/models/fuse/Easy-level-Feature-Fusion.yaml", "easy_fusion_n_3ep"),
        ("ultralytics/cfg/models/fuse/Mid-level-Feature-Fusion.yaml", "mid_fusion_n_3ep"),
        ("ultralytics/cfg/models/fuse/Decision-level-Fusion.yaml", "decision_fusion_n_3ep"),
    ]
    for cfg, name in runs:
        run_once(cfg, name)


if __name__ == "__main__":
    main()
