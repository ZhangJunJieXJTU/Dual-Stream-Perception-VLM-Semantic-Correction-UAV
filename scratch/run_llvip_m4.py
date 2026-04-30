from pathlib import Path

from ultralytics import YOLO


def main():
    root = Path(__file__).resolve().parents[1]
    data_yaml = root / "scratch" / "llvip_m4.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"Missing dataset yaml: {data_yaml}")

    model = YOLO(str(root / "ultralytics/cfg/models/fuse/Easy-level-Feature-Fusion.yaml"))
    model.train(
        data=str(data_yaml),
        ch=6,
        imgsz=640,
        epochs=5,
        batch=4,
        workers=2,
        device="cpu",
        optimizer="AdamW",
        amp=False,
        cache=False,
        project="runs/m4_train",
        name="easy_fuse_n_llvip",
        pretrained=False,
        patience=0,
        close_mosaic=0,
        fraction=0.1,
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
