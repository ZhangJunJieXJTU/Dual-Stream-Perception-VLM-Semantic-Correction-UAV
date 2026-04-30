from pathlib import Path
import shutil

import cv2


def ensure_empty_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_pairs(src_vi: Path, src_ir: Path, dst_img: Path, dst_ir: Path) -> list[Path]:
    files = sorted(src_vi.glob("*.jpg"))
    copied = []
    for vi in files:
        ir = src_ir / vi.name
        if not ir.exists():
            continue
        shutil.copy2(vi, dst_img / vi.name)
        shutil.copy2(ir, dst_ir / vi.name)
        copied.append(dst_img / vi.name)
    return copied


def write_label_file(path: Path, boxes) -> None:
    lines = []
    for x, y, w, h in boxes:
        lines.append(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
    path.write_text("\n".join(lines), encoding="utf-8")


def heuristic_person_boxes(img_path: Path) -> list[tuple[float, float, float, float]]:
    im = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if im is None:
        return []
    blur = cv2.GaussianBlur(im, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h_img, w_img = im.shape[:2]
    out = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if area < 120:
            continue
        if h < w:
            continue
        x_c = (x + w / 2) / w_img
        y_c = (y + h / 2) / h_img
        w_n = w / w_img
        h_n = h / h_img
        out.append((x_c, y_c, w_n, h_n))
    out.sort(key=lambda z: z[2] * z[3], reverse=True)
    return out[:5]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    raw = root / "datasets" / "LLVIP_raw" / "FusionGAN"
    out = root / "datasets" / "llvip_sample_pseudo"

    train_img = out / "images" / "train"
    val_img = out / "images" / "val"
    train_ir = out / "imagesIR" / "train"
    val_ir = out / "imagesIR" / "val"
    train_lb = out / "labels" / "train"
    val_lb = out / "labels" / "val"

    ensure_empty_dir(out)
    for d in [train_img, val_img, train_ir, val_ir, train_lb, val_lb]:
        d.mkdir(parents=True, exist_ok=True)

    train_pairs = copy_pairs(raw / "Train_LLVIP_vi", raw / "Train_LLVIP_ir", train_img, train_ir)
    val_pairs = copy_pairs(raw / "Test_LLVIP_vi", raw / "Test_LLVIP_ir", val_img, val_ir)

    all_imgs = [(p, train_lb) for p in train_pairs] + [(p, val_lb) for p in val_pairs]
    for img_path, label_dir in all_imgs:
        boxes = heuristic_person_boxes(img_path)
        write_label_file(label_dir / f"{img_path.stem}.txt", boxes)

    yaml_path = root / "scratch" / "llvip_sample_pseudo.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {out}",
                "train: images/train",
                "val: images/val",
                "test:",
                "names:",
                "  0: person",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Prepared dataset: {out}")
    print(f"Train pairs: {len(train_pairs)}, Val pairs: {len(val_pairs)}")
    print(f"YAML: {yaml_path}")


if __name__ == "__main__":
    main()
