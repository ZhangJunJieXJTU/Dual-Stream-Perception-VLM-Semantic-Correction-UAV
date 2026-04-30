from pathlib import Path
import csv


def read_last_row(csv_path: Path) -> dict:
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[-1]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    run_root = root / "runs" / "m4_ablation"
    out_csv = root / "scratch" / "experiment_outputs" / "m4_ablation_summary.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    headers = [
        "exp_name",
        "precision",
        "recall",
        "mAP50",
        "mAP50-95",
        "train_box_loss",
        "train_cls_loss",
        "train_dfl_loss",
    ]
    rows = []
    for exp_dir in sorted(run_root.glob("*")):
        csv_path = exp_dir / "results.csv"
        if not csv_path.exists():
            continue
        row = read_last_row(csv_path)
        rows.append(
            {
                "exp_name": exp_dir.name,
                "precision": row.get("metrics/precision(B)", ""),
                "recall": row.get("metrics/recall(B)", ""),
                "mAP50": row.get("metrics/mAP50(B)", ""),
                "mAP50-95": row.get("metrics/mAP50-95(B)", ""),
                "train_box_loss": row.get("train/box_loss", ""),
                "train_cls_loss": row.get("train/cls_loss", ""),
                "train_dfl_loss": row.get("train/dfl_loss", ""),
            }
        )

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote summary: {out_csv}")


if __name__ == "__main__":
    main()
