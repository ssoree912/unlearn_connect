import argparse
import os
import re
from pathlib import Path

import pandas as pd


TRIAL_RE = re.compile(r"keep_([^/]+)/lr_([^/]+)")


def parse_args():
    p = argparse.ArgumentParser(description="Select the best SalUn tuning run for one ratio")
    p.add_argument("--tune_root", required=True, type=str, help="root directory containing keep_*/lr_*/endpoint_metrics.csv")
    p.add_argument("--retrain_metrics_path", required=True, type=str, help="endpoint_metrics.csv from retrain oracle for this ratio")
    p.add_argument("--output_env", required=True, type=str, help="where to write BEST_KEEP_RATIO/BEST_LR/BEST_SCORE")
    p.add_argument("--output_csv", required=True, type=str, help="where to write the ranked tuning table")
    p.add_argument("--score_cols", default="ua,mia,acc_retain,acc_test", type=str,
                   help="comma-separated metrics to compare against retrain (default: ua,mia,acc_retain,acc_test)")
    return p.parse_args()


def load_final_row(csv_path: Path) -> pd.Series | None:
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    if df.empty:
        return None
    if "epoch" in df.columns:
        max_epoch = df["epoch"].max()
        df = df[df["epoch"] == max_epoch]
    return df.iloc[-1]


def parse_trial_info(csv_path: Path):
    m = TRIAL_RE.search(str(csv_path.parent))
    if not m:
        return None
    keep_ratio, lr = m.group(1), m.group(2)
    return keep_ratio, lr


def is_valid_numeric_row(row: pd.Series, cols: list[str]) -> bool:
    for col in cols:
        if col not in row.index:
            return False
        value = row[col]
        if pd.isna(value):
            return False
    return True


def main():
    args = parse_args()
    score_cols = [c.strip() for c in args.score_cols.split(",") if c.strip()]

    retrain_row = load_final_row(Path(args.retrain_metrics_path))
    if retrain_row is None:
        raise FileNotFoundError(f"Missing or empty retrain metrics: {args.retrain_metrics_path}")

    tune_root = Path(args.tune_root)
    candidates = []
    for csv_path in tune_root.rglob("endpoint_metrics.csv"):
        trial = parse_trial_info(csv_path)
        if trial is None:
            continue
        keep_ratio, lr = trial
        row = load_final_row(csv_path)
        if row is None or not is_valid_numeric_row(row, score_cols):
            continue

        record = {
            "keep_ratio": keep_ratio,
            "lr": lr,
            "trial_csv": str(csv_path),
            "epoch": int(row["epoch"]) if "epoch" in row.index and pd.notna(row["epoch"]) else None,
        }

        score_terms = []
        for col in score_cols:
            gap = float(row[col]) - float(retrain_row[col])
            record[col] = float(row[col])
            record[f"retrain_{col}"] = float(retrain_row[col])
            record[f"gap_{col}"] = gap
            record[f"abs_gap_{col}"] = abs(gap)
            score_terms.append(abs(gap))

        # Utility safeguard: if test accuracy collapses, push the trial down even if UA/MIA look good.
        if "acc_test" in row.index and float(row["acc_test"]) < float(retrain_row["acc_test"]) - 2.0:
            score_terms.append(10.0)
            record["hard_penalty"] = 10.0
        else:
            record["hard_penalty"] = 0.0

        record["score"] = sum(score_terms) / len(score_terms)
        candidates.append(record)

    if not candidates:
        raise RuntimeError(f"No valid SalUn tuning runs found under {args.tune_root}")

    ranked = pd.DataFrame(candidates).sort_values(["score", "abs_gap_ua", "abs_gap_acc_test"], ascending=[True, True, True])
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    ranked.to_csv(args.output_csv, index=False)

    best = ranked.iloc[0]
    os.makedirs(os.path.dirname(args.output_env) or ".", exist_ok=True)
    with open(args.output_env, "w", encoding="utf-8") as f:
        f.write(f"BEST_KEEP_RATIO={best['keep_ratio']}\n")
        f.write(f"BEST_LR={best['lr']}\n")
        f.write(f"BEST_SCORE={best['score']}\n")
        f.write(f"BEST_TRIAL_CSV={best['trial_csv']}\n")

    print(ranked.head(10).to_string(index=False))
    print(f"[selected] keep={best['keep_ratio']} lr={best['lr']} score={best['score']:.6f}")


if __name__ == "__main__":
    main()
