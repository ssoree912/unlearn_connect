import argparse
import csv
import os
from pathlib import Path

import select_best_salun as selector


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect all SalUn tuning trial rows into one CSV"
    )
    parser.add_argument("--tune_root", required=True, type=str)
    parser.add_argument("--output_csv", required=True, type=str)
    parser.add_argument(
        "--score_cols",
        default="ua,acc_retain,acc_test,mia",
        type=str,
    )
    parser.add_argument(
        "--score_weights",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--reference_mode",
        default="retrain_oracle",
        choices=["paper_target", "retrain_oracle"],
    )
    parser.add_argument("--reference_name", default=None, type=str)
    parser.add_argument("--retrain_metrics_path", default=None, type=str)
    parser.add_argument("--target_ua", default=None, type=float)
    parser.add_argument("--target_acc_retain", default=None, type=float)
    parser.add_argument("--target_acc_test", default=None, type=float)
    parser.add_argument("--target_mia", default=None, type=float)
    parser.add_argument("--target_acc_forget", default=None, type=float)
    parser.add_argument("--target_acc_val", default=None, type=float)
    parser.add_argument("--target_loss_retain", default=None, type=float)
    parser.add_argument("--target_loss_forget", default=None, type=float)
    parser.add_argument("--target_loss_val", default=None, type=float)
    parser.add_argument("--target_loss_test", default=None, type=float)
    parser.add_argument("--min_acc_retain", default=None, type=float)
    parser.add_argument("--min_acc_test", default=None, type=float)
    return parser.parse_args()


def collect_fieldnames(rows):
    preferred = [
        "keep_ratio",
        "lr",
        "trial_csv",
        "epoch",
        "trial_unlearn_epochs",
        "valid_run",
        "passes_constraints",
        "valid_for_score",
        "score",
        "reference_mode",
        "reference_name",
    ]
    fieldnames = list(preferred)
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    return fieldnames


def main():
    args = parse_args()
    score_cols = selector.parse_score_cols(args.score_cols)
    score_weights = selector.parse_score_weights(args.score_weights, score_cols)
    reference_name, reference_row = selector.build_reference_row(args, score_cols)

    collected = []
    tune_root = Path(args.tune_root)
    for csv_path in tune_root.rglob("endpoint_metrics.csv"):
        rows = selector.read_csv_rows(csv_path)
        if not rows:
            continue

        trial = selector.parse_trial_info(csv_path)
        if trial is None:
            continue

        keep_ratio, lr, trial_unlearn_epochs = trial

        for row in rows:
            record = {
                "keep_ratio": keep_ratio,
                "lr": lr,
                "trial_csv": str(csv_path),
                "epoch": row.get("epoch"),
                "trial_unlearn_epochs": trial_unlearn_epochs,
                "valid_run": row.get("valid_run", True),
                "reference_mode": args.reference_mode,
                "reference_name": reference_name,
            }

            for key, value in row.items():
                if key not in record:
                    record[key] = value

            passes_constraints = selector.meets_constraints(row, args)
            valid_for_score = selector.is_valid_candidate(row, score_cols)
            record["passes_constraints"] = passes_constraints
            record["valid_for_score"] = valid_for_score

            if valid_for_score:
                scored = selector.build_record(
                    row,
                    keep_ratio,
                    lr,
                    trial_unlearn_epochs,
                    reference_row,
                    args,
                    reference_name,
                    score_cols,
                    score_weights,
                    csv_path,
                )
                for key, value in scored.items():
                    if key not in {"keep_ratio", "lr", "trial_csv", "trial_unlearn_epochs"}:
                        record[key] = value

            collected.append(record)

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    with open(args.output_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=collect_fieldnames(collected))
        writer.writeheader()
        writer.writerows(collected)


if __name__ == "__main__":
    main()
