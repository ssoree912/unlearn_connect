import argparse
import csv
import math
import os
import re
import shlex
from pathlib import Path


TRIAL_RE = re.compile(r"keep_([^/]+)/lr_([^/]+)/epochs_([^/]+)")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Select the best SalUn tuning run for one ratio"
    )
    parser.add_argument(
        "--tune_root",
        required=True,
        type=str,
        help="root directory containing keep_*/lr_*/endpoint_metrics.csv",
    )
    parser.add_argument(
        "--retrain_metrics_path",
        default=None,
        type=str,
        help="endpoint_metrics.csv from the retrain oracle for this ratio",
    )
    parser.add_argument(
        "--output_env",
        required=True,
        type=str,
        help="where to write BEST_KEEP_RATIO/BEST_LR/BEST_SCORE",
    )
    parser.add_argument(
        "--output_csv",
        required=True,
        type=str,
        help="where to write the ranked tuning table",
    )
    parser.add_argument(
        "--score_cols",
        default="ua,acc_retain,acc_test,mia",
        type=str,
        help="comma-separated metrics to compare against the selected reference",
    )
    parser.add_argument(
        "--reference_mode",
        default="paper_target",
        choices=["paper_target", "retrain_oracle"],
        help="which reference to use when scoring candidates",
    )
    parser.add_argument(
        "--reference_name",
        default=None,
        type=str,
        help="name written into the leaderboard for the chosen reference",
    )
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
    parser.add_argument(
        "--min_acc_retain",
        default=None,
        type=float,
        help="optional hard lower bound on retain accuracy",
    )
    parser.add_argument(
        "--min_acc_test",
        default=None,
        type=float,
        help="optional hard lower bound on test accuracy",
    )
    return parser.parse_args()


def parse_score_cols(spec):
    return [token.strip() for token in str(spec).split(",") if token.strip()]


def read_csv_rows(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def is_truthy(value):
    if value is None:
        return True
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def to_float(value):
    return float(value)


def is_finite_number(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def load_final_row(path):
    csv_path = Path(path)
    if not csv_path.exists():
        return None
    rows = read_csv_rows(csv_path)
    if not rows:
        return None

    if "epoch" not in rows[0]:
        return rows[-1]

    best_row = None
    best_epoch = None
    for row in rows:
        epoch_value = row.get("epoch")
        if not is_finite_number(epoch_value):
            continue
        epoch = int(float(epoch_value))
        if best_epoch is None or epoch >= best_epoch:
            best_epoch = epoch
            best_row = row
    return best_row


def build_reference_row(args, score_cols):
    if args.reference_mode == "retrain_oracle":
        if not args.retrain_metrics_path:
            raise ValueError("--retrain_metrics_path is required for retrain_oracle mode")
        retrain_row = load_final_row(args.retrain_metrics_path)
        if retrain_row is None:
            raise FileNotFoundError(
                f"Missing or empty retrain metrics: {args.retrain_metrics_path}"
            )
        return (
            args.reference_name or "retrain_oracle",
            retrain_row,
        )

    reference_row = {}
    explicit_targets = {
        "ua": args.target_ua,
        "acc_retain": args.target_acc_retain,
        "acc_test": args.target_acc_test,
        "mia": args.target_mia,
        "acc_forget": args.target_acc_forget,
        "acc_val": args.target_acc_val,
        "loss_retain": args.target_loss_retain,
        "loss_forget": args.target_loss_forget,
        "loss_val": args.target_loss_val,
        "loss_test": args.target_loss_test,
    }
    for col in score_cols:
        if col not in explicit_targets or explicit_targets[col] is None:
            raise ValueError(f"Missing explicit target for score column '{col}'")
        reference_row[col] = explicit_targets[col]
    return args.reference_name or "paper_target", reference_row


def parse_trial_info(csv_path):
    match = TRIAL_RE.search(str(csv_path.parent).replace(os.sep, "/"))
    if match is None:
        return None
    return match.group(1), match.group(2), match.group(3)


def is_valid_candidate(row, score_cols):
    if row is None:
        return False
    if "valid_run" in row and not is_truthy(row["valid_run"]):
        return False
    for col in score_cols:
        if col not in row or not is_finite_number(row[col]):
            return False
    return True


def meets_constraints(row, args):
    if args.min_acc_retain is not None:
        if "acc_retain" not in row or not is_finite_number(row["acc_retain"]):
            return False
        if to_float(row["acc_retain"]) < args.min_acc_retain:
            return False
    if args.min_acc_test is not None:
        if "acc_test" not in row or not is_finite_number(row["acc_test"]):
            return False
        if to_float(row["acc_test"]) < args.min_acc_test:
            return False
    return True


def build_record(row, keep_ratio, lr, trial_unlearn_epochs, reference_row, args, reference_name, score_cols, csv_path):
    record = {
        "keep_ratio": keep_ratio,
        "lr": lr,
        "trial_csv": str(csv_path),
        "valid_run": row.get("valid_run", True),
        "selected_epoch": int(float(row["epoch"])) if is_finite_number(row.get("epoch")) else None,
        "trial_unlearn_epochs": int(float(trial_unlearn_epochs)) if is_finite_number(trial_unlearn_epochs) else trial_unlearn_epochs,
        "reference_mode": args.reference_mode,
        "reference_name": reference_name,
    }

    score_terms = []
    for col in score_cols:
        reference_value = to_float(reference_row[col])
        trial_value = to_float(row[col])
        gap = trial_value - reference_value
        record[col] = trial_value
        record[f"reference_{col}"] = reference_value
        record[f"gap_{col}"] = gap
        record[f"abs_gap_{col}"] = abs(gap)
        score_terms.append(abs(gap))

    if (
        "acc_test" in row
        and is_finite_number(row["acc_test"])
        and is_finite_number(reference_row.get("acc_test"))
        and to_float(row["acc_test"]) < to_float(reference_row["acc_test"]) - 2.0
    ):
        record["hard_penalty"] = 10.0
        score_terms.append(10.0)
    else:
        record["hard_penalty"] = 0.0

    record["score"] = sum(score_terms) / len(score_terms)
    return record


def collect_fieldnames(rows, preferred_prefix):
    fieldnames = list(preferred_prefix)
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    return fieldnames


def write_csv(path, rows, preferred_prefix):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = collect_fieldnames(rows, preferred_prefix)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    score_cols = parse_score_cols(args.score_cols)
    reference_name, reference_row = build_reference_row(args, score_cols)

    candidates = []
    tune_root = Path(args.tune_root)
    for csv_path in tune_root.rglob("endpoint_metrics.csv"):
        trial = parse_trial_info(csv_path)
        if trial is None:
            continue

        keep_ratio, lr, trial_unlearn_epochs = trial
        rows = read_csv_rows(csv_path)
        if not rows:
            continue

        best_record = None
        for row in rows:
            if not is_valid_candidate(row, score_cols):
                continue
            if not meets_constraints(row, args):
                continue

            record = build_record(
                row,
                keep_ratio,
                lr,
                trial_unlearn_epochs,
                reference_row,
                args,
                reference_name,
                score_cols,
                csv_path,
            )
            if best_record is None:
                best_record = record
                continue

            current_key = (
                record["score"],
                record.get("abs_gap_ua", math.inf),
                record.get("abs_gap_acc_test", math.inf),
                record["selected_epoch"] if record["selected_epoch"] is not None else math.inf,
            )
            best_key = (
                best_record["score"],
                best_record.get("abs_gap_ua", math.inf),
                best_record.get("abs_gap_acc_test", math.inf),
                best_record["selected_epoch"] if best_record["selected_epoch"] is not None else math.inf,
            )
            if current_key < best_key:
                best_record = record

        if best_record is not None:
            candidates.append(best_record)

    if not candidates:
        raise RuntimeError(f"No valid SalUn tuning runs found under {args.tune_root}")

    def sort_key(record):
        return (
            record["score"],
            record.get("abs_gap_ua", math.inf),
            record.get("abs_gap_acc_test", math.inf),
            float(record["keep_ratio"]),
            float(record["lr"]),
        )

    ranked = sorted(candidates, key=sort_key)
    write_csv(
        args.output_csv,
        ranked,
        preferred_prefix=[
            "keep_ratio",
            "lr",
            "trial_csv",
            "selected_epoch",
            "trial_unlearn_epochs",
            "valid_run",
            "score",
            "hard_penalty",
        ],
    )

    best = ranked[0]
    os.makedirs(os.path.dirname(args.output_env) or ".", exist_ok=True)
    with open(args.output_env, "w", encoding="utf-8") as handle:
        handle.write(f"BEST_KEEP_RATIO={shlex.quote(str(best['keep_ratio']))}\n")
        handle.write(f"BEST_LR={shlex.quote(str(best['lr']))}\n")
        handle.write(f"BEST_EPOCH={shlex.quote(str(best['selected_epoch']))}\n")
        handle.write(f"BEST_TRIAL_UNLEARN_EPOCHS={shlex.quote(str(best['trial_unlearn_epochs']))}\n")
        handle.write(f"BEST_SCORE={shlex.quote(str(best['score']))}\n")
        handle.write(f"BEST_TRIAL_CSV={shlex.quote(str(best['trial_csv']))}\n")
        handle.write(f"BEST_REFERENCE_MODE={shlex.quote(str(best['reference_mode']))}\n")
        handle.write(f"BEST_REFERENCE_NAME={shlex.quote(str(best['reference_name']))}\n")

    for row in ranked[:10]:
        print(row)
    print(
        f"[selected] keep={best['keep_ratio']} lr={best['lr']} epoch={best['selected_epoch']} score={best['score']:.6f}"
    )


if __name__ == "__main__":
    main()
