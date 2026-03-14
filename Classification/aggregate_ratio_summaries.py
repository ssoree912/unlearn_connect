import argparse
import csv
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate per-ratio CSV artifacts into combined summary CSVs"
    )
    parser.add_argument(
        "--runs_dir",
        default="runs",
        type=str,
        help="root directory containing per-ratio run folders",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        help="directory where aggregate CSVs will be written (defaults to <runs_dir>/summary)",
    )
    parser.add_argument(
        "--ratios",
        default=None,
        type=str,
        help="optional comma-separated ratio list to aggregate; defaults to all numeric subdirectories",
    )
    return parser.parse_args()


def parse_ratio_spec(spec):
    if spec is None:
        return None
    ratios = []
    for item in spec.split(","):
        value = item.strip()
        if not value:
            continue
        ratios.append(int(value))
    return ratios


def discover_ratios(runs_dir):
    ratios = []
    if not os.path.isdir(runs_dir):
        return ratios
    for name in os.listdir(runs_dir):
        path = os.path.join(runs_dir, name)
        if os.path.isdir(path) and name.isdigit():
            ratios.append(int(name))
    return sorted(ratios)


def read_csv_rows(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def collect_endpoint_rows(runs_dir, ratios):
    rows = []
    methods = ["retrain", "salun_A", "salun_B", "ft", "ga"]
    for ratio in ratios:
        ratio_dir = os.path.join(runs_dir, str(ratio))
        for method in methods:
            csv_path = os.path.join(ratio_dir, method, "endpoint_metrics.csv")
            if not os.path.exists(csv_path):
                continue
            for row in read_csv_rows(csv_path):
                rows.append(
                    {
                        "ratio": ratio,
                        "method": method,
                        "source_csv": os.path.abspath(csv_path),
                        **row,
                    }
                )
    return rows


def collect_interpolation_rows(runs_dir, ratios, filename):
    rows = []
    for ratio in ratios:
        csv_path = os.path.join(runs_dir, str(ratio), "interpolation", filename)
        if not os.path.exists(csv_path):
            continue
        for row in read_csv_rows(csv_path):
            rows.append(
                {
                    "ratio": ratio,
                    "source_csv": os.path.abspath(csv_path),
                    **row,
                }
            )
    return rows


def collect_fieldnames(rows, preferred_prefix):
    fieldnames = list(preferred_prefix)
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    return fieldnames


def write_csv(path, rows, preferred_prefix):
    if not rows:
        return False
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = collect_fieldnames(rows, preferred_prefix)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return True


def main():
    args = parse_args()
    output_dir = args.output_dir or os.path.join(args.runs_dir, "summary")
    ratios = parse_ratio_spec(args.ratios) or discover_ratios(args.runs_dir)

    if not ratios:
        raise ValueError(f"No ratio directories found under {args.runs_dir}")

    endpoint_rows = collect_endpoint_rows(args.runs_dir, ratios)
    barrier_rows = collect_interpolation_rows(
        args.runs_dir, ratios, "barrier_summary.csv"
    )
    gap_rows = collect_interpolation_rows(
        args.runs_dir, ratios, "retrain_gap_summary.csv"
    )

    wrote_any = False
    wrote_any |= write_csv(
        os.path.join(output_dir, "all_endpoint_metrics.csv"),
        endpoint_rows,
        preferred_prefix=["ratio", "method", "source_csv"],
    )
    wrote_any |= write_csv(
        os.path.join(output_dir, "all_barrier_summary.csv"),
        barrier_rows,
        preferred_prefix=["ratio", "source_csv"],
    )
    wrote_any |= write_csv(
        os.path.join(output_dir, "all_retrain_gap_summary.csv"),
        gap_rows,
        preferred_prefix=["ratio", "source_csv", "run"],
    )

    if not wrote_any:
        print(f"No aggregate CSVs were written from {os.path.abspath(args.runs_dir)}")
        return

    print(f"Wrote aggregate CSVs to {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    main()
