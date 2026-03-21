import copy
import csv
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

import arg_parser
import experiment_helpers as experiment
import utils


SERIES_ORDER = ["endpoint_a", "endpoint_b", "avg_ab", "merge_best", "retrain"]
SERIES_LABELS = {
    "endpoint_a": "A",
    "endpoint_b": "B",
    "avg_ab": "avg(A,B)",
    "merge_best": "merge_best",
    "retrain": "retrain",
}
SERIES_STYLES = {
    "endpoint_a": {"color": "#1f77b4", "linestyle": "-", "marker": "o"},
    "endpoint_b": {"color": "#ff7f0e", "linestyle": "-", "marker": "o"},
    "avg_ab": {"color": "#2ca02c", "linestyle": "-", "marker": "o"},
    "merge_best": {"color": "#d62728", "linestyle": "-", "marker": "o"},
    "retrain": {"color": "#111111", "linestyle": "--", "marker": None},
}
METRIC_SPECS = [
    ("ua", "UA"),
    ("ra", "RA"),
    ("ta", "TA"),
    ("mia", "MIA"),
]


def build_parser():
    parser = arg_parser.build_parser()
    parser.description = "Plot step-wise A/B/avg/merge_best vs retrain metrics"
    parser.add_argument(
        "--runs_root",
        required=True,
        type=str,
        help="root directory that contains per-ratio run folders and summary/",
    )
    parser.add_argument(
        "--ratio_run_specs",
        default="10=10/step_connectivity,20=20/step_connectivity,30=30_epoch10/step_connectivity",
        type=str,
        help="comma-separated ratio=run_dir specs, relative to runs_root",
    )
    parser.add_argument(
        "--summary_dir",
        default=None,
        type=str,
        help="directory where the summary CSV/PNG files are written (defaults to <runs_root>/summary)",
    )
    parser.add_argument(
        "--output_csv",
        default="step_axis_ab_avg_merge_best_vs_retrain_metrics.csv",
        type=str,
        help="output CSV filename written under summary_dir",
    )
    parser.add_argument(
        "--output_png",
        default="step_axis_ab_avg_merge_best_vs_retrain_ua_ra_ta_mia.png",
        type=str,
        help="output PNG filename written under summary_dir",
    )
    return parser


def parse_ratio_run_specs(spec):
    parsed = []
    for token in str(spec).split(","):
        token = token.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError("Each ratio run spec must be ratio=relative/run_dir")
        ratio_text, relative_dir = token.split("=", 1)
        ratio = int(ratio_text.strip())
        relative_dir = relative_dir.strip()
        if not relative_dir:
            raise ValueError("Relative run dir cannot be empty")
        parsed.append({"ratio": ratio, "relative_dir": relative_dir})
    return parsed


def read_csv_rows(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def finite_float(value):
    if value in (None, "", "None"):
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def connectivity_dir_from_spec(runs_root, relative_dir):
    root = os.path.abspath(os.path.join(runs_root, relative_dir))
    if os.path.basename(root) == "connectivity":
        return root
    return os.path.join(root, "connectivity")


def load_retrain_metrics(path):
    rows = read_csv_rows(path)
    if not rows:
        raise ValueError(f"No retrain metrics found in {path}")
    row = rows[-1]
    return {
        "ua": finite_float(row.get("ua")),
        "ra": finite_float(row.get("acc_retain")),
        "ta": finite_float(row.get("acc_test")),
        "mia": finite_float(row.get("mia")),
    }


def ratio_eval_args(base_args, checkpoint_path):
    checkpoint = experiment.load_checkpoint_file(checkpoint_path, torch.device("cpu"))
    _, metadata = experiment.extract_state_dict(checkpoint)

    args = copy.deepcopy(base_args)
    args.forget_seed = int(metadata.get("forget_seed"))
    args.unlearn_seed = int(metadata.get("unlearn_seed"))
    args.batch_size = int(metadata.get("batch_size") or args.batch_size)
    args.model_path = metadata.get("base_checkpoint_path") or args.model_path
    args.forget_index_path = metadata.get("forget_index_path") or args.forget_index_path
    args = experiment.prepare_experiment_args(args)
    return args


def build_eval_context(args):
    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))
        device = torch.device(f"cuda:{int(args.gpu)}")
    else:
        device = torch.device("cpu")

    utils.setup_seed(args.forget_seed)
    model, _train_loader_full, val_loader, test_loader, marked_loader = utils.setup_model_dataset(args)
    model.to(device)
    utils.setup_seed(args.unlearn_seed)
    data_loaders, forget_dataset, retain_dataset = experiment.build_eval_data_loaders(
        marked_loader, val_loader, test_loader, args
    )
    criterion = nn.CrossEntropyLoss()
    return {
        "args": args,
        "device": device,
        "model": model,
        "criterion": criterion,
        "data_loaders": data_loaders,
        "forget_dataset": forget_dataset,
        "retain_dataset": retain_dataset,
    }


def metric_row_from_eval(metrics):
    return {
        "ua": finite_float(metrics["ua"]),
        "ra": finite_float(metrics["acc_retain"]),
        "ta": finite_float(metrics["acc_test"]),
        "mia": finite_float(metrics["mia"]),
    }


def evaluate_state(model, state_dict, eval_context):
    model.load_state_dict(state_dict, strict=False)
    metrics = experiment.evaluate_model(
        model,
        eval_context["data_loaders"],
        eval_context["forget_dataset"],
        eval_context["retain_dataset"],
        eval_context["criterion"],
        eval_context["args"],
        compute_mia=True,
    )
    return metric_row_from_eval(metrics)


def average_metric_rows(row_a, row_b):
    averaged = {}
    for key in ["ua", "ra", "ta", "mia"]:
        values = [finite_float(row_a[key]), finite_float(row_b[key])]
        finite_values = [value for value in values if math.isfinite(value)]
        averaged[key] = float(np.mean(finite_values)) if finite_values else np.nan
    return averaged


def record_series_row(rows, ratio, step, epoch_float, series_type, alpha, metrics):
    rows.append(
        {
            "ratio": ratio,
            "step": step,
            "epoch_float": epoch_float,
            "series_type": series_type,
            "series_label": SERIES_LABELS[series_type],
            "alpha": alpha,
            "ua": metrics["ua"],
            "ra": metrics["ra"],
            "ta": metrics["ta"],
            "mia": metrics["mia"],
        }
    )


def evaluate_ratio(base_args, runs_root, ratio, relative_dir):
    connectivity_dir = connectivity_dir_from_spec(runs_root, relative_dir)
    checkpoint_manifest_path = os.path.join(connectivity_dir, "checkpoint_manifest.csv")
    full_eval_path = os.path.join(connectivity_dir, "full_eval.csv")
    retrain_metrics_path = os.path.join(runs_root, str(ratio), "retrain", "endpoint_metrics.csv")

    checkpoint_rows = read_csv_rows(checkpoint_manifest_path)
    full_eval_rows = read_csv_rows(full_eval_path)
    retrain_metrics = load_retrain_metrics(retrain_metrics_path)

    manifest_by_step = {}
    for row in checkpoint_rows:
        step = int(row["step"])
        manifest_by_step.setdefault(step, {})[row["model_id"]] = row

    merge_best_by_step = {}
    for row in full_eval_rows:
        if row["candidate_type"] != "merge_best":
            continue
        merge_best_by_step[int(row["step"])] = row

    if not manifest_by_step:
        raise ValueError(f"No checkpoint rows found in {checkpoint_manifest_path}")

    first_step = min(manifest_by_step.keys())
    first_checkpoint_path = manifest_by_step[first_step]["A"]["ckpt_path"]
    ratio_args = ratio_eval_args(base_args, first_checkpoint_path)
    eval_context = build_eval_context(ratio_args)
    device = eval_context["device"]
    model = eval_context["model"]

    output_rows = []
    for step in sorted(manifest_by_step.keys()):
        step_manifest = manifest_by_step[step]
        row_a = step_manifest["A"]
        row_b = step_manifest["B"]
        epoch_float = finite_float(row_a["epoch_float"])
        checkpoint_a = experiment.load_checkpoint_file(row_a["ckpt_path"], device)
        checkpoint_b = experiment.load_checkpoint_file(row_b["ckpt_path"], device)
        state_a, _meta_a = experiment.extract_state_dict(checkpoint_a)
        state_b, _meta_b = experiment.extract_state_dict(checkpoint_b)

        metrics_a = evaluate_state(model, state_a, eval_context)
        metrics_b = evaluate_state(model, state_b, eval_context)
        metrics_avg = average_metric_rows(metrics_a, metrics_b)

        record_series_row(output_rows, ratio, step, epoch_float, "endpoint_a", 0.0, metrics_a)
        record_series_row(output_rows, ratio, step, epoch_float, "endpoint_b", 1.0, metrics_b)
        record_series_row(output_rows, ratio, step, epoch_float, "avg_ab", 0.5, metrics_avg)

        merge_best = merge_best_by_step.get(step)
        merge_alpha = finite_float(merge_best["alpha"]) if merge_best else np.nan
        if math.isfinite(merge_alpha):
            interpolated_state = experiment.interpolate_state_dict(state_a, state_b, merge_alpha)
            metrics_merge = evaluate_state(model, interpolated_state, eval_context)
        else:
            metrics_merge = {"ua": np.nan, "ra": np.nan, "ta": np.nan, "mia": np.nan}
        record_series_row(output_rows, ratio, step, epoch_float, "merge_best", merge_alpha, metrics_merge)
        record_series_row(output_rows, ratio, step, epoch_float, "retrain", np.nan, retrain_metrics)

    return output_rows


def plot_rows(rows, output_path):
    ratios = sorted({int(row["ratio"]) for row in rows})
    fig, axes = plt.subplots(len(ratios), len(METRIC_SPECS), figsize=(18, 4.8 * len(ratios)), sharex=False)
    if len(ratios) == 1:
        axes = np.array([axes])

    legend_handles = []
    legend_labels = []

    for row_index, ratio in enumerate(ratios):
        ratio_rows = [row for row in rows if int(row["ratio"]) == ratio]
        for col_index, (metric_key, metric_title) in enumerate(METRIC_SPECS):
            axis = axes[row_index, col_index]
            for series_type in SERIES_ORDER:
                series_rows = [row for row in ratio_rows if row["series_type"] == series_type]
                series_rows.sort(key=lambda row: int(row["step"]))
                if not series_rows:
                    continue
                steps = [int(row["step"]) for row in series_rows]
                values = [finite_float(row[metric_key]) for row in series_rows]
                style = SERIES_STYLES[series_type]
                if series_type == "retrain":
                    handle = axis.axhline(
                        values[0],
                        color=style["color"],
                        linestyle=style["linestyle"],
                        linewidth=1.8,
                        label=SERIES_LABELS[series_type],
                    )
                else:
                    (handle,) = axis.plot(
                        steps,
                        values,
                        color=style["color"],
                        linestyle=style["linestyle"],
                        marker=style["marker"],
                        markersize=3.5,
                        linewidth=1.6,
                        label=SERIES_LABELS[series_type],
                    )
                if row_index == 0 and col_index == 0:
                    legend_handles.append(handle)
                    legend_labels.append(SERIES_LABELS[series_type])
            if row_index == 0:
                axis.set_title(metric_title)
            if col_index == 0:
                axis.set_ylabel(f"{ratio}%")
            if row_index == len(ratios) - 1:
                axis.set_xlabel("step")
            axis.grid(alpha=0.25)

    fig.suptitle("Step-wise A/B/avg/merge_best vs retrain")
    fig.legend(legend_handles, legend_labels, loc="upper center", ncol=len(legend_labels), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    args = build_parser().parse_args()
    args = experiment.prepare_experiment_args(args)

    ratio_specs = parse_ratio_run_specs(args.ratio_run_specs)
    runs_root = os.path.abspath(args.runs_root)
    summary_dir = os.path.abspath(args.summary_dir or os.path.join(runs_root, "summary"))
    os.makedirs(summary_dir, exist_ok=True)

    all_rows = []
    for spec in ratio_specs:
        ratio_rows = evaluate_ratio(args, runs_root, spec["ratio"], spec["relative_dir"])
        all_rows.extend(ratio_rows)

    csv_path = os.path.join(summary_dir, args.output_csv)
    experiment.write_csv(
        csv_path,
        all_rows,
        fieldnames=[
            "ratio",
            "step",
            "epoch_float",
            "series_type",
            "series_label",
            "alpha",
            "ua",
            "ra",
            "ta",
            "mia",
        ],
    )
    png_path = os.path.join(summary_dir, args.output_png)
    plot_rows(all_rows, png_path)
    print(f"Wrote {csv_path}")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
