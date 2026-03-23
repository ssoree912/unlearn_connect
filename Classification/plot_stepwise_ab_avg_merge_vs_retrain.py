import argparse
import csv
import math
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


BARRIER_METRICS = [
    ("retain_barrier", "Retain Barrier"),
    ("forget_barrier", "Forget Barrier"),
    ("val_barrier", "Val Barrier"),
    ("test_barrier", "Test Barrier"),
]
CURVE_METRICS = [
    ("dr_loss", "DR Eval Loss"),
    ("val_loss", "Val Eval Loss"),
    ("j_score", "J Score"),
    ("ua", "UA"),
]
METRIC_LABELS = {
    "ua": "UA",
    "dr_acc": "DR Acc",
    "df_acc": "DF Acc",
    "val_acc": "Val Acc",
    "test_acc": "Test Acc",
    "dr_loss": "DR Loss",
    "df_loss": "DF Loss",
    "val_loss": "Val Loss",
    "test_loss": "Test Loss",
    "mia": "MIA",
}
SERIES_ORDER = ["main", "alt", "avg", "merge_05", "merge_center", "merge_best", "retrain"]
SERIES_STYLES = {
    "main": {"dash": "solid", "symbol": "circle-open"},
    "alt": {"dash": "solid", "symbol": "square-open"},
    "avg": {"dash": "dashdot", "symbol": "diamond-open"},
    "merge_05": {"dash": "dot", "symbol": "x"},
    "merge_center": {"dash": "dot", "symbol": "x"},
    "merge_best": {"dash": "solid", "symbol": "circle"},
    "retrain": {"dash": "dash", "symbol": "line-ns-open"},
}
COMPARISON_LABELS = {
    "low": "lower lr",
    "high": "higher lr",
    "same": "same lr",
}
COMPARISON_ORDER = {"low": 0, "high": 1, "same": 2}
COMPARISON_COLORS = {
    "low": "#1f77b4",
    "high": "#2ca02c",
    "same": "#ff7f0e",
}


def build_parser():
    parser = argparse.ArgumentParser(
        description="Summarize connectivity runs into reference-style CSV/HTML reports plus path-curve and barrier PNGs"
    )
    parser.add_argument("--runs_root", required=True, type=str, help="root directory that contains per-ratio connectivity run folders")
    parser.add_argument("--retrain_root", default=None, type=str, help="root directory that contains <ratio>/retrain/endpoint_metrics.csv")
    parser.add_argument("--prepend_runs_root", default=None, type=str, help="optional earlier runs_root to prepend before the current runs")
    parser.add_argument("--prepend_retrain_root", default=None, type=str, help="optional retrain_root for prepend_runs_root")
    parser.add_argument("--summary_dir", default=None, type=str, help="output directory for the generated summary artifacts")
    parser.add_argument("--output_prefix", default=None, type=str, help="prefix for generated filenames")
    parser.add_argument("--ratio_run_specs", default="", type=str, help="optional comma-separated ratio=relative/connectivity_dir specs; auto-discovered if empty")
    parser.add_argument("--prepend_ratio_run_specs", default="", type=str, help="optional ratio_run_specs for prepend_runs_root")
    parser.add_argument("--epoch_offset", default=0.0, type=float, help="epoch offset applied to the current runs_root data")
    parser.add_argument("--prepend_epoch_offset", default=0.0, type=float, help="epoch offset applied to prepend_runs_root data")
    parser.add_argument(
        "--metrics",
        default="ua,dr_acc,df_acc,val_acc,test_acc,dr_loss,df_loss,val_loss,test_loss,mia",
        type=str,
        help="metrics written into the CSV reports",
    )
    parser.add_argument(
        "--html_metrics",
        default="ua,dr_acc,df_acc,val_acc,test_acc,mia",
        type=str,
        help="metrics rendered as *_by_ratio.html",
    )
    parser.add_argument(
        "--plot_series",
        default="main,alt,merge_05,merge_best,retrain,merge_center",
        type=str,
        help="series rendered in the HTML plots",
    )
    return parser


def parse_csv_list(spec):
    return [token.strip() for token in str(spec).split(",") if token.strip()]


def finite_float(value):
    if value in (None, "", "None", "nan", "NaN"):
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def safe_mean(values):
    finite_values = [finite_float(value) for value in values if math.isfinite(finite_float(value))]
    if not finite_values:
        return np.nan
    return float(np.mean(finite_values))


def read_csv_rows(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def normalize_cell(value):
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: normalize_cell(row.get(name)) for name in fieldnames})


def default_output_prefix(runs_root):
    runs_root = os.path.abspath(runs_root)
    base = os.path.basename(runs_root)
    if base == "runs":
        base = os.path.basename(os.path.dirname(runs_root))
    if "_epoch" in base:
        return base.split("_epoch", 1)[0]
    return base


def default_summary_dir(runs_root):
    return os.path.join(os.path.abspath(runs_root), "summary")


def connectivity_dir_from_value(runs_root, relative_dir):
    root = os.path.abspath(os.path.join(runs_root, relative_dir))
    if os.path.isfile(os.path.join(root, "full_eval.csv")):
        return root
    candidate = os.path.join(root, "connectivity")
    if os.path.isfile(os.path.join(candidate, "full_eval.csv")):
        return candidate
    raise FileNotFoundError(f"Could not resolve connectivity dir from {relative_dir}")


def parse_ratio_run_specs(runs_root, spec):
    parsed = []
    for token in parse_csv_list(spec):
        if "=" not in token:
            raise ValueError("Each ratio_run_spec must be ratio=relative/connectivity_dir")
        ratio_text, relative_dir = token.split("=", 1)
        ratio = int(ratio_text.strip())
        connectivity_dir = connectivity_dir_from_value(runs_root, relative_dir.strip())
        path_label = os.path.basename(os.path.dirname(connectivity_dir)) if os.path.basename(connectivity_dir) == "connectivity" else os.path.basename(connectivity_dir)
        parsed.append({"ratio": ratio, "connectivity_dir": connectivity_dir, "path_label": path_label})
    return parsed


def auto_discover_specs(runs_root):
    specs = []
    runs_root = os.path.abspath(runs_root)
    for ratio_name in sorted(os.listdir(runs_root)):
        ratio_dir = os.path.join(runs_root, ratio_name)
        if not os.path.isdir(ratio_dir):
            continue
        try:
            ratio = int(ratio_name)
        except ValueError:
            continue
        for root, _dirs, files in os.walk(ratio_dir):
            file_set = set(files)
            if {"full_eval.csv", "step_summary.csv", "path_scan.csv"}.issubset(file_set):
                path_label = os.path.basename(os.path.dirname(root)) if os.path.basename(root) == "connectivity" else os.path.basename(root)
                specs.append({"ratio": ratio, "connectivity_dir": root, "path_label": path_label})
    specs.sort(key=lambda item: (item["ratio"], item["path_label"]))
    return specs


def canonical_metric(row, key):
    mapping = {
        "ua": ["ua"],
        "dr_acc": ["dr_acc", "acc_retain"],
        "df_acc": ["df_acc", "acc_forget"],
        "val_acc": ["val_acc", "acc_val"],
        "test_acc": ["test_acc", "acc_test"],
        "dr_loss": ["dr_loss", "loss_retain"],
        "df_loss": ["df_loss", "loss_forget"],
        "val_loss": ["val_loss", "loss_val"],
        "test_loss": ["test_loss", "loss_test"],
        "mia": ["mia"],
    }
    for candidate in mapping.get(key, [key]):
        value = finite_float(row.get(candidate))
        if math.isfinite(value):
            return value
    return np.nan


def load_retrain_metrics(retrain_root, ratio):
    if retrain_root is None:
        return None
    path = os.path.join(os.path.abspath(retrain_root), str(ratio), "retrain", "endpoint_metrics.csv")
    if not os.path.isfile(path):
        return None
    rows = read_csv_rows(path)
    if not rows:
        return None
    row = rows[-1]
    metrics = {
        key: canonical_metric(row, key)
        for key in ["ua", "dr_acc", "df_acc", "val_acc", "test_acc", "dr_loss", "df_loss", "val_loss", "test_loss", "mia"]
    }
    return metrics


def load_run_manifest(connectivity_dir):
    manifest_path = os.path.join(connectivity_dir, "run_manifest.csv")
    if not os.path.isfile(manifest_path):
        return {}
    rows = read_csv_rows(manifest_path)
    return rows[0] if rows else {}


def load_best_curve_map(connectivity_dir):
    step_summary_path = os.path.join(connectivity_dir, "step_summary.csv")
    if not os.path.isfile(step_summary_path):
        return {}
    rows = read_csv_rows(step_summary_path)
    out = {}
    for row in rows:
        best_curve = (row.get("best_curve_id") or row.get("best_control") or "").strip()
        if best_curve:
            out[int(row["step"])] = best_curve
    return out


def parse_pair_metadata(spec, manifest):
    pair_id = manifest.get("pair_id", "") or spec.get("path_label", "")
    notes = manifest.get("notes", "") or ""
    path_label = spec.get("path_label", "")

    main_lr = np.nan
    alt_lr = np.nan
    match = re.search(r"main_lr=([0-9.]+)\s+alt_lr=([0-9.]+)", notes)
    if match:
        main_lr = float(match.group(1))
        alt_lr = float(match.group(2))
    else:
        match = re.search(r"lr([0-9.]+)_vs_([0-9.]+)", pair_id)
        if match:
            main_lr = float(match.group(1))
            alt_lr = float(match.group(2))
        else:
            match = re.search(r"main([0-9]+)_vs_([0-9]+)", path_label)
            if match:
                main_lr = float(match.group(1)) / 1000.0
                alt_lr = float(match.group(2)) / 1000.0

    comparison = "same"
    comparison_label = path_label or pair_id or "path"
    lr_gap = np.nan
    if math.isfinite(main_lr) and math.isfinite(alt_lr):
        if alt_lr < main_lr:
            comparison = "low"
        elif alt_lr > main_lr:
            comparison = "high"
        comparison_label = f"main {main_lr:.3f} vs alt {alt_lr:.3f}"
        lr_gap = alt_lr - main_lr

    return {
        "pair_id": pair_id,
        "path_label": path_label,
        "main_lr": main_lr,
        "alt_lr": alt_lr,
        "lr_gap": lr_gap,
        "comparison": comparison,
        "comparison_label": comparison_label,
    }


def metric_payload(row, metrics):
    return {metric: canonical_metric(row, metric) for metric in metrics}


def average_series_metrics(row_a, row_b, metrics):
    averaged = {}
    for metric_name in metrics:
        averaged[metric_name] = safe_mean([canonical_metric(row_a, metric_name), canonical_metric(row_b, metric_name)])
    return averaged


def append_metric_triples(target, metrics, metric_values, retrain_metrics):
    for metric_name in metrics:
        value = metric_values.get(metric_name, np.nan)
        retrain_value = retrain_metrics.get(metric_name, np.nan) if retrain_metrics is not None else np.nan
        gap = value - retrain_value if math.isfinite(value) and math.isfinite(retrain_value) else np.nan
        target[metric_name] = value
        target[f"gap_vs_retrain_{metric_name}"] = gap
        target[f"retrain_{metric_name}"] = retrain_value


def choose_path_rows_for_step(step_rows, best_curve):
    coarse_rows = [row for row in step_rows if row.get("scan_stage", "") == "coarse"]
    rows = coarse_rows or step_rows
    alpha_rows = [row for row in rows if math.isfinite(finite_float(row.get("alpha")))]
    rows = alpha_rows or rows
    if not rows:
        return []
    if "curve_id" not in rows[0]:
        return rows
    curve_ids = sorted({(row.get("curve_id") or "").strip() for row in rows if (row.get("curve_id") or "").strip()})
    if len(curve_ids) <= 1:
        return rows
    if best_curve:
        filtered = [row for row in rows if (row.get("curve_id") or "").strip() == best_curve]
        if filtered:
            return filtered
    return rows


def shift_epoch(rows, offset):
    if not offset:
        return rows
    shifted = []
    for row in rows:
        row = dict(row)
        epoch_value = finite_float(row.get("epoch_float"))
        if math.isfinite(epoch_value):
            row["epoch_float"] = epoch_value + offset
        shifted.append(row)
    return shifted


def collect_dataset(runs_root, retrain_root, ratio_run_specs, metrics, epoch_offset):
    specs = parse_ratio_run_specs(runs_root, ratio_run_specs) if ratio_run_specs else auto_discover_specs(runs_root)
    if not specs:
        raise ValueError(f"No connectivity result folders found under {runs_root}")

    performance_long_rows = []
    performance_wide_rows = []
    performance_row_rows = []
    barrier_rows = []
    path_curve_rows = []
    seen_merge_center = False
    prepared_specs = []

    for spec in specs:
        manifest = load_run_manifest(spec["connectivity_dir"])
        spec = dict(spec)
        spec.update(parse_pair_metadata(spec, manifest))
        prepared_specs.append(spec)
        retrain_metrics = load_retrain_metrics(retrain_root, spec["ratio"])

        full_eval_rows = read_csv_rows(os.path.join(spec["connectivity_dir"], "full_eval.csv"))
        by_step = defaultdict(dict)
        for row in full_eval_rows:
            by_step[int(row["step"])][row["candidate_type"]] = row

        for step in sorted(by_step):
            candidates = by_step[step]
            main_row = candidates.get("endpoint_a")
            alt_row = candidates.get("endpoint_b")
            merge05_row = candidates.get("merge_05")
            merge_center_row = candidates.get("merge_center")
            merge_best_row = candidates.get("merge_best")
            seen_merge_center = seen_merge_center or merge_center_row is not None
            base_row = merge_best_row or merge05_row or merge_center_row or main_row or alt_row or next(iter(candidates.values()))
            epoch_float = finite_float(base_row.get("epoch_float"))
            if math.isfinite(epoch_float):
                epoch_float += epoch_offset

            series_payloads = []
            if main_row is not None:
                series_payloads.append(("main", "main", finite_float(main_row.get("alpha")), metric_payload(main_row, metrics)))
            if alt_row is not None:
                series_payloads.append(("alt", "alt", finite_float(alt_row.get("alpha")), metric_payload(alt_row, metrics)))
            if main_row is not None and alt_row is not None:
                series_payloads.append(("avg", "avg", 0.5, average_series_metrics(main_row, alt_row, metrics)))
            if merge05_row is not None:
                series_payloads.append(("merge_05", "merge_05", finite_float(merge05_row.get("alpha")), metric_payload(merge05_row, metrics)))
            if merge_center_row is not None:
                series_payloads.append(("merge_center", "merge_center", finite_float(merge_center_row.get("alpha")), metric_payload(merge_center_row, metrics)))
            if merge_best_row is not None:
                series_payloads.append(("merge_best", "merge_best", finite_float(merge_best_row.get("alpha")), metric_payload(merge_best_row, metrics)))
            if retrain_metrics is not None:
                series_payloads.append(("retrain", "retrain", np.nan, {metric: retrain_metrics.get(metric, np.nan) for metric in metrics}))

            for series_type, series_label, alpha, metric_values in series_payloads:
                row = {
                    "ratio": spec["ratio"],
                    "comparison": spec["comparison"],
                    "comparison_label": spec["comparison_label"],
                    "main_lr": spec["main_lr"],
                    "alt_lr": spec["alt_lr"],
                    "lr_gap": spec["lr_gap"],
                    "step": step,
                    "epoch_float": epoch_float,
                    "series_type": series_type,
                    "series_label": series_label,
                    "alpha": alpha,
                }
                append_metric_triples(row, metrics, metric_values, retrain_metrics)
                performance_long_rows.append(row)
                for metric_name in metrics:
                    performance_row_rows.append(
                        {
                            "ratio": spec["ratio"],
                            "comparison": spec["comparison"],
                            "comparison_label": spec["comparison_label"],
                            "main_lr": spec["main_lr"],
                            "alt_lr": spec["alt_lr"],
                            "lr_gap": spec["lr_gap"],
                            "step": step,
                            "epoch_float": epoch_float,
                            "series_type": series_type,
                            "series_label": series_label,
                            "alpha": alpha,
                            "metric_name": metric_name,
                            "metric_value": metric_values.get(metric_name, np.nan),
                            "retrain_value": retrain_metrics.get(metric_name, np.nan) if retrain_metrics is not None else np.nan,
                            "gap_vs_retrain": row[f"gap_vs_retrain_{metric_name}"],
                        }
                    )

            series_metrics = {series_type: values for series_type, _label, _alpha, values in series_payloads}
            series_alpha = {series_type: alpha for series_type, _label, alpha, _values in series_payloads}
            wide = {
                "ratio": spec["ratio"],
                "comparison": spec["comparison"],
                "comparison_label": spec["comparison_label"],
                "main_lr": spec["main_lr"],
                "alt_lr": spec["alt_lr"],
                "lr_gap": spec["lr_gap"],
                "step": step,
                "epoch_float": epoch_float,
                "merge_best_alpha": series_alpha.get("merge_best", np.nan),
                "merge_05_alpha": series_alpha.get("merge_05", np.nan),
                "merge_center_alpha": series_alpha.get("merge_center", np.nan),
            }
            for metric_name in metrics:
                retrain_value = retrain_metrics.get(metric_name, np.nan) if retrain_metrics is not None else np.nan
                for series_type in ["main", "alt", "avg", "merge_best", "merge_05", "merge_center"]:
                    value = series_metrics.get(series_type, {}).get(metric_name, np.nan)
                    wide[f"{series_type}_{metric_name}"] = value
                    wide[f"gap_{series_type}_vs_retrain_{metric_name}"] = value - retrain_value if math.isfinite(value) and math.isfinite(retrain_value) else np.nan
                wide[f"retrain_{metric_name}"] = retrain_value
            performance_wide_rows.append(wide)

        for row in read_csv_rows(os.path.join(spec["connectivity_dir"], "step_summary.csv")):
            epoch_float = finite_float(row.get("epoch_float"))
            if math.isfinite(epoch_float):
                epoch_float += epoch_offset
            barrier_rows.append(
                {
                    "ratio": spec["ratio"],
                    "comparison": spec["comparison"],
                    "comparison_label": spec["comparison_label"],
                    "main_lr": spec["main_lr"],
                    "alt_lr": spec["alt_lr"],
                    "lr_gap": spec["lr_gap"],
                    "step": int(row["step"]),
                    "epoch_float": epoch_float,
                    "alpha_star_coarse": finite_float(row.get("alpha_star_coarse")),
                    "alpha_star_refined": finite_float(row.get("alpha_star_refined")),
                    "delta_05": finite_float(row.get("delta_05")),
                    "delta_int": finite_float(row.get("delta_int")),
                    "retain_barrier": finite_float(row.get("retain_barrier")),
                    "forget_barrier": finite_float(row.get("forget_barrier")),
                    "val_barrier": finite_float(row.get("val_barrier")),
                    "test_barrier": finite_float(row.get("test_barrier")),
                }
            )

        grouped_scan = defaultdict(list)
        best_curve_map = load_best_curve_map(spec["connectivity_dir"])
        for row in read_csv_rows(os.path.join(spec["connectivity_dir"], "path_scan.csv")):
            grouped_scan[int(row["step"])] .append(row)
        for step in sorted(grouped_scan):
            selected_rows = choose_path_rows_for_step(grouped_scan[step], best_curve_map.get(step, ""))
            for row in selected_rows:
                alpha = finite_float(row.get("alpha"))
                if not math.isfinite(alpha):
                    continue
                epoch_float = finite_float(row.get("epoch_float"))
                if math.isfinite(epoch_float):
                    epoch_float += epoch_offset
                path_curve_rows.append(
                    {
                        "ratio": spec["ratio"],
                        "comparison": spec["comparison"],
                        "comparison_label": spec["comparison_label"],
                        "main_lr": spec["main_lr"],
                        "alt_lr": spec["alt_lr"],
                        "step": step,
                        "epoch_float": epoch_float,
                        "alpha": alpha,
                        "ua": finite_float(row.get("ua")),
                        "dr_loss": finite_float(row.get("dr_loss")),
                        "val_loss": finite_float(row.get("val_loss")),
                        "j_score": finite_float(row.get("j_score")),
                        "feasible": int(finite_float(row.get("feasible"))) if math.isfinite(finite_float(row.get("feasible"))) else "",
                    }
                )

    return prepared_specs, performance_long_rows, performance_wide_rows, performance_row_rows, barrier_rows, path_curve_rows, seen_merge_center


def comparison_sort_key(item):
    comparison = item.get("comparison", "same")
    alt_lr = finite_float(item.get("alt_lr"))
    return (COMPARISON_ORDER.get(comparison, 99), alt_lr if math.isfinite(alt_lr) else 0.0, item.get("comparison_label", ""))


def series_sort_key(item):
    return SERIES_ORDER.index(item) if item in SERIES_ORDER else len(SERIES_ORDER)


def pretty_axis_label(row):
    comparison = row.get("comparison", "same")
    relation = COMPARISON_LABELS.get(comparison, comparison)
    main_lr = finite_float(row.get("main_lr"))
    alt_lr = finite_float(row.get("alt_lr"))
    if math.isfinite(main_lr) and math.isfinite(alt_lr):
        return f"main vs {relation}\n{main_lr:.3f} vs {alt_lr:.3f}"
    return row.get("comparison_label", comparison)


def unique_epoch_keys(rows):
    keys = sorted({round(finite_float(row["epoch_float"]), 6) for row in rows if math.isfinite(finite_float(row["epoch_float"]))})
    return keys


def plot_path_curves_png(rows, ratio, output_path):
    ratio_rows = [row for row in rows if int(row["ratio"]) == int(ratio) and math.isfinite(finite_float(row.get("epoch_float")))]
    if not ratio_rows:
        return False
    comparison_groups = sorted({row["comparison_label"] for row in ratio_rows}, key=lambda label: comparison_sort_key(next(item for item in ratio_rows if item["comparison_label"] == label)))
    epoch_keys = unique_epoch_keys(ratio_rows)
    if not comparison_groups or not epoch_keys:
        return False

    colors = plt.cm.viridis(np.linspace(0.03, 0.97, len(epoch_keys)))
    color_map = {epoch_key: colors[index] for index, epoch_key in enumerate(epoch_keys)}
    fig, axes = plt.subplots(len(comparison_groups), len(CURVE_METRICS), figsize=(18, 4.2 * len(comparison_groups)), sharex=True)
    axes = np.atleast_2d(axes)

    legend_handles = []
    legend_labels = []
    for row_index, comparison_label in enumerate(comparison_groups):
        group_rows = [row for row in ratio_rows if row["comparison_label"] == comparison_label]
        row_label = pretty_axis_label(group_rows[0])
        for col_index, (metric_key, metric_title) in enumerate(CURVE_METRICS):
            axis = axes[row_index, col_index]
            for epoch_key in epoch_keys:
                series = [row for row in group_rows if round(finite_float(row.get("epoch_float")), 6) == epoch_key]
                series.sort(key=lambda item: float(item["alpha"]))
                if not series:
                    continue
                handle, = axis.plot(
                    [float(item["alpha"]) for item in series],
                    [finite_float(item[metric_key]) for item in series],
                    color=color_map[epoch_key],
                    linewidth=1.4,
                    alpha=0.95,
                    label=f"{epoch_key:.1f}",
                )
                if row_index == 0 and col_index == 0:
                    legend_handles.append(handle)
                    legend_labels.append(f"{epoch_key:.1f}")
            if row_index == 0:
                axis.set_title(metric_title)
            if col_index == 0:
                axis.set_ylabel(row_label)
            if row_index == len(comparison_groups) - 1:
                axis.set_xlabel("alpha")
            axis.grid(alpha=0.2)

    fig.suptitle(f"Ratio {ratio}%: Path Curves by Epoch")
    fig.legend(legend_handles, legend_labels, loc="lower center", ncol=min(9, max(1, len(legend_labels))), frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return True


def plot_barrier_png(rows, ratio, output_path):
    ratio_rows = [row for row in rows if int(row["ratio"]) == int(ratio) and math.isfinite(finite_float(row.get("epoch_float")))]
    if not ratio_rows:
        return False
    groups = sorted({row["comparison_label"] for row in ratio_rows}, key=lambda label: comparison_sort_key(next(item for item in ratio_rows if item["comparison_label"] == label)))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False)
    for axis, (metric_key, metric_label) in zip(axes.flat, BARRIER_METRICS):
        for comparison_label in groups:
            series = [row for row in ratio_rows if row["comparison_label"] == comparison_label]
            series.sort(key=lambda item: finite_float(item["epoch_float"]))
            if not series:
                continue
            first = series[0]
            color = COMPARISON_COLORS.get(first.get("comparison", "same"), "#444444")
            axis.plot(
                [finite_float(item["epoch_float"]) for item in series],
                [finite_float(item[metric_key]) for item in series],
                label=pretty_axis_label(first).replace("\n", " | "),
                marker="o",
                linewidth=1.8,
                color=color,
            )
        axis.set_title(metric_label)
        axis.set_xlabel("epoch")
        axis.grid(alpha=0.25)
    axes[0, 0].legend(frameon=False)
    fig.suptitle(f"Ratio {ratio}%: Barrier by Epoch")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return True


def build_metric_html(long_rows, metric_name, plot_series):
    metric_rows = [row for row in long_rows if row["series_type"] in plot_series and math.isfinite(finite_float(row.get(metric_name))) and math.isfinite(finite_float(row.get("epoch_float")))]
    ratios = sorted({int(row["ratio"]) for row in metric_rows})
    if not ratios:
        return None

    fig = make_subplots(rows=len(ratios), cols=1, subplot_titles=[f"ratio {ratio}" for ratio in ratios], vertical_spacing=0.08)
    legend_seen = set()

    for row_index, ratio in enumerate(ratios, start=1):
        ratio_rows = [row for row in metric_rows if int(row["ratio"]) == ratio]
        epoch_values = sorted({round(finite_float(row["epoch_float"]), 6) for row in ratio_rows})
        retrain_rows = [row for row in ratio_rows if row["series_type"] == "retrain"]
        if retrain_rows and epoch_values and "retrain" in plot_series:
            retrain_value = finite_float(retrain_rows[0].get(metric_name))
            if math.isfinite(retrain_value):
                name = "retrain"
                fig.add_trace(
                    go.Scatter(
                        x=epoch_values,
                        y=[retrain_value] * len(epoch_values),
                        mode="lines",
                        name=name,
                        legendgroup=name,
                        showlegend=name not in legend_seen,
                        line={"color": "#111111", "dash": "dash", "width": 2},
                        hovertemplate=f"ratio {ratio}<br>retrain<br>epoch=%{{x:.2f}}<br>{METRIC_LABELS.get(metric_name, metric_name)}=%{{y:.4f}}<extra></extra>",
                    ),
                    row=row_index,
                    col=1,
                )
                legend_seen.add(name)

        comparison_labels = sorted(
            {row["comparison_label"] for row in ratio_rows if row["series_type"] != "retrain"},
            key=lambda label: comparison_sort_key(next(item for item in ratio_rows if item["comparison_label"] == label)),
        )
        for comparison_label in comparison_labels:
            comparison_rows = [row for row in ratio_rows if row["comparison_label"] == comparison_label]
            first = comparison_rows[0]
            color = COMPARISON_COLORS.get(first.get("comparison", "same"), "#444444")
            prefix = COMPARISON_LABELS.get(first.get("comparison", "same"), first.get("comparison", "same"))
            for series_type in plot_series:
                if series_type == "retrain":
                    continue
                series = [row for row in comparison_rows if row["series_type"] == series_type and math.isfinite(finite_float(row.get(metric_name))) and math.isfinite(finite_float(row.get("epoch_float")))]
                if not series:
                    continue
                series.sort(key=lambda item: (finite_float(item["epoch_float"]), int(item["step"])))
                style = SERIES_STYLES.get(series_type, SERIES_STYLES["merge_best"])
                name = f"{prefix} / {series_type}"
                fig.add_trace(
                    go.Scatter(
                        x=[finite_float(item["epoch_float"]) for item in series],
                        y=[finite_float(item[metric_name]) for item in series],
                        mode="lines+markers",
                        name=name,
                        legendgroup=name,
                        showlegend=name not in legend_seen,
                        line={"color": color, "dash": style["dash"], "width": 2},
                        marker={"symbol": style["symbol"], "size": 7, "color": color},
                        customdata=[
                            [item["comparison_label"], item["series_label"], item["step"], normalize_cell(item.get("alpha")), normalize_cell(item.get(f"gap_vs_retrain_{metric_name}"))]
                            for item in series
                        ],
                        hovertemplate=(
                            "ratio " + str(ratio) +
                            "<br>%{customdata[0]}" +
                            "<br>series=%{customdata[1]}" +
                            "<br>epoch=%{x:.2f}" +
                            "<br>step=%{customdata[2]}" +
                            f"<br>{METRIC_LABELS.get(metric_name, metric_name)}=%{{y:.4f}}" +
                            "<br>alpha=%{customdata[3]}" +
                            "<br>gap_vs_retrain=%{customdata[4]}" +
                            "<extra></extra>"
                        ),
                    ),
                    row=row_index,
                    col=1,
                )
                legend_seen.add(name)

        fig.update_xaxes(title_text="epoch", row=row_index, col=1)
        fig.update_yaxes(title_text=METRIC_LABELS.get(metric_name, metric_name), row=row_index, col=1)

    fig.update_layout(
        height=320 * len(ratios) + 120,
        width=1200,
        title=f"{METRIC_LABELS.get(metric_name, metric_name)} by ratio",
        hovermode="x unified",
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0.0},
        margin={"l": 70, "r": 30, "t": 100, "b": 50},
    )
    return fig


def build_pair_summary(barrier_rows, wide_rows):
    grouped_barriers = defaultdict(list)
    grouped_wide = defaultdict(list)
    for row in barrier_rows:
        grouped_barriers[(row["ratio"], row["comparison_label"])].append(row)
    for row in wide_rows:
        grouped_wide[(row["ratio"], row["comparison_label"])].append(row)

    summary_rows = []
    for key in sorted(grouped_barriers.keys() | grouped_wide.keys(), key=lambda item: (item[0], item[1])):
        ratio, comparison_label = key
        barrier_group = grouped_barriers.get(key, [])
        wide_group = grouped_wide.get(key, [])
        base = (barrier_group or wide_group)[0]
        delta_int_values = [finite_float(item.get("delta_int")) for item in barrier_group if math.isfinite(finite_float(item.get("delta_int")))]
        best_step = ""
        if delta_int_values:
            best_row = min((item for item in barrier_group if math.isfinite(finite_float(item.get("delta_int")))), key=lambda item: finite_float(item.get("delta_int")))
            best_step = int(best_row["step"])
        final_barrier_row = sorted(barrier_group, key=lambda item: finite_float(item.get("epoch_float")))[-1] if barrier_group else None
        summary_rows.append(
            {
                "ratio": ratio,
                "comparison": base.get("comparison", "same"),
                "comparison_label": comparison_label,
                "main_lr": base.get("main_lr", np.nan),
                "alt_lr": base.get("alt_lr", np.nan),
                "mean_delta_05": safe_mean([item.get("delta_05") for item in barrier_group]),
                "mean_delta_int": safe_mean([item.get("delta_int") for item in barrier_group]),
                "min_delta_int": min(delta_int_values) if delta_int_values else np.nan,
                "best_step_by_delta_int": best_step,
                "mean_retain_barrier": safe_mean([item.get("retain_barrier") for item in barrier_group]),
                "mean_forget_barrier": safe_mean([item.get("forget_barrier") for item in barrier_group]),
                "mean_val_barrier": safe_mean([item.get("val_barrier") for item in barrier_group]),
                "mean_test_barrier": safe_mean([item.get("test_barrier") for item in barrier_group]),
                "final_alpha_star_refined": finite_float(final_barrier_row.get("alpha_star_refined")) if final_barrier_row else np.nan,
                "final_delta_int": finite_float(final_barrier_row.get("delta_int")) if final_barrier_row else np.nan,
                "mean_merge_best_ua": safe_mean([item.get("merge_best_ua") for item in wide_group]),
                "mean_merge_best_dr_loss": safe_mean([item.get("merge_best_dr_loss") for item in wide_group]),
                "mean_merge_best_df_loss": safe_mean([item.get("merge_best_df_loss") for item in wide_group]),
                "mean_merge_best_test_loss": safe_mean([item.get("merge_best_test_loss") for item in wide_group]),
                "mean_merge_best_test_acc": safe_mean([item.get("merge_best_test_acc") for item in wide_group]),
                "mean_merge_05_ua": safe_mean([item.get("merge_05_ua") for item in wide_group]),
                "mean_merge_05_dr_loss": safe_mean([item.get("merge_05_dr_loss") for item in wide_group]),
                "mean_merge_05_df_loss": safe_mean([item.get("merge_05_df_loss") for item in wide_group]),
                "mean_merge_05_test_loss": safe_mean([item.get("merge_05_test_loss") for item in wide_group]),
                "mean_merge_05_test_acc": safe_mean([item.get("merge_05_test_acc") for item in wide_group]),
            }
        )
    summary_rows.sort(key=lambda item: (int(item["ratio"]), COMPARISON_ORDER.get(item.get("comparison", "same"), 99), item.get("comparison_label", "")))
    return summary_rows


def main():
    args = build_parser().parse_args()
    metrics = parse_csv_list(args.metrics)
    html_metrics = parse_csv_list(args.html_metrics)
    plot_series = parse_csv_list(args.plot_series)

    runs_root = os.path.abspath(args.runs_root)
    retrain_root = os.path.abspath(args.retrain_root) if args.retrain_root else runs_root
    summary_dir = os.path.abspath(args.summary_dir or default_summary_dir(runs_root))
    output_prefix = args.output_prefix or default_output_prefix(runs_root)
    os.makedirs(summary_dir, exist_ok=True)

    prepared_specs = []
    performance_long_rows = []
    performance_wide_rows = []
    performance_row_rows = []
    barrier_rows = []
    path_curve_rows = []
    seen_merge_center = False

    if args.prepend_runs_root:
        prepend_runs_root = os.path.abspath(args.prepend_runs_root)
        prepend_retrain_root = os.path.abspath(args.prepend_retrain_root) if args.prepend_retrain_root else retrain_root
        prep = collect_dataset(prepend_runs_root, prepend_retrain_root, args.prepend_ratio_run_specs, metrics, args.prepend_epoch_offset)
        prepared_specs.extend(prep[0])
        performance_long_rows.extend(prep[1])
        performance_wide_rows.extend(prep[2])
        performance_row_rows.extend(prep[3])
        barrier_rows.extend(prep[4])
        path_curve_rows.extend(prep[5])
        seen_merge_center = seen_merge_center or prep[6]

    cur = collect_dataset(runs_root, retrain_root, args.ratio_run_specs, metrics, args.epoch_offset)
    prepared_specs.extend(cur[0])
    performance_long_rows.extend(cur[1])
    performance_wide_rows.extend(cur[2])
    performance_row_rows.extend(cur[3])
    barrier_rows.extend(cur[4])
    path_curve_rows.extend(cur[5])
    seen_merge_center = seen_merge_center or cur[6]

    performance_long_rows.sort(key=lambda item: (int(item["ratio"]), COMPARISON_ORDER.get(item["comparison"], 99), finite_float(item["epoch_float"]), series_sort_key(item["series_type"]), int(item["step"])))
    performance_wide_rows.sort(key=lambda item: (int(item["ratio"]), COMPARISON_ORDER.get(item["comparison"], 99), finite_float(item["epoch_float"]), int(item["step"])))
    performance_row_rows.sort(key=lambda item: (int(item["ratio"]), COMPARISON_ORDER.get(item["comparison"], 99), finite_float(item["epoch_float"]), series_sort_key(item["series_type"]), item["metric_name"], int(item["step"])))
    barrier_rows.sort(key=lambda item: (int(item["ratio"]), COMPARISON_ORDER.get(item["comparison"], 99), finite_float(item["epoch_float"]), int(item["step"])))
    path_curve_rows.sort(key=lambda item: (int(item["ratio"]), COMPARISON_ORDER.get(item["comparison"], 99), finite_float(item["epoch_float"]), float(item["alpha"])))

    base_fields = ["ratio", "comparison", "comparison_label", "main_lr", "alt_lr", "lr_gap", "step", "epoch_float"]
    long_fieldnames = base_fields + ["series_type", "series_label", "alpha"]
    for metric_name in metrics:
        long_fieldnames.extend([metric_name, f"gap_vs_retrain_{metric_name}", f"retrain_{metric_name}"])

    wide_fieldnames = base_fields + ["merge_best_alpha", "merge_05_alpha"]
    if seen_merge_center:
        wide_fieldnames.append("merge_center_alpha")
    wide_series = ["main", "alt", "avg", "merge_best", "merge_05"] + (["merge_center"] if seen_merge_center else [])
    for metric_name in metrics:
        for series_type in wide_series:
            wide_fieldnames.append(f"{series_type}_{metric_name}")
        wide_fieldnames.append(f"retrain_{metric_name}")
        for series_type in wide_series:
            wide_fieldnames.append(f"gap_{series_type}_vs_retrain_{metric_name}")

    row_fieldnames = base_fields + ["series_type", "series_label", "alpha", "metric_name", "metric_value", "retrain_value", "gap_vs_retrain"]
    barrier_fieldnames = base_fields + ["alpha_star_coarse", "alpha_star_refined", "delta_05", "delta_int", "retain_barrier", "forget_barrier", "val_barrier", "test_barrier"]
    path_curve_fieldnames = ["ratio", "comparison", "comparison_label", "main_lr", "alt_lr", "step", "epoch_float", "alpha", "ua", "dr_loss", "val_loss", "j_score", "feasible"]

    performance_long_path = os.path.join(summary_dir, f"{output_prefix}_epoch_performance_vs_retrain_long.csv")
    performance_wide_path = os.path.join(summary_dir, f"{output_prefix}_epoch_performance_vs_retrain.csv")
    performance_rows_path = os.path.join(summary_dir, f"{output_prefix}_epoch_performance_vs_retrain_rows.csv")
    barrier_csv_path = os.path.join(summary_dir, f"{output_prefix}_barrier_by_step.csv")
    path_curve_csv_path = os.path.join(summary_dir, f"{output_prefix}_path_curves_by_step.csv")
    pair_summary_path = os.path.join(summary_dir, f"{output_prefix}_pair_summary.csv")

    write_csv(performance_long_path, performance_long_rows, long_fieldnames)
    print(f"Wrote {performance_long_path}")
    write_csv(performance_wide_path, performance_wide_rows, wide_fieldnames)
    print(f"Wrote {performance_wide_path}")
    write_csv(performance_rows_path, performance_row_rows, row_fieldnames)
    print(f"Wrote {performance_rows_path}")
    write_csv(barrier_csv_path, barrier_rows, barrier_fieldnames)
    print(f"Wrote {barrier_csv_path}")
    write_csv(path_curve_csv_path, path_curve_rows, path_curve_fieldnames)
    print(f"Wrote {path_curve_csv_path}")
    write_csv(
        pair_summary_path,
        build_pair_summary(barrier_rows, performance_wide_rows),
        [
            "ratio", "comparison", "comparison_label", "main_lr", "alt_lr", "mean_delta_05", "mean_delta_int", "min_delta_int",
            "best_step_by_delta_int", "mean_retain_barrier", "mean_forget_barrier", "mean_val_barrier", "mean_test_barrier",
            "final_alpha_star_refined", "final_delta_int", "mean_merge_best_ua", "mean_merge_best_dr_loss", "mean_merge_best_df_loss",
            "mean_merge_best_test_loss", "mean_merge_best_test_acc", "mean_merge_05_ua", "mean_merge_05_dr_loss", "mean_merge_05_df_loss",
            "mean_merge_05_test_loss", "mean_merge_05_test_acc",
        ],
    )
    print(f"Wrote {pair_summary_path}")

    for ratio in sorted({int(spec["ratio"]) for spec in prepared_specs}):
        path_curve_path = os.path.join(summary_dir, f"{output_prefix}_path_curves_by_step_ratio{ratio}.png")
        if plot_path_curves_png(path_curve_rows, ratio, path_curve_path):
            print(f"Wrote {path_curve_path}")
        barrier_path = os.path.join(summary_dir, f"{output_prefix}_barrier_by_step_ratio{ratio}.png")
        if plot_barrier_png(barrier_rows, ratio, barrier_path):
            print(f"Wrote {barrier_path}")

    for metric_name in html_metrics:
        fig = build_metric_html(performance_long_rows, metric_name, plot_series)
        if fig is None:
            continue
        html_path = os.path.join(summary_dir, f"{output_prefix}_{metric_name}_by_ratio.html")
        fig.write_html(html_path, include_plotlyjs=True, full_html=True)
        print(f"Wrote {html_path}")


if __name__ == "__main__":
    main()
