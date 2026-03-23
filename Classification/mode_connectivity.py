import datetime as dt
import math
import os

import numpy as np
import torch
import torch.nn as nn

import arg_parser
import experiment_helpers as experiment
import utils


RUN_MANIFEST_FIELDS = [
    "exp_id",
    "ratio",
    "pair_id",
    "connectivity_mode",
    "selection_mode",
    "forget_seed",
    "unlearn_seed_a",
    "unlearn_seed_b",
    "control_models",
    "forget_index_hash",
    "base_ckpt_hash",
    "optimizer",
    "lr_schedule",
    "batch_size",
    "total_steps",
    "save_every_steps",
    "alpha_grid",
    "beta",
    "delta",
    "simplex_weight_step",
    "abs_balance_a",
    "abs_balance_b",
    "notes",
    "timestamp",
]

CHECKPOINT_MANIFEST_FIELDS = [
    "exp_id",
    "ratio",
    "pair_id",
    "model_id",
    "step",
    "epoch_float",
    "ckpt_path",
    "ckpt_hash",
]

ENDPOINT_METRICS_FIELDS = [
    "exp_id",
    "ratio",
    "pair_id",
    "connectivity_mode",
    "step",
    "epoch_float",
    "model_id",
    "alpha",
    "ua",
    "dr_loss",
    "df_loss",
    "val_loss",
    "test_loss",
    "dr_acc",
    "df_acc",
    "val_acc",
    "test_acc",
]

PATH_SCAN_FIELDS = [
    "exp_id",
    "ratio",
    "pair_id",
    "connectivity_mode",
    "step",
    "epoch_float",
    "curve_id",
    "control",
    "weights",
    "alpha",
    "is_endpoint",
    "is_interior",
    "ua",
    "dr_loss",
    "val_loss",
    "ua_threshold",
    "feasible",
    "dr_loss_norm",
    "val_loss_norm",
    "j_score",
    "center_distance",
    "scan_stage",
]

STEP_SUMMARY_FIELDS = [
    "exp_id",
    "ratio",
    "pair_id",
    "connectivity_mode",
    "step",
    "epoch_float",
    "ua_a",
    "ua_b",
    "tau_ua",
    "feasible_count",
    "best_curve_id",
    "best_control",
    "best_weights",
    "alpha_star_coarse",
    "alpha_star_refined",
    "j_endpoint_best",
    "j_merge_05",
    "j_merge_best",
    "delta_05",
    "delta_int",
    "endpoint_best_model",
    "retain_barrier",
    "forget_barrier",
    "val_barrier",
    "test_barrier",
]

FULL_EVAL_FIELDS = [
    "exp_id",
    "ratio",
    "pair_id",
    "connectivity_mode",
    "step",
    "epoch_float",
    "candidate_type",
    "curve_id",
    "control",
    "weights",
    "alpha",
    "ua",
    "dr_loss",
    "df_loss",
    "val_loss",
    "test_loss",
    "dr_acc",
    "df_acc",
    "val_acc",
    "test_acc",
    "j_score",
    "feasible",
]

SIMPLEX_BARRIER_FIELDS = [
    "exp_id",
    "ratio",
    "pair_id",
    "step",
    "epoch_float",
    "vertex",
    "soup_weights",
    "retain_barrier",
    "forget_barrier",
    "val_barrier",
    "test_barrier",
]


class ProtocolLogger:
    def __init__(self, path):
        self._handle = open(path, "w", encoding="utf-8")

    def close(self):
        self._handle.close()

    def emit(self, tag, **kwargs):
        parts = [f"[{tag}]"]
        for key, value in kwargs.items():
            parts.append(f"{key}={self._format(value)}")
        line = " ".join(parts)
        print(line)
        self._handle.write(line + "\n")
        self._handle.flush()

    @staticmethod
    def _format(value):
        if isinstance(value, bool):
            return "1" if value else "0"
        if isinstance(value, (np.bool_,)):
            return "1" if bool(value) else "0"
        if value is None:
            return "nan"
        if isinstance(value, (float, np.floating)):
            value = float(value)
            if math.isnan(value):
                return "nan"
            return f"{value:.6f}".rstrip("0").rstrip(".")
        return str(value)


def build_parser():
    parser = arg_parser.build_parser()
    parser.description = "Step-wise mode connectivity scan between aligned unlearning runs"
    parser.add_argument("--run_a_dir", required=True, type=str, help="directory for checkpoint trajectory A")
    parser.add_argument("--run_b_dir", required=True, type=str, help="directory for checkpoint trajectory B")
    parser.add_argument(
        "--control_run_dirs",
        default="",
        type=str,
        help="optional comma-separated control/extra vertex dirs as label=path,label=path",
    )
    parser.add_argument("--output_dir", required=True, type=str, help="directory where connectivity artifacts are written")
    parser.add_argument("--ratio", required=True, type=int, help="forget ratio for this group")
    parser.add_argument("--exp_id", default="conn_mode_v1", type=str, help="experiment id written to CSV/logs")
    parser.add_argument("--pair_id", default=None, type=str, help="pair identifier, defaults to A<seedA>_B<seedB>")
    parser.add_argument("--label_a", default="A", type=str, help="label for endpoint A")
    parser.add_argument("--label_b", default="B", type=str, help="label for endpoint B")
    parser.add_argument(
        "--connectivity_mode",
        default="linear",
        choices=["linear", "quadratic_bezier", "simplex"],
        help="connectivity family to evaluate at each shared checkpoint step",
    )
    parser.add_argument("--curve_steps", default=None, type=str, help="optional comma-separated global steps to scan")
    parser.add_argument("--alpha_grid", default="0.0:0.1:1.0", type=str, help="alpha/t grid as start:step:end or comma-separated values")
    parser.add_argument("--beta", default=0.3, type=float, help="weight for normalized val loss in the combo objective")
    parser.add_argument("--delta", default=0.0, type=float, help="UA threshold slack subtracted from the reference UA")
    parser.add_argument("--refine_radius", default=0.1, type=float, help="radius around the coarse best parameter for refined scanning")
    parser.add_argument("--refine_step", default=0.02, type=float, help="step size for the refined parameter grid")
    parser.add_argument("--simplex_weight_step", default=0.1, type=float, help="simplex grid step size")
    parser.add_argument("--abs_balance_a", default=1.0, type=float, help="coefficient a in |a * DR - b * DF|")
    parser.add_argument("--abs_balance_b", default=1.0, type=float, help="coefficient b in |a * DR - b * DF|")
    parser.add_argument("--eps", default=1e-12, type=float, help="epsilon for normalization denominators")
    parser.add_argument(
        "--selection_mode",
        default="dr_min_then_val",
        choices=["dr_min_then_val", "abs_balance", "normalized_combo"],
        help="candidate selection rule for feasible interior candidates",
    )
    parser.add_argument(
        "--allow_same_unlearn_seed",
        action="store_true",
        help="allow trajectories to share the same unlearn_seed",
    )
    parser.add_argument(
        "--allow_mismatched_base_checkpoint_hash",
        action="store_true",
        help="allow trajectories to originate from different parent checkpoints",
    )
    parser.add_argument("--notes", default="", type=str, help="free-form note stored in run_manifest.csv")
    return parser


def parse_alpha_grid(spec, include_defaults=True):
    if spec is None or str(spec).strip() == "":
        spec = "0.0:0.1:1.0"
    spec = str(spec).strip()
    if ":" in spec and "," not in spec:
        parts = [part.strip() for part in spec.split(":")]
        if len(parts) != 3:
            raise ValueError("Alpha grid range spec must be start:step:end")
        start, step, end = map(float, parts)
        if step <= 0:
            raise ValueError("Alpha grid step must be positive")
        values = np.arange(start, end + step / 2.0, step)
    else:
        values = [float(token.strip()) for token in spec.split(",") if token.strip()]

    parsed = []
    for value in values:
        rounded = round(float(value), 10)
        if rounded < 0.0 or rounded > 1.0:
            raise ValueError("Alpha values must lie in [0, 1]")
        parsed.append(rounded)

    if include_defaults:
        parsed.extend([0.0, 0.5, 1.0])

    return sorted(set(parsed))


def build_refined_grid(center, radius, step):
    if center is None or not math.isfinite(center):
        return []
    lower = max(0.0, center - radius)
    upper = min(1.0, center + radius)
    values = np.arange(lower, upper + step / 2.0, step)
    parsed = [round(float(value), 10) for value in values]
    parsed.append(round(float(center), 10))
    return sorted(set(parsed))


def alpha_to_string(alpha):
    return f"{float(alpha):.6f}".rstrip("0").rstrip(".")


def canonical_alpha_grid(alphas):
    return ",".join(alpha_to_string(alpha) for alpha in alphas)


def finite_float(value):
    if value is None:
        return np.nan
    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan
    return value


def normalize_value(value, min_value, max_value, eps):
    value = finite_float(value)
    min_value = finite_float(min_value)
    max_value = finite_float(max_value)
    if not math.isfinite(value) or not math.isfinite(min_value) or not math.isfinite(max_value):
        return np.nan
    return (value - min_value) / (max_value - min_value + eps)


def parse_named_run_dirs(spec):
    if spec is None or str(spec).strip() == "":
        return []

    entries = []
    used_labels = set()
    fallback_index = 1
    for raw_token in str(spec).split(","):
        token = raw_token.strip()
        if not token:
            continue
        if "=" in token:
            label, path = token.split("=", 1)
            label = label.strip()
            path = path.strip()
        else:
            label = f"C{fallback_index}"
            path = token
            fallback_index += 1
        if not label or not path:
            raise ValueError(f"Invalid control run entry: '{token}'")
        if label in used_labels:
            raise ValueError(f"Duplicate control label '{label}'")
        used_labels.add(label)
        entries.append((label, path))
    return entries


def normalize_weights(weights):
    total = sum(float(weight) for weight in weights)
    if total <= 0.0:
        raise ValueError("Weights must sum to a positive value")
    return [float(weight) / total for weight in weights]


def merge_state_dicts(state_dicts, weights):
    if len(state_dicts) != len(weights):
        raise ValueError("state_dicts and weights must have the same length")
    if not state_dicts:
        raise ValueError("At least one state_dict is required")

    weights = normalize_weights(weights)
    first_keys = list(state_dicts[0].keys())
    first_key_set = set(first_keys)
    for state_dict in state_dicts[1:]:
        if set(state_dict.keys()) != first_key_set:
            raise ValueError("All state_dicts must share the same keys")

    max_idx = int(np.argmax(weights))
    merged = {}
    for key in first_keys:
        values = [state_dict[key] for state_dict in state_dicts]
        reference = values[0]
        if torch.is_tensor(reference):
            if any(not torch.is_tensor(value) for value in values):
                raise TypeError(f"Mixed tensor/non-tensor values for '{key}'")
            if any(value.shape != reference.shape for value in values):
                raise ValueError(f"Shape mismatch for '{key}'")
            if torch.is_floating_point(reference) or torch.is_complex(reference):
                target_dtype = reference.dtype
                acc_dtype = torch.complex64 if torch.is_complex(reference) else torch.float32
                accumulator = torch.zeros_like(reference, dtype=acc_dtype)
                for weight, value in zip(weights, values):
                    accumulator = accumulator + value.detach().to(acc_dtype) * float(weight)
                merged[key] = accumulator.to(target_dtype)
            else:
                merged[key] = values[max_idx].detach().clone()
        else:
            merged[key] = values[max_idx]
    return merged


def linear_state_dict(state_a, state_b, alpha):
    return merge_state_dicts([state_a, state_b], [1.0 - float(alpha), float(alpha)])


def quadratic_bezier_state_dict(state_a, state_c, state_b, alpha):
    alpha = float(alpha)
    return merge_state_dicts(
        [state_a, state_c, state_b],
        [(1.0 - alpha) ** 2, 2.0 * (1.0 - alpha) * alpha, alpha ** 2],
    )


def simplex_state_dict(state_dicts, weights):
    return merge_state_dicts(state_dicts, weights)


def format_weights(labels, weights):
    parts = []
    for label, weight in zip(labels, weights):
        parts.append(f"{label}={float(weight):.6f}".rstrip("0").rstrip("."))
    return ";".join(parts)


def simplex_center_distance(weights):
    if not weights:
        return np.nan
    target = 1.0 / float(len(weights))
    return float(np.linalg.norm(np.asarray(weights, dtype=float) - target))


def is_simplex_vertex(weights, atol=1e-8):
    positive = [weight for weight in weights if float(weight) > atol]
    total = sum(float(weight) for weight in weights)
    return len(positive) == 1 and math.isclose(total, 1.0, abs_tol=atol)


def _integer_compositions(total, parts):
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in _integer_compositions(total - first, parts - 1):
            yield (first,) + rest


def enumerate_simplex_weights(num_vertices, step):
    if num_vertices < 2:
        raise ValueError("num_vertices must be at least 2")
    inv = round(1.0 / float(step))
    if not math.isclose(inv * float(step), 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("simplex_weight_step must evenly divide 1.0")
    total = int(inv)
    return [[count / total for count in composition] for composition in _integer_compositions(total, num_vertices)]


def compute_barrier(curve_rows, key):
    for row in curve_rows:
        if not experiment.is_finite_number(row[key]):
            return np.nan
    start_value = float(curve_rows[0][key])
    end_value = float(curve_rows[-1][key])
    barrier = []
    for row in curve_rows:
        alpha = float(row["alpha"])
        linear_value = (1.0 - alpha) * start_value + alpha * end_value
        barrier.append(float(row[key]) - linear_value)
    return max(barrier)


def build_metric_row(
    alpha,
    metrics,
    curve_id="",
    control="",
    weights="",
    center_distance=np.nan,
    is_endpoint=0,
    is_interior=1,
    weight_values=None,
):
    return {
        "curve_id": curve_id,
        "control": control,
        "weights": weights,
        "weight_values": list(weight_values) if weight_values is not None else None,
        "center_distance": finite_float(center_distance),
        "alpha": finite_float(alpha),
        "is_endpoint": int(is_endpoint),
        "is_interior": int(is_interior),
        "ua": finite_float(metrics["ua"]),
        "dr_loss": finite_float(metrics["loss_retain"]),
        "df_loss": finite_float(metrics["loss_forget"]),
        "val_loss": finite_float(metrics["loss_val"]),
        "test_loss": finite_float(metrics["loss_test"]),
        "dr_acc": finite_float(metrics["acc_retain"]),
        "df_acc": finite_float(metrics["acc_forget"]),
        "val_acc": finite_float(metrics["acc_val"]),
        "test_acc": finite_float(metrics["acc_test"]),
        "valid_run": bool(metrics["valid_run"]),
    }


def evaluate_state(
    model,
    state_dict,
    eval_context,
    alpha=np.nan,
    curve_id="",
    control="",
    weights="",
    center_distance=np.nan,
    is_endpoint=0,
    is_interior=1,
    weight_values=None,
):
    model.load_state_dict(state_dict, strict=False)
    metrics = experiment.evaluate_model(
        model,
        eval_context["data_loaders"],
        eval_context["forget_dataset"],
        eval_context["retain_dataset"],
        eval_context["criterion"],
        eval_context["args"],
        compute_mia=False,
    )
    return build_metric_row(
        alpha=alpha,
        metrics=metrics,
        curve_id=curve_id,
        control=control,
        weights=weights,
        center_distance=center_distance,
        is_endpoint=is_endpoint,
        is_interior=is_interior,
        weight_values=weight_values,
    )


def compute_norm_stats(rows):
    stats = {}
    for key in ["dr_loss", "val_loss"]:
        values = [float(row[key]) for row in rows if experiment.is_finite_number(row[key])]
        if not values:
            stats[f"{key}_min"] = np.nan
            stats[f"{key}_max"] = np.nan
            continue
        stats[f"{key}_min"] = min(values)
        stats[f"{key}_max"] = max(values)
    return stats


def annotate_scan_rows(rows, epoch_float, tau_ua, norm_stats, selection_mode, beta, eps, abs_a, abs_b, stage):
    annotated = []
    for row in rows:
        dr_norm = normalize_value(row["dr_loss"], norm_stats["dr_loss_min"], norm_stats["dr_loss_max"], eps)
        val_norm = normalize_value(row["val_loss"], norm_stats["val_loss_min"], norm_stats["val_loss_max"], eps)
        feasible = 0
        if int(row.get("is_interior", 0)) == 1 and experiment.is_finite_number(row["ua"]):
            feasible = int(float(row["ua"]) >= float(tau_ua))

        j_score = np.nan
        if selection_mode == "normalized_combo":
            if experiment.is_finite_number(dr_norm) and experiment.is_finite_number(val_norm):
                j_score = float(dr_norm) + float(beta) * float(val_norm)
        elif selection_mode == "abs_balance":
            if experiment.is_finite_number(row["dr_loss"]) and experiment.is_finite_number(row["df_loss"]):
                j_score = abs(float(abs_a) * float(row["dr_loss"]) - float(abs_b) * float(row["df_loss"]))
        elif experiment.is_finite_number(row["dr_loss"]):
            j_score = float(row["dr_loss"])

        annotated.append(
            {
                **row,
                "epoch_float": epoch_float,
                "ua_threshold": tau_ua,
                "feasible": feasible,
                "dr_loss_norm": dr_norm,
                "val_loss_norm": val_norm,
                "j_score": j_score,
                "scan_stage": stage,
            }
        )
    return annotated


def row_center_distance(row):
    center_distance = finite_float(row.get("center_distance"))
    if math.isfinite(center_distance):
        return center_distance
    alpha = finite_float(row.get("alpha"))
    if math.isfinite(alpha):
        return abs(alpha - 0.5)
    return float("inf")


def stable_float(value, default=float("inf")):
    value = finite_float(value)
    if math.isfinite(value):
        return float(value)
    return default


def candidate_sort_key(row, selection_mode):
    center = row_center_distance(row)
    curve_id = str(row.get("curve_id", ""))
    weights = str(row.get("weights", ""))
    alpha = stable_float(row.get("alpha"))
    if selection_mode == "normalized_combo":
        return (
            stable_float(row.get("j_score")),
            stable_float(row.get("val_loss")),
            center,
            curve_id,
            weights,
            alpha,
        )
    if selection_mode == "abs_balance":
        return (
            stable_float(row.get("j_score")),
            stable_float(row.get("dr_loss")),
            stable_float(row.get("val_loss")),
            center,
            curve_id,
            weights,
            alpha,
        )
    return (
        stable_float(row.get("dr_loss")),
        stable_float(row.get("val_loss")),
        center,
        curve_id,
        weights,
        alpha,
    )


def choose_best_candidate(rows, selection_mode, feasible_only=True):
    candidates = []
    for row in rows:
        if feasible_only and int(row.get("feasible", 0)) != 1:
            continue
        if not experiment.is_finite_number(row.get("j_score")):
            continue
        candidates.append(row)
    if not candidates:
        return None
    return min(candidates, key=lambda row: candidate_sort_key(row, selection_mode))


def choose_best_labeled_row(rows_by_label, selection_mode):
    candidates = [
        (label, row)
        for label, row in rows_by_label.items()
        if experiment.is_finite_number(row.get("j_score"))
    ]
    if not candidates:
        return None, None
    label, row = min(candidates, key=lambda item: candidate_sort_key(item[1], selection_mode))
    return label, row


def find_best_row_by_alpha(rows, alpha, selection_mode):
    candidates = []
    for row in rows:
        row_alpha = finite_float(row.get("alpha"))
        if math.isfinite(row_alpha) and math.isclose(row_alpha, float(alpha), abs_tol=1e-8):
            if experiment.is_finite_number(row.get("j_score")):
                candidates.append(row)
    if not candidates:
        return None
    return min(candidates, key=lambda row: candidate_sort_key(row, selection_mode))


def find_simplex_center_row(rows, selection_mode):
    candidates = [row for row in rows if int(row.get("is_interior", 0)) == 1 and experiment.is_finite_number(row.get("j_score"))]
    if not candidates:
        return None
    best_distance = min(row_center_distance(row) for row in candidates)
    close = [row for row in candidates if abs(row_center_distance(row) - best_distance) <= 1e-12]
    return min(close, key=lambda row: candidate_sort_key(row, selection_mode))


def filter_rows_by_curve(rows, curve_id):
    return [row for row in rows if str(row.get("curve_id", "")) == str(curve_id)]


def step_checkpoint_map(run_dir, args):
    checkpoint_dir = experiment.resolve_checkpoint_dir(args, base_dir=run_dir)
    if not os.path.isdir(checkpoint_dir):
        raise ValueError(f"Checkpoint directory not found: {checkpoint_dir}")
    mapping = {}
    for name in os.listdir(checkpoint_dir):
        path = os.path.join(checkpoint_dir, name)
        step = experiment.parse_step_from_path(path)
        if step is None:
            continue
        mapping[step] = os.path.abspath(path)
    if not mapping:
        raise ValueError(f"No step_*.pth.tar checkpoints found under {checkpoint_dir}")
    return mapping


def resolve_epoch_float(step, meta_list):
    candidates = []
    steps_per_epoch = []
    for meta in meta_list:
        value = meta.get("epoch_float")
        if experiment.is_finite_number(value):
            candidates.append(float(value))
        spe = meta.get("steps_per_epoch")
        if experiment.is_finite_number(spe) and float(spe) > 0:
            steps_per_epoch.append(float(spe))
    if candidates and max(candidates) - min(candidates) > 1e-6:
        raise ValueError(f"Mismatched epoch_float for step {step}: {candidates}")
    if candidates:
        return candidates[0]
    if steps_per_epoch and max(steps_per_epoch) - min(steps_per_epoch) > 1e-6:
        raise ValueError(f"Mismatched steps_per_epoch for step {step}: {steps_per_epoch}")
    if steps_per_epoch:
        return float(step) / float(steps_per_epoch[0])
    return np.nan


def require_equal(name, value_a, value_b):
    if value_a is None or value_b is None:
        return
    if value_a != value_b:
        raise ValueError(f"{name} mismatch: {value_a} vs {value_b}")


def prefer_non_null(*values):
    for value in values:
        if value not in (None, ""):
            return value
    return None


def validate_pair_metadata(meta_a, meta_b, allow_same_unlearn_seed=False, allow_mismatched_base_checkpoint_hash=False):
    require_equal("forget_seed", meta_a.get("forget_seed"), meta_b.get("forget_seed"))
    require_equal("forget_index_hash", meta_a.get("forget_index_hash"), meta_b.get("forget_index_hash"))
    if not allow_mismatched_base_checkpoint_hash:
        require_equal(
            "base_checkpoint_hash",
            meta_a.get("base_checkpoint_hash"),
            meta_b.get("base_checkpoint_hash"),
        )
    require_equal("optimizer", meta_a.get("optimizer"), meta_b.get("optimizer"))
    require_equal("lr_schedule", meta_a.get("lr_schedule"), meta_b.get("lr_schedule"))
    require_equal("batch_size", meta_a.get("batch_size"), meta_b.get("batch_size"))
    require_equal("steps_per_epoch", meta_a.get("steps_per_epoch"), meta_b.get("steps_per_epoch"))
    seed_a = meta_a.get("unlearn_seed")
    seed_b = meta_b.get("unlearn_seed")
    if (
        not allow_same_unlearn_seed
        and seed_a is not None
        and seed_b is not None
        and seed_a == seed_b
    ):
        raise ValueError("Trajectories share the same unlearn_seed; pass --allow_same_unlearn_seed to override")


def validate_all_run_metadata(meta_by_label, args):
    labels = list(meta_by_label.keys())
    anchor_label = labels[0]
    anchor_meta = meta_by_label[anchor_label]
    for label in labels[1:]:
        validate_pair_metadata(
            anchor_meta,
            meta_by_label[label],
            allow_same_unlearn_seed=args.allow_same_unlearn_seed,
            allow_mismatched_base_checkpoint_hash=args.allow_mismatched_base_checkpoint_hash,
        )


def build_run_manifest_row(args, pair_id, meta_a, meta_b, alpha_grid, common_steps, control_labels):
    return {
        "exp_id": args.exp_id,
        "ratio": args.ratio,
        "pair_id": pair_id,
        "connectivity_mode": args.connectivity_mode,
        "selection_mode": args.selection_mode,
        "forget_seed": prefer_non_null(meta_a.get("forget_seed"), meta_b.get("forget_seed"), args.forget_seed),
        "unlearn_seed_a": meta_a.get("unlearn_seed"),
        "unlearn_seed_b": meta_b.get("unlearn_seed"),
        "control_models": ",".join(control_labels),
        "forget_index_hash": prefer_non_null(
            meta_a.get("forget_index_hash"),
            meta_b.get("forget_index_hash"),
            experiment.maybe_file_sha256(getattr(args, "forget_index_path", None)),
        ),
        "base_ckpt_hash": prefer_non_null(
            meta_a.get("base_checkpoint_hash"),
            meta_b.get("base_checkpoint_hash"),
            experiment.maybe_file_sha256(getattr(args, "model_path", None)),
        ),
        "optimizer": prefer_non_null(meta_a.get("optimizer"), meta_b.get("optimizer")),
        "lr_schedule": prefer_non_null(meta_a.get("lr_schedule"), meta_b.get("lr_schedule")),
        "batch_size": prefer_non_null(meta_a.get("batch_size"), meta_b.get("batch_size"), args.batch_size),
        "total_steps": prefer_non_null(meta_a.get("total_unlearn_steps"), meta_b.get("total_unlearn_steps"), common_steps[-1]),
        "save_every_steps": prefer_non_null(
            meta_a.get("checkpoint_every_steps"),
            meta_b.get("checkpoint_every_steps"),
            getattr(args, "checkpoint_every_steps", None),
        ),
        "alpha_grid": canonical_alpha_grid(alpha_grid),
        "beta": args.beta,
        "delta": args.delta,
        "simplex_weight_step": args.simplex_weight_step,
        "abs_balance_a": args.abs_balance_a,
        "abs_balance_b": args.abs_balance_b,
        "notes": args.notes,
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
    }


def metrics_to_endpoint_row(args, pair_id, step, epoch_float, model_id, row):
    return {
        "exp_id": args.exp_id,
        "ratio": args.ratio,
        "pair_id": pair_id,
        "connectivity_mode": args.connectivity_mode,
        "step": step,
        "epoch_float": epoch_float,
        "model_id": model_id,
        "alpha": row["alpha"],
        "ua": row["ua"],
        "dr_loss": row["dr_loss"],
        "df_loss": row["df_loss"],
        "val_loss": row["val_loss"],
        "test_loss": row["test_loss"],
        "dr_acc": row["dr_acc"],
        "df_acc": row["df_acc"],
        "val_acc": row["val_acc"],
        "test_acc": row["test_acc"],
    }


def scan_row_to_csv(args, pair_id, step, row):
    return {
        "exp_id": args.exp_id,
        "ratio": args.ratio,
        "pair_id": pair_id,
        "connectivity_mode": args.connectivity_mode,
        "step": step,
        "epoch_float": row["epoch_float"],
        "curve_id": row["curve_id"],
        "control": row["control"],
        "weights": row["weights"],
        "alpha": row["alpha"],
        "is_endpoint": row["is_endpoint"],
        "is_interior": row["is_interior"],
        "ua": row["ua"],
        "dr_loss": row["dr_loss"],
        "val_loss": row["val_loss"],
        "ua_threshold": row["ua_threshold"],
        "feasible": row["feasible"],
        "dr_loss_norm": row["dr_loss_norm"],
        "val_loss_norm": row["val_loss_norm"],
        "j_score": row["j_score"],
        "center_distance": row["center_distance"],
        "scan_stage": row["scan_stage"],
    }


def full_eval_row(args, pair_id, step, epoch_float, candidate_type, row):
    if row is None:
        return {
            "exp_id": args.exp_id,
            "ratio": args.ratio,
            "pair_id": pair_id,
            "connectivity_mode": args.connectivity_mode,
            "step": step,
            "epoch_float": epoch_float,
            "candidate_type": candidate_type,
            "curve_id": "",
            "control": "",
            "weights": "",
            "alpha": np.nan,
            "ua": np.nan,
            "dr_loss": np.nan,
            "df_loss": np.nan,
            "val_loss": np.nan,
            "test_loss": np.nan,
            "dr_acc": np.nan,
            "df_acc": np.nan,
            "val_acc": np.nan,
            "test_acc": np.nan,
            "j_score": np.nan,
            "feasible": 0,
        }
    return {
        "exp_id": args.exp_id,
        "ratio": args.ratio,
        "pair_id": pair_id,
        "connectivity_mode": args.connectivity_mode,
        "step": step,
        "epoch_float": epoch_float,
        "candidate_type": candidate_type,
        "curve_id": row["curve_id"],
        "control": row["control"],
        "weights": row["weights"],
        "alpha": row["alpha"],
        "ua": row["ua"],
        "dr_loss": row["dr_loss"],
        "df_loss": row["df_loss"],
        "val_loss": row["val_loss"],
        "test_loss": row["test_loss"],
        "dr_acc": row["dr_acc"],
        "df_acc": row["df_acc"],
        "val_acc": row["val_acc"],
        "test_acc": row["test_acc"],
        "j_score": row["j_score"],
        "feasible": row["feasible"],
    }


def evaluate_linear_point(model, state_a, state_b, alpha, eval_context, curve_id):
    alpha = round(float(alpha), 10)
    if math.isclose(alpha, 0.0, abs_tol=1e-10):
        state = state_a
    elif math.isclose(alpha, 1.0, abs_tol=1e-10):
        state = state_b
    else:
        state = linear_state_dict(state_a, state_b, alpha)
    is_endpoint = int(math.isclose(alpha, 0.0, abs_tol=1e-10) or math.isclose(alpha, 1.0, abs_tol=1e-10))
    return evaluate_state(
        model,
        state,
        eval_context,
        alpha=alpha,
        curve_id=curve_id,
        control="",
        weights="",
        center_distance=abs(alpha - 0.5),
        is_endpoint=is_endpoint,
        is_interior=1 - is_endpoint,
    )


def evaluate_quadratic_point(model, state_a, state_b, state_c, alpha, eval_context, control_label):
    alpha = round(float(alpha), 10)
    if math.isclose(alpha, 0.0, abs_tol=1e-10):
        state = state_a
    elif math.isclose(alpha, 1.0, abs_tol=1e-10):
        state = state_b
    else:
        state = quadratic_bezier_state_dict(state_a, state_c, state_b, alpha)
    is_endpoint = int(math.isclose(alpha, 0.0, abs_tol=1e-10) or math.isclose(alpha, 1.0, abs_tol=1e-10))
    return evaluate_state(
        model,
        state,
        eval_context,
        alpha=alpha,
        curve_id=control_label,
        control=control_label,
        weights="",
        center_distance=abs(alpha - 0.5),
        is_endpoint=is_endpoint,
        is_interior=1 - is_endpoint,
    )


def evaluate_simplex_point(model, vertex_states, labels, weights, eval_context):
    soup_state = simplex_state_dict(vertex_states, weights)
    vertex = is_simplex_vertex(weights)
    return evaluate_state(
        model,
        soup_state,
        eval_context,
        alpha=np.nan,
        curve_id="simplex",
        control="",
        weights=format_weights(labels, weights),
        center_distance=simplex_center_distance(weights),
        is_endpoint=int(vertex),
        is_interior=int(not vertex),
        weight_values=weights,
    )


def build_mode_result(
    args,
    label_a,
    label_b,
    vertex_rows,
    coarse_rows,
    refined_rows,
    best_row,
    center_row,
    endpoint_best_label,
    endpoint_best_row,
    tau_ua,
    feasible_count,
    alpha_star_coarse,
    alpha_star_refined,
    best_curve_id,
    best_control,
    best_weights,
    barriers,
    candidate_rows,
    simplex_barrier_rows,
):
    j_endpoint_best = endpoint_best_row["j_score"] if endpoint_best_row is not None else np.nan
    j_merge_05 = center_row["j_score"] if center_row is not None else np.nan
    j_merge_best = best_row["j_score"] if best_row is not None else np.nan
    delta_05 = (
        float(j_merge_05) - float(j_endpoint_best)
        if experiment.is_finite_number(j_merge_05) and experiment.is_finite_number(j_endpoint_best)
        else np.nan
    )
    delta_int = (
        float(j_merge_best) - float(j_endpoint_best)
        if experiment.is_finite_number(j_merge_best) and experiment.is_finite_number(j_endpoint_best)
        else np.nan
    )
    return {
        "vertex_rows": vertex_rows,
        "coarse_rows": coarse_rows,
        "refined_rows": refined_rows,
        "best_row": best_row,
        "center_row": center_row,
        "candidate_rows": candidate_rows,
        "simplex_barrier_rows": simplex_barrier_rows,
        "step_summary": {
            "exp_id": args.exp_id,
            "ratio": args.ratio,
            "pair_id": None,
            "connectivity_mode": args.connectivity_mode,
            "step": None,
            "epoch_float": None,
            "ua_a": vertex_rows[label_a]["ua"],
            "ua_b": vertex_rows[label_b]["ua"],
            "tau_ua": tau_ua,
            "feasible_count": feasible_count,
            "best_curve_id": best_curve_id,
            "best_control": best_control,
            "best_weights": best_weights,
            "alpha_star_coarse": alpha_star_coarse,
            "alpha_star_refined": alpha_star_refined,
            "j_endpoint_best": j_endpoint_best,
            "j_merge_05": j_merge_05,
            "j_merge_best": j_merge_best,
            "delta_05": delta_05,
            "delta_int": delta_int,
            "endpoint_best_model": endpoint_best_label,
            "retain_barrier": barriers["retain_barrier"],
            "forget_barrier": barriers["forget_barrier"],
            "val_barrier": barriers["val_barrier"],
            "test_barrier": barriers["test_barrier"],
        },
    }


def scan_linear_mode(model, state_a, state_b, args, eval_context, label_a, label_b):
    curve_id = f"{label_a}-{label_b}"
    coarse_raw = [evaluate_linear_point(model, state_a, state_b, alpha, eval_context, curve_id) for alpha in args.alpha_grid_values]
    tau_ua = min(float(coarse_raw[0]["ua"]), float(coarse_raw[-1]["ua"])) - float(args.delta)
    norm_stats = compute_norm_stats(coarse_raw)
    coarse_rows = annotate_scan_rows(
        coarse_raw,
        np.nan,
        tau_ua,
        norm_stats,
        args.selection_mode,
        args.beta,
        args.eps,
        args.abs_balance_a,
        args.abs_balance_b,
        "coarse",
    )
    endpoint_a = find_best_row_by_alpha(coarse_rows, 0.0, args.selection_mode)
    endpoint_b = find_best_row_by_alpha(coarse_rows, 1.0, args.selection_mode)
    vertex_rows = {label_a: endpoint_a, label_b: endpoint_b}

    feasible_coarse = [row for row in coarse_rows if int(row["is_interior"]) == 1 and int(row["feasible"]) == 1]
    coarse_best = choose_best_candidate(feasible_coarse, args.selection_mode)

    refined_rows = []
    if coarse_best is not None:
        refined_grid = build_refined_grid(float(coarse_best["alpha"]), args.refine_radius, args.refine_step)
        refined_raw = [evaluate_linear_point(model, state_a, state_b, alpha, eval_context, curve_id) for alpha in refined_grid]
        refined_rows = annotate_scan_rows(
            refined_raw,
            np.nan,
            tau_ua,
            norm_stats,
            args.selection_mode,
            args.beta,
            args.eps,
            args.abs_balance_a,
            args.abs_balance_b,
            "refined",
        )

    feasible_refined = [row for row in refined_rows if int(row["is_interior"]) == 1 and int(row["feasible"]) == 1]
    refined_best = choose_best_candidate(feasible_refined, args.selection_mode)
    best_row = refined_best or coarse_best
    center_row = find_best_row_by_alpha(coarse_rows, 0.5, args.selection_mode)
    endpoint_best_label, endpoint_best_row = choose_best_labeled_row(vertex_rows, args.selection_mode)

    barrier_curve = sorted(coarse_rows, key=lambda row: float(row["alpha"]))
    result = build_mode_result(
        args,
        label_a,
        label_b,
        vertex_rows,
        coarse_rows,
        refined_rows,
        best_row,
        center_row,
        endpoint_best_label,
        endpoint_best_row,
        tau_ua,
        len(feasible_coarse),
        coarse_best["alpha"] if coarse_best is not None else np.nan,
        best_row["alpha"] if best_row is not None else np.nan,
        curve_id,
        "",
        "",
        {
            "retain_barrier": compute_barrier(barrier_curve, "dr_loss"),
            "forget_barrier": compute_barrier(barrier_curve, "df_loss"),
            "val_barrier": compute_barrier(barrier_curve, "val_loss"),
            "test_barrier": compute_barrier(barrier_curve, "test_loss"),
        },
        [
            ("endpoint_a", endpoint_a),
            ("endpoint_b", endpoint_b),
            ("merge_05", center_row),
            ("merge_best", best_row),
        ],
        [],
    )
    return result


def scan_quadratic_bezier_mode(model, state_a, state_b, control_states, args, eval_context, label_a, label_b):
    endpoint_a_raw = evaluate_state(
        model,
        state_a,
        eval_context,
        alpha=0.0,
        curve_id=label_a,
        center_distance=0.5,
        is_endpoint=1,
        is_interior=0,
    )
    endpoint_b_raw = evaluate_state(
        model,
        state_b,
        eval_context,
        alpha=1.0,
        curve_id=label_b,
        center_distance=0.5,
        is_endpoint=1,
        is_interior=0,
    )
    tau_ua = min(float(endpoint_a_raw["ua"]), float(endpoint_b_raw["ua"])) - float(args.delta)

    coarse_raw = []
    for control_label, control_state in control_states.items():
        for alpha in args.alpha_grid_values:
            coarse_raw.append(evaluate_quadratic_point(model, state_a, state_b, control_state, alpha, eval_context, control_label))

    norm_stats = compute_norm_stats(coarse_raw)
    coarse_rows = annotate_scan_rows(
        coarse_raw,
        np.nan,
        tau_ua,
        norm_stats,
        args.selection_mode,
        args.beta,
        args.eps,
        args.abs_balance_a,
        args.abs_balance_b,
        "coarse",
    )

    vertex_rows = {
        label_a: annotate_scan_rows(
            [endpoint_a_raw],
            np.nan,
            tau_ua,
            norm_stats,
            args.selection_mode,
            args.beta,
            args.eps,
            args.abs_balance_a,
            args.abs_balance_b,
            "vertex",
        )[0],
        label_b: annotate_scan_rows(
            [endpoint_b_raw],
            np.nan,
            tau_ua,
            norm_stats,
            args.selection_mode,
            args.beta,
            args.eps,
            args.abs_balance_a,
            args.abs_balance_b,
            "vertex",
        )[0],
    }
    for control_label, control_state in control_states.items():
        control_raw = evaluate_state(
            model,
            control_state,
            eval_context,
            alpha=np.nan,
            curve_id=control_label,
            control=control_label,
            center_distance=np.nan,
            is_endpoint=1,
            is_interior=0,
        )
        vertex_rows[control_label] = annotate_scan_rows(
            [control_raw],
            np.nan,
            tau_ua,
            norm_stats,
            args.selection_mode,
            args.beta,
            args.eps,
            args.abs_balance_a,
            args.abs_balance_b,
            "vertex",
        )[0]

    feasible_coarse = [row for row in coarse_rows if int(row["is_interior"]) == 1 and int(row["feasible"]) == 1]
    coarse_best = choose_best_candidate(feasible_coarse, args.selection_mode)

    refined_rows = []
    if coarse_best is not None:
        best_control = coarse_best["control"]
        refined_grid = build_refined_grid(float(coarse_best["alpha"]), args.refine_radius, args.refine_step)
        refined_raw = [
            evaluate_quadratic_point(model, state_a, state_b, control_states[best_control], alpha, eval_context, best_control)
            for alpha in refined_grid
        ]
        refined_rows = annotate_scan_rows(
            refined_raw,
            np.nan,
            tau_ua,
            norm_stats,
            args.selection_mode,
            args.beta,
            args.eps,
            args.abs_balance_a,
            args.abs_balance_b,
            "refined",
        )

    feasible_refined = [row for row in refined_rows if int(row["is_interior"]) == 1 and int(row["feasible"]) == 1]
    refined_best = choose_best_candidate(feasible_refined, args.selection_mode)
    best_row = refined_best or coarse_best
    center_row = find_best_row_by_alpha(coarse_rows, 0.5, args.selection_mode)
    endpoint_best_label, endpoint_best_row = choose_best_labeled_row(
        {label_a: vertex_rows[label_a], label_b: vertex_rows[label_b]},
        args.selection_mode,
    )

    control_labels = list(control_states.keys())
    active_curve = best_row["control"] if best_row is not None else center_row["control"] if center_row is not None else control_labels[0]
    barrier_curve = sorted(filter_rows_by_curve(coarse_rows, active_curve), key=lambda row: float(row["alpha"]))
    result = build_mode_result(
        args,
        label_a,
        label_b,
        vertex_rows,
        coarse_rows,
        refined_rows,
        best_row,
        center_row,
        endpoint_best_label,
        endpoint_best_row,
        tau_ua,
        len(feasible_coarse),
        coarse_best["alpha"] if coarse_best is not None else np.nan,
        best_row["alpha"] if best_row is not None else np.nan,
        active_curve,
        best_row["control"] if best_row is not None else active_curve,
        "",
        {
            "retain_barrier": compute_barrier(barrier_curve, "dr_loss") if barrier_curve else np.nan,
            "forget_barrier": compute_barrier(barrier_curve, "df_loss") if barrier_curve else np.nan,
            "val_barrier": compute_barrier(barrier_curve, "val_loss") if barrier_curve else np.nan,
            "test_barrier": compute_barrier(barrier_curve, "test_loss") if barrier_curve else np.nan,
        },
        [
            ("endpoint_a", vertex_rows[label_a]),
            ("endpoint_b", vertex_rows[label_b]),
            ("merge_05", center_row),
            ("merge_best", best_row),
        ],
        [],
    )
    return result


def scan_simplex_mode(model, vertex_states_by_label, args, eval_context, label_a, label_b):
    labels = list(vertex_states_by_label.keys())
    vertex_states = [vertex_states_by_label[label] for label in labels]

    vertex_raw = {}
    for index, label in enumerate(labels):
        weights = [0.0] * len(labels)
        weights[index] = 1.0
        vertex_raw[label] = evaluate_state(
            model,
            vertex_states_by_label[label],
            eval_context,
            alpha=np.nan,
            curve_id="simplex",
            control="",
            weights=format_weights(labels, weights),
            center_distance=simplex_center_distance(weights),
            is_endpoint=1,
            is_interior=0,
            weight_values=weights,
        )

    tau_ua = min(float(row["ua"]) for row in vertex_raw.values()) - float(args.delta)
    coarse_raw = [evaluate_simplex_point(model, vertex_states, labels, weights, eval_context) for weights in enumerate_simplex_weights(len(labels), args.simplex_weight_step)]
    norm_stats = compute_norm_stats(coarse_raw)
    coarse_rows = annotate_scan_rows(
        coarse_raw,
        np.nan,
        tau_ua,
        norm_stats,
        args.selection_mode,
        args.beta,
        args.eps,
        args.abs_balance_a,
        args.abs_balance_b,
        "coarse",
    )
    vertex_rows = {}
    for label, row in vertex_raw.items():
        vertex_rows[label] = annotate_scan_rows(
            [row],
            np.nan,
            tau_ua,
            norm_stats,
            args.selection_mode,
            args.beta,
            args.eps,
            args.abs_balance_a,
            args.abs_balance_b,
            "vertex",
        )[0]

    feasible_coarse = [row for row in coarse_rows if int(row["is_interior"]) == 1 and int(row["feasible"]) == 1]
    best_row = choose_best_candidate(feasible_coarse, args.selection_mode)
    center_row = find_simplex_center_row(coarse_rows, args.selection_mode)
    endpoint_best_label, endpoint_best_row = choose_best_labeled_row(vertex_rows, args.selection_mode)

    simplex_barrier_rows = []
    barriers = {
        "retain_barrier": np.nan,
        "forget_barrier": np.nan,
        "val_barrier": np.nan,
        "test_barrier": np.nan,
    }
    if best_row is not None:
        soup_state = simplex_state_dict(vertex_states, best_row["weight_values"])
        barrier_values = {"retain_barrier": [], "forget_barrier": [], "val_barrier": [], "test_barrier": []}
        for label, vertex_state in vertex_states_by_label.items():
            curve_rows = [evaluate_linear_point(model, vertex_state, soup_state, alpha, eval_context, f"{label}->soup") for alpha in args.alpha_grid_values]
            barrier_curve = sorted(curve_rows, key=lambda row: float(row["alpha"]))
            record = {
                "exp_id": args.exp_id,
                "ratio": args.ratio,
                "pair_id": None,
                "step": None,
                "epoch_float": None,
                "vertex": label,
                "soup_weights": best_row["weights"],
                "retain_barrier": compute_barrier(barrier_curve, "dr_loss"),
                "forget_barrier": compute_barrier(barrier_curve, "df_loss"),
                "val_barrier": compute_barrier(barrier_curve, "val_loss"),
                "test_barrier": compute_barrier(barrier_curve, "test_loss"),
            }
            simplex_barrier_rows.append(record)
            for key in barrier_values:
                barrier_values[key].append(record[key])
        for key, values in barrier_values.items():
            barriers[key] = max(values) if values else np.nan

    candidate_rows = [(f"vertex_{label.lower()}", row) for label, row in vertex_rows.items()]
    candidate_rows.extend([
        ("merge_center", center_row),
        ("merge_best", best_row),
    ])

    result = build_mode_result(
        args,
        label_a,
        label_b,
        vertex_rows,
        coarse_rows,
        [],
        best_row,
        center_row,
        endpoint_best_label,
        endpoint_best_row,
        tau_ua,
        len(feasible_coarse),
        np.nan,
        np.nan,
        "simplex",
        "",
        best_row["weights"] if best_row is not None else "",
        barriers,
        candidate_rows,
        simplex_barrier_rows,
    )
    return result


def main():
    args = build_parser().parse_args()
    args = experiment.prepare_experiment_args(args)
    args.alpha_grid_values = parse_alpha_grid(args.alpha_grid)

    if args.connectivity_mode in {"quadratic_bezier", "simplex"} and not str(args.control_run_dirs).strip():
        raise ValueError("--control_run_dirs is required for quadratic_bezier and simplex modes")
    if args.connectivity_mode != "simplex" and (args.refine_radius <= 0 or args.refine_step <= 0):
        raise ValueError("Refinement radius and step must be positive")

    control_specs = parse_named_run_dirs(args.control_run_dirs)
    used_labels = {args.label_a, args.label_b}
    for label, _path in control_specs:
        if label in used_labels:
            raise ValueError(f"Control label '{label}' conflicts with endpoint labels")
        used_labels.add(label)

    relevant_runs = [(args.label_a, args.run_a_dir), (args.label_b, args.run_b_dir)]
    if args.connectivity_mode in {"quadratic_bezier", "simplex"}:
        relevant_runs.extend(control_specs)

    step_maps = {label: step_checkpoint_map(run_dir, args) for label, run_dir in relevant_runs}
    requested_steps = experiment.parse_step_spec(args.curve_steps)
    common_steps = sorted(set.intersection(*(set(mapping.keys()) for mapping in step_maps.values())))
    if requested_steps:
        common_steps = [step for step in common_steps if step in requested_steps]
    if not common_steps:
        raise ValueError("No common step checkpoints found for the requested group")

    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))
        device = torch.device(f"cuda:{int(args.gpu)}")
    else:
        device = torch.device("cpu")

    utils.setup_seed(args.forget_seed)
    model, _train_loader_full, val_loader, test_loader, marked_loader = utils.setup_model_dataset(args)
    model.to(device)
    data_loaders, forget_dataset, retain_dataset = experiment.build_eval_data_loaders(marked_loader, val_loader, test_loader, args)
    criterion = nn.CrossEntropyLoss()
    eval_context = {
        "args": args,
        "criterion": criterion,
        "data_loaders": data_loaders,
        "forget_dataset": forget_dataset,
        "retain_dataset": retain_dataset,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    logger = ProtocolLogger(os.path.join(args.output_dir, "protocol.log"))
    try:
        first_meta_by_label = {}
        for label, _run_dir in relevant_runs:
            checkpoint = experiment.load_checkpoint_file(step_maps[label][common_steps[0]], device)
            _state, meta = experiment.extract_state_dict(checkpoint)
            first_meta_by_label[label] = meta
        validate_all_run_metadata(first_meta_by_label, args)

        meta_a = first_meta_by_label[args.label_a]
        meta_b = first_meta_by_label[args.label_b]
        pair_id = args.pair_id or f"A{meta_a.get('unlearn_seed')}_B{meta_b.get('unlearn_seed')}"
        run_manifest_row = build_run_manifest_row(
            args,
            pair_id,
            meta_a,
            meta_b,
            args.alpha_grid_values,
            common_steps,
            [label for label, _path in control_specs],
        )
        experiment.write_csv(os.path.join(args.output_dir, "run_manifest.csv"), [run_manifest_row], RUN_MANIFEST_FIELDS)
        logger.emit(
            "META",
            exp_id=args.exp_id,
            ratio=args.ratio,
            pair_id=pair_id,
            mode=args.connectivity_mode,
            selection_mode=args.selection_mode,
            controls=run_manifest_row["control_models"],
            forget_seed=run_manifest_row["forget_seed"],
            unlearn_seed_a=run_manifest_row["unlearn_seed_a"],
            unlearn_seed_b=run_manifest_row["unlearn_seed_b"],
            forget_index_hash=run_manifest_row["forget_index_hash"],
            base_ckpt_hash=run_manifest_row["base_ckpt_hash"],
            optimizer=run_manifest_row["optimizer"],
            lr_schedule=run_manifest_row["lr_schedule"],
            batch_size=run_manifest_row["batch_size"],
            eval_shuffle=0,
        )

        checkpoint_manifest_rows = []
        endpoint_rows = []
        path_scan_rows = []
        step_summary_rows = []
        full_eval_rows = []
        simplex_barrier_rows = []

        for step in common_steps:
            state_by_label = {}
            meta_by_label = {}
            for label, _run_dir in relevant_runs:
                checkpoint = experiment.load_checkpoint_file(step_maps[label][step], device)
                state, meta = experiment.extract_state_dict(checkpoint)
                state_by_label[label] = state
                meta_by_label[label] = meta
            validate_all_run_metadata(meta_by_label, args)
            epoch_float = resolve_epoch_float(step, list(meta_by_label.values()))

            for label, _run_dir in relevant_runs:
                checkpoint_manifest_rows.append(
                    {
                        "exp_id": args.exp_id,
                        "ratio": args.ratio,
                        "pair_id": pair_id,
                        "model_id": label,
                        "step": step,
                        "epoch_float": epoch_float,
                        "ckpt_path": step_maps[label][step],
                        "ckpt_hash": experiment.file_sha256(step_maps[label][step]),
                    }
                )
                logger.emit(
                    "CKPT",
                    ratio=args.ratio,
                    pair_id=pair_id,
                    model=label,
                    step=step,
                    epoch=epoch_float,
                    ckpt=step_maps[label][step],
                )

            if args.connectivity_mode == "linear":
                result = scan_linear_mode(
                    model,
                    state_by_label[args.label_a],
                    state_by_label[args.label_b],
                    args,
                    eval_context,
                    args.label_a,
                    args.label_b,
                )
            elif args.connectivity_mode == "quadratic_bezier":
                control_states = {label: state_by_label[label] for label, _path in control_specs}
                result = scan_quadratic_bezier_mode(
                    model,
                    state_by_label[args.label_a],
                    state_by_label[args.label_b],
                    control_states,
                    args,
                    eval_context,
                    args.label_a,
                    args.label_b,
                )
            else:
                simplex_states = {args.label_a: state_by_label[args.label_a]}
                for label, _path in control_specs:
                    simplex_states[label] = state_by_label[label]
                simplex_states[args.label_b] = state_by_label[args.label_b]
                result = scan_simplex_mode(
                    model,
                    simplex_states,
                    args,
                    eval_context,
                    args.label_a,
                    args.label_b,
                )

            logger.emit(
                "ENDPOINT",
                ratio=args.ratio,
                step=step,
                ua_A=result["vertex_rows"][args.label_a]["ua"],
                ua_B=result["vertex_rows"][args.label_b]["ua"],
                dr_A=result["vertex_rows"][args.label_a]["dr_loss"],
                dr_B=result["vertex_rows"][args.label_b]["dr_loss"],
                val_A=result["vertex_rows"][args.label_a]["val_loss"],
                val_B=result["vertex_rows"][args.label_b]["val_loss"],
                tau_ua=result["step_summary"]["tau_ua"],
            )

            for row in result["coarse_rows"]:
                row["epoch_float"] = epoch_float
                logger.emit(
                    "SCAN",
                    ratio=args.ratio,
                    step=step,
                    stage=row["scan_stage"],
                    curve=row["curve_id"],
                    alpha=row["alpha"],
                    weights=row["weights"],
                    ua=row["ua"],
                    dr=row["dr_loss"],
                    val=row["val_loss"],
                    feasible=row["feasible"],
                )
            for row in result["refined_rows"]:
                row["epoch_float"] = epoch_float
                logger.emit(
                    "SCAN",
                    ratio=args.ratio,
                    step=step,
                    stage=row["scan_stage"],
                    curve=row["curve_id"],
                    alpha=row["alpha"],
                    weights=row["weights"],
                    ua=row["ua"],
                    dr=row["dr_loss"],
                    val=row["val_loss"],
                    feasible=row["feasible"],
                )

            for label, row in result["vertex_rows"].items():
                endpoint_rows.append(metrics_to_endpoint_row(args, pair_id, step, epoch_float, label, row))

            path_scan_rows.extend(scan_row_to_csv(args, pair_id, step, row) for row in result["coarse_rows"])
            path_scan_rows.extend(scan_row_to_csv(args, pair_id, step, row) for row in result["refined_rows"])

            step_summary = dict(result["step_summary"])
            step_summary["pair_id"] = pair_id
            step_summary["step"] = step
            step_summary["epoch_float"] = epoch_float
            step_summary_rows.append(step_summary)
            logger.emit(
                "SELECT",
                ratio=args.ratio,
                step=step,
                mode=args.connectivity_mode,
                feasible_count=step_summary["feasible_count"],
                best_curve=step_summary["best_curve_id"],
                best_control=step_summary["best_control"],
                best_weights=step_summary["best_weights"],
                alpha_star=step_summary["alpha_star_coarse"],
                alpha_star_refined=step_summary["alpha_star_refined"],
                J_end=step_summary["j_endpoint_best"],
                J_05=step_summary["j_merge_05"],
                J_star=step_summary["j_merge_best"],
                delta_05=step_summary["delta_05"],
                delta_int=step_summary["delta_int"],
            )
            logger.emit(
                "BARRIER",
                ratio=args.ratio,
                step=step,
                retain_barrier=step_summary["retain_barrier"],
                forget_barrier=step_summary["forget_barrier"],
                val_barrier=step_summary["val_barrier"],
                test_barrier=step_summary["test_barrier"],
            )

            for candidate_type, row in result["candidate_rows"]:
                full_row = full_eval_row(args, pair_id, step, epoch_float, candidate_type, row)
                full_eval_rows.append(full_row)
                logger.emit(
                    "FULL",
                    ratio=args.ratio,
                    step=step,
                    candidate=candidate_type,
                    alpha=full_row["alpha"],
                    curve=full_row["curve_id"],
                    weights=full_row["weights"],
                    ua=full_row["ua"],
                    dr_loss=full_row["dr_loss"],
                    df_loss=full_row["df_loss"],
                    val_loss=full_row["val_loss"],
                    test_loss=full_row["test_loss"],
                    dr_acc=full_row["dr_acc"],
                    df_acc=full_row["df_acc"],
                    val_acc=full_row["val_acc"],
                    test_acc=full_row["test_acc"],
                )

            for record in result["simplex_barrier_rows"]:
                record["pair_id"] = pair_id
                record["step"] = step
                record["epoch_float"] = epoch_float
                simplex_barrier_rows.append(record)

        experiment.write_csv(os.path.join(args.output_dir, "checkpoint_manifest.csv"), checkpoint_manifest_rows, CHECKPOINT_MANIFEST_FIELDS)
        experiment.write_csv(os.path.join(args.output_dir, "endpoint_metrics.csv"), endpoint_rows, ENDPOINT_METRICS_FIELDS)
        experiment.write_csv(os.path.join(args.output_dir, "path_scan.csv"), path_scan_rows, PATH_SCAN_FIELDS)
        experiment.write_csv(os.path.join(args.output_dir, "step_summary.csv"), step_summary_rows, STEP_SUMMARY_FIELDS)
        experiment.write_csv(os.path.join(args.output_dir, "full_eval.csv"), full_eval_rows, FULL_EVAL_FIELDS)
        experiment.write_csv(os.path.join(args.output_dir, "simplex_barriers.csv"), simplex_barrier_rows, SIMPLEX_BARRIER_FIELDS)
        print(f"Wrote mode connectivity artifacts to {os.path.abspath(args.output_dir)}")
    finally:
        logger.close()


if __name__ == "__main__":
    main()
