import datetime as dt
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


RUN_MANIFEST_FIELDS = [
    "exp_id",
    "ratio",
    "pair_id",
    "forget_seed",
    "unlearn_seed_a",
    "unlearn_seed_b",
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
    "step",
    "epoch_float",
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
    "scan_stage",
]

STEP_SUMMARY_FIELDS = [
    "exp_id",
    "ratio",
    "pair_id",
    "step",
    "epoch_float",
    "ua_a",
    "ua_b",
    "tau_ua",
    "feasible_count",
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
    "step",
    "epoch_float",
    "candidate_type",
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
    parser.description = "Step-wise linear connectivity scan between two unlearning runs"
    parser.add_argument("--run_a_dir", required=True, type=str, help="directory for checkpoint trajectory A")
    parser.add_argument("--run_b_dir", required=True, type=str, help="directory for checkpoint trajectory B")
    parser.add_argument("--output_dir", required=True, type=str, help="directory where step connectivity artifacts are written")
    parser.add_argument("--ratio", required=True, type=int, help="forget ratio for this pair")
    parser.add_argument("--exp_id", default="conn_step_v1", type=str, help="experiment id written to CSV/logs")
    parser.add_argument("--pair_id", default=None, type=str, help="pair identifier, defaults to A<seedA>_B<seedB>")
    parser.add_argument("--label_a", default="A", type=str, help="label for endpoint A")
    parser.add_argument("--label_b", default="B", type=str, help="label for endpoint B")
    parser.add_argument("--curve_steps", default=None, type=str, help="optional comma-separated global steps to scan")
    parser.add_argument("--alpha_grid", default="0.0:0.1:1.0", type=str, help="alpha grid as start:step:end or comma-separated values")
    parser.add_argument("--beta", default=0.3, type=float, help="weight for normalized val loss in the selection objective")
    parser.add_argument("--delta", default=0.0, type=float, help="UA threshold slack subtracted from min(endpoint UA)")
    parser.add_argument("--refine_radius", default=0.1, type=float, help="radius around the coarse best alpha for refined scanning")
    parser.add_argument("--refine_step", default=0.02, type=float, help="step size for the refined alpha grid")
    parser.add_argument("--eps", default=1e-12, type=float, help="epsilon for normalization denominators")
    parser.add_argument(
        "--selection_mode",
        default="normalized_combo",
        choices=["normalized_combo", "dr_min_then_val"],
        help="candidate selection rule for feasible interior alphas",
    )
    parser.add_argument(
        "--allow_same_unlearn_seed",
        action="store_true",
        help="allow A/B to share the same unlearn_seed; useful for cross-hyperparameter comparisons",
    )
    parser.add_argument("--skip_plots", action="store_true", help="skip PNG plot generation")
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


def build_metric_row(alpha, metrics):
    return {
        "alpha": float(alpha),
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


def evaluate_state(model, state_dict, eval_context):
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
    return build_metric_row(eval_context["alpha"], metrics)


def evaluate_alpha(model, state_a, state_b, alpha, eval_context):
    alpha = round(float(alpha), 10)
    eval_context["alpha"] = alpha
    if math.isclose(alpha, 0.0, abs_tol=1e-10):
        return evaluate_state(model, state_a, eval_context)
    if math.isclose(alpha, 1.0, abs_tol=1e-10):
        return evaluate_state(model, state_b, eval_context)
    interpolated_state = experiment.interpolate_state_dict(state_a, state_b, alpha)
    return evaluate_state(model, interpolated_state, eval_context)


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


def annotate_scan_rows(rows, epoch_float, tau_ua, norm_stats, selection_mode, beta, eps, stage):
    annotated = []
    for row in rows:
        alpha = float(row["alpha"])
        dr_norm = normalize_value(
            row["dr_loss"],
            norm_stats["dr_loss_min"],
            norm_stats["dr_loss_max"],
            eps,
        )
        val_norm = normalize_value(
            row["val_loss"],
            norm_stats["val_loss_min"],
            norm_stats["val_loss_max"],
            eps,
        )
        feasible = 0
        if alpha > 0.0 and alpha < 1.0 and experiment.is_finite_number(row["ua"]):
            feasible = int(float(row["ua"]) >= float(tau_ua))
        j_score = np.nan
        if selection_mode == "normalized_combo":
            if experiment.is_finite_number(dr_norm) and experiment.is_finite_number(val_norm):
                j_score = float(dr_norm) + beta * float(val_norm)
        elif experiment.is_finite_number(row["dr_loss"]):
            j_score = float(row["dr_loss"])
        annotated.append(
            {
                **row,
                "epoch_float": epoch_float,
                "is_endpoint": int(alpha in (0.0, 1.0)),
                "is_interior": int(alpha > 0.0 and alpha < 1.0),
                "ua_threshold": tau_ua,
                "feasible": feasible,
                "dr_loss_norm": dr_norm,
                "val_loss_norm": val_norm,
                "j_score": j_score,
                "scan_stage": stage,
            }
        )
    return annotated


def candidate_sort_key(row, selection_mode):
    alpha = float(row["alpha"])
    if selection_mode == "normalized_combo":
        return (float(row["j_score"]), float(row["val_loss"]), abs(alpha - 0.5), alpha)
    return (float(row["dr_loss"]), float(row["val_loss"]), abs(alpha - 0.5), alpha)


def choose_best_candidate(rows, selection_mode):
    candidates = [
        row
        for row in rows
        if row["feasible"] == 1 and experiment.is_finite_number(row["j_score"])
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda row: candidate_sort_key(row, selection_mode))


def choose_best_endpoint(rows, selection_mode):
    candidates = [row for row in rows if experiment.is_finite_number(row["j_score"])]
    if not candidates:
        return None
    return min(candidates, key=lambda row: candidate_sort_key(row, selection_mode))


def find_row_by_alpha(rows, alpha):
    for row in rows:
        if math.isclose(float(row["alpha"]), float(alpha), abs_tol=1e-8):
            return row
    return None


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


def resolve_epoch_float(step, meta_a, meta_b):
    candidates = []
    for meta in (meta_a, meta_b):
        value = meta.get("epoch_float")
        if experiment.is_finite_number(value):
            candidates.append(float(value))
    if len(candidates) == 2 and abs(candidates[0] - candidates[1]) > 1e-6:
        raise ValueError(f"Mismatched epoch_float for step {step}: {candidates[0]} vs {candidates[1]}")
    if candidates:
        return candidates[0]
    steps_per_epoch = meta_a.get("steps_per_epoch") or meta_b.get("steps_per_epoch")
    if experiment.is_finite_number(steps_per_epoch) and float(steps_per_epoch) > 0:
        return float(step) / float(steps_per_epoch)
    return np.nan


def require_equal(name, value_a, value_b):
    if value_a is None or value_b is None:
        return
    if value_a != value_b:
        raise ValueError(f"{name} mismatch between A and B: {value_a} vs {value_b}")


def prefer_non_null(*values):
    for value in values:
        if value not in (None, ""):
            return value
    return None


def validate_pair_metadata(meta_a, meta_b, allow_same_unlearn_seed=False):
    require_equal("forget_seed", meta_a.get("forget_seed"), meta_b.get("forget_seed"))
    require_equal(
        "forget_index_hash", meta_a.get("forget_index_hash"), meta_b.get("forget_index_hash")
    )
    require_equal(
        "base_checkpoint_hash", meta_a.get("base_checkpoint_hash"), meta_b.get("base_checkpoint_hash")
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
        raise ValueError("A and B use the same unlearn_seed; the protocol expects different seeds")


def build_run_manifest_row(args, pair_id, meta_a, meta_b, alpha_grid, common_steps):
    return {
        "exp_id": args.exp_id,
        "ratio": args.ratio,
        "pair_id": pair_id,
        "forget_seed": prefer_non_null(meta_a.get("forget_seed"), meta_b.get("forget_seed"), args.forget_seed),
        "unlearn_seed_a": meta_a.get("unlearn_seed"),
        "unlearn_seed_b": meta_b.get("unlearn_seed"),
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
        "save_every_steps": prefer_non_null(meta_a.get("checkpoint_every_steps"), meta_b.get("checkpoint_every_steps"), getattr(args, "checkpoint_every_steps", None)),
        "alpha_grid": canonical_alpha_grid(alpha_grid),
        "beta": args.beta,
        "delta": args.delta,
        "notes": args.notes,
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
    }


def metrics_to_endpoint_row(args, pair_id, step, epoch_float, model_id, alpha, row):
    return {
        "exp_id": args.exp_id,
        "ratio": args.ratio,
        "pair_id": pair_id,
        "step": step,
        "epoch_float": epoch_float,
        "model_id": model_id,
        "alpha": alpha,
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
        "step": step,
        "epoch_float": row["epoch_float"],
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
        "scan_stage": row["scan_stage"],
    }


def full_eval_row(args, pair_id, step, epoch_float, candidate_type, row):
    if row is None:
        return {
            "exp_id": args.exp_id,
            "ratio": args.ratio,
            "pair_id": pair_id,
            "step": step,
            "epoch_float": epoch_float,
            "candidate_type": candidate_type,
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
        "step": step,
        "epoch_float": epoch_float,
        "candidate_type": candidate_type,
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


def plot_outputs(output_dir, step_summaries, full_eval_rows):
    steps = [float(row["step"]) for row in step_summaries]
    if not steps:
        return

    plt.figure(figsize=(9, 5))
    plt.plot(steps, [float(row["retain_barrier"]) for row in step_summaries], marker="o", label="retain")
    plt.plot(steps, [float(row["forget_barrier"]) for row in step_summaries], marker="o", label="forget")
    plt.plot(steps, [float(row["val_barrier"]) for row in step_summaries], marker="o", label="val")
    plt.plot(steps, [float(row["test_barrier"]) for row in step_summaries], marker="o", label="test")
    plt.xlabel("step")
    plt.ylabel("barrier")
    plt.title("Barrier vs Step")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "barrier_vs_step.png"))
    plt.close()

    plt.figure(figsize=(9, 4))
    plt.plot(steps, [finite_float(row["alpha_star_refined"]) for row in step_summaries], marker="o")
    plt.xlabel("step")
    plt.ylabel("alpha*")
    plt.title("Alpha Star vs Step")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "alpha_star_vs_step.png"))
    plt.close()

    plt.figure(figsize=(9, 4))
    plt.plot(steps, [finite_float(row["delta_05"]) for row in step_summaries], marker="o", label="delta_05")
    plt.plot(steps, [finite_float(row["delta_int"]) for row in step_summaries], marker="o", label="delta_int")
    plt.xlabel("step")
    plt.ylabel("gap")
    plt.title("Merge Gap vs Step")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "gap_vs_step.png"))
    plt.close()

    merge_best_rows = [row for row in full_eval_rows if row["candidate_type"] == "merge_best"]
    merge_steps = [float(row["step"]) for row in merge_best_rows]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    metric_specs = [
        ("ua", "UA"),
        ("dr_loss", "DR loss"),
        ("val_loss", "Val loss"),
        ("test_loss", "Test loss"),
    ]
    for axis, (key, title) in zip(axes.flat, metric_specs):
        axis.plot(merge_steps, [finite_float(row[key]) for row in merge_best_rows], marker="o")
        axis.set_title(title)
        axis.set_xlabel("step")
    fig.suptitle("Merge Best Metrics vs Step")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "merge_best_metrics_vs_step.png"))
    plt.close(fig)


def main():
    args = build_parser().parse_args()
    args = experiment.prepare_experiment_args(args)

    alpha_grid = parse_alpha_grid(args.alpha_grid)
    if args.refine_radius <= 0 or args.refine_step <= 0:
        raise ValueError("Refinement radius and step must be positive")

    step_map_a = step_checkpoint_map(args.run_a_dir, args)
    step_map_b = step_checkpoint_map(args.run_b_dir, args)
    requested_steps = experiment.parse_step_spec(args.curve_steps)
    common_steps = sorted(set(step_map_a.keys()) & set(step_map_b.keys()))
    if requested_steps:
        common_steps = [step for step in common_steps if step in requested_steps]
    if not common_steps:
        raise ValueError("No common step checkpoints found for the requested pair")

    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))
        device = torch.device(f"cuda:{int(args.gpu)}")
    else:
        device = torch.device("cpu")

    utils.setup_seed(args.forget_seed)
    model, _train_loader_full, val_loader, test_loader, marked_loader = utils.setup_model_dataset(args)
    model.to(device)
    data_loaders, forget_dataset, retain_dataset = experiment.build_eval_data_loaders(
        marked_loader, val_loader, test_loader, args
    )
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
        first_checkpoint_a = experiment.load_checkpoint_file(step_map_a[common_steps[0]], device)
        first_checkpoint_b = experiment.load_checkpoint_file(step_map_b[common_steps[0]], device)
        _, meta_a = experiment.extract_state_dict(first_checkpoint_a)
        _, meta_b = experiment.extract_state_dict(first_checkpoint_b)
        validate_pair_metadata(meta_a, meta_b, allow_same_unlearn_seed=args.allow_same_unlearn_seed)

        pair_id = args.pair_id or f"A{meta_a.get('unlearn_seed')}_B{meta_b.get('unlearn_seed')}"
        run_manifest_row = build_run_manifest_row(args, pair_id, meta_a, meta_b, alpha_grid, common_steps)
        experiment.write_csv(
            os.path.join(args.output_dir, "run_manifest.csv"),
            [run_manifest_row],
            RUN_MANIFEST_FIELDS,
        )
        logger.emit(
            "META",
            exp_id=args.exp_id,
            ratio=args.ratio,
            pair_id=pair_id,
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

        for step in common_steps:
            checkpoint_a = experiment.load_checkpoint_file(step_map_a[step], device)
            checkpoint_b = experiment.load_checkpoint_file(step_map_b[step], device)
            state_a, meta_a = experiment.extract_state_dict(checkpoint_a)
            state_b, meta_b = experiment.extract_state_dict(checkpoint_b)
            validate_pair_metadata(meta_a, meta_b, allow_same_unlearn_seed=args.allow_same_unlearn_seed)
            epoch_float = resolve_epoch_float(step, meta_a, meta_b)

            checkpoint_manifest_rows.append(
                {
                    "exp_id": args.exp_id,
                    "ratio": args.ratio,
                    "pair_id": pair_id,
                    "model_id": args.label_a,
                    "step": step,
                    "epoch_float": epoch_float,
                    "ckpt_path": step_map_a[step],
                    "ckpt_hash": experiment.file_sha256(step_map_a[step]),
                }
            )
            checkpoint_manifest_rows.append(
                {
                    "exp_id": args.exp_id,
                    "ratio": args.ratio,
                    "pair_id": pair_id,
                    "model_id": args.label_b,
                    "step": step,
                    "epoch_float": epoch_float,
                    "ckpt_path": step_map_b[step],
                    "ckpt_hash": experiment.file_sha256(step_map_b[step]),
                }
            )
            logger.emit(
                "CKPT",
                ratio=args.ratio,
                pair_id=pair_id,
                model=args.label_a,
                step=step,
                epoch=epoch_float,
                ckpt=step_map_a[step],
            )
            logger.emit(
                "CKPT",
                ratio=args.ratio,
                pair_id=pair_id,
                model=args.label_b,
                step=step,
                epoch=epoch_float,
                ckpt=step_map_b[step],
            )

            endpoint_a = evaluate_alpha(model, state_a, state_b, 0.0, eval_context)
            endpoint_b = evaluate_alpha(model, state_a, state_b, 1.0, eval_context)
            tau_ua = min(float(endpoint_a["ua"]), float(endpoint_b["ua"])) - float(args.delta)
            logger.emit(
                "ENDPOINT",
                ratio=args.ratio,
                step=step,
                ua_A=endpoint_a["ua"],
                ua_B=endpoint_b["ua"],
                dr_A=endpoint_a["dr_loss"],
                dr_B=endpoint_b["dr_loss"],
                val_A=endpoint_a["val_loss"],
                val_B=endpoint_b["val_loss"],
                tau_ua=tau_ua,
            )

            coarse_rows = []
            for alpha in alpha_grid:
                if math.isclose(alpha, 0.0, abs_tol=1e-10):
                    row = dict(endpoint_a)
                    row["alpha"] = 0.0
                elif math.isclose(alpha, 1.0, abs_tol=1e-10):
                    row = dict(endpoint_b)
                    row["alpha"] = 1.0
                else:
                    row = evaluate_alpha(model, state_a, state_b, alpha, eval_context)
                coarse_rows.append(row)

            norm_stats = compute_norm_stats(coarse_rows)
            coarse_rows = annotate_scan_rows(
                coarse_rows,
                epoch_float,
                tau_ua,
                norm_stats,
                args.selection_mode,
                args.beta,
                args.eps,
                "coarse",
            )
            for row in coarse_rows:
                logger.emit(
                    "SCAN",
                    ratio=args.ratio,
                    step=step,
                    stage=row["scan_stage"],
                    alpha=row["alpha"],
                    ua=row["ua"],
                    dr=row["dr_loss"],
                    val=row["val_loss"],
                    feasible=row["feasible"],
                )

            feasible_coarse = [row for row in coarse_rows if row["is_interior"] == 1 and row["feasible"] == 1]
            alpha_star_coarse_row = choose_best_candidate(feasible_coarse, args.selection_mode)
            refined_rows = []
            if alpha_star_coarse_row is not None:
                refined_grid = build_refined_grid(
                    float(alpha_star_coarse_row["alpha"]), args.refine_radius, args.refine_step
                )
                for alpha in refined_grid:
                    row = evaluate_alpha(model, state_a, state_b, alpha, eval_context)
                    refined_rows.append(row)
                refined_rows = annotate_scan_rows(
                    refined_rows,
                    epoch_float,
                    tau_ua,
                    norm_stats,
                    args.selection_mode,
                    args.beta,
                    args.eps,
                    "refined",
                )
                for row in refined_rows:
                    logger.emit(
                        "SCAN",
                        ratio=args.ratio,
                        step=step,
                        stage=row["scan_stage"],
                        alpha=row["alpha"],
                        ua=row["ua"],
                        dr=row["dr_loss"],
                        val=row["val_loss"],
                        feasible=row["feasible"],
                    )

            feasible_refined = [row for row in refined_rows if row["is_interior"] == 1 and row["feasible"] == 1]
            alpha_star_refined_row = choose_best_candidate(feasible_refined, args.selection_mode)
            best_row = alpha_star_refined_row or alpha_star_coarse_row

            endpoint_rows.append(
                metrics_to_endpoint_row(args, pair_id, step, epoch_float, args.label_a, 0.0, endpoint_a)
            )
            endpoint_rows.append(
                metrics_to_endpoint_row(args, pair_id, step, epoch_float, args.label_b, 1.0, endpoint_b)
            )
            path_scan_rows.extend(scan_row_to_csv(args, pair_id, step, row) for row in coarse_rows)
            path_scan_rows.extend(scan_row_to_csv(args, pair_id, step, row) for row in refined_rows)

            endpoint_best_row = choose_best_endpoint([coarse_rows[0], coarse_rows[-1]], args.selection_mode)
            merge_05_row = find_row_by_alpha(coarse_rows, 0.5)
            if merge_05_row is None:
                raise ValueError("Alpha grid did not contain 0.5 after normalization")

            j_endpoint_best = endpoint_best_row["j_score"] if endpoint_best_row is not None else np.nan
            j_merge_05 = merge_05_row["j_score"]
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
            coarse_curve = sorted(coarse_rows, key=lambda row: float(row["alpha"]))
            retain_barrier = compute_barrier(coarse_curve, "dr_loss")
            forget_barrier = compute_barrier(coarse_curve, "df_loss")
            val_barrier = compute_barrier(coarse_curve, "val_loss")
            test_barrier = compute_barrier(coarse_curve, "test_loss")
            step_summary = {
                "exp_id": args.exp_id,
                "ratio": args.ratio,
                "pair_id": pair_id,
                "step": step,
                "epoch_float": epoch_float,
                "ua_a": endpoint_a["ua"],
                "ua_b": endpoint_b["ua"],
                "tau_ua": tau_ua,
                "feasible_count": len(feasible_coarse),
                "alpha_star_coarse": alpha_star_coarse_row["alpha"] if alpha_star_coarse_row is not None else np.nan,
                "alpha_star_refined": best_row["alpha"] if best_row is not None else np.nan,
                "j_endpoint_best": j_endpoint_best,
                "j_merge_05": j_merge_05,
                "j_merge_best": j_merge_best,
                "delta_05": delta_05,
                "delta_int": delta_int,
                "endpoint_best_model": args.label_a if endpoint_best_row is coarse_rows[0] else args.label_b,
                "retain_barrier": retain_barrier,
                "forget_barrier": forget_barrier,
                "val_barrier": val_barrier,
                "test_barrier": test_barrier,
            }
            step_summary_rows.append(step_summary)
            logger.emit(
                "SELECT",
                ratio=args.ratio,
                step=step,
                feasible_count=len(feasible_coarse),
                alpha_star=step_summary["alpha_star_coarse"],
                alpha_star_refined=step_summary["alpha_star_refined"],
                J_end=j_endpoint_best,
                J_05=j_merge_05,
                J_star=j_merge_best,
                delta_05=delta_05,
                delta_int=delta_int,
            )
            logger.emit(
                "BARRIER",
                ratio=args.ratio,
                step=step,
                retain_barrier=retain_barrier,
                forget_barrier=forget_barrier,
                val_barrier=val_barrier,
                test_barrier=test_barrier,
            )

            candidate_rows = [
                ("endpoint_a", coarse_rows[0]),
                ("endpoint_b", coarse_rows[-1]),
                ("merge_05", merge_05_row),
                ("merge_best", best_row),
            ]
            for candidate_type, row in candidate_rows:
                full_row = full_eval_row(args, pair_id, step, epoch_float, candidate_type, row)
                full_eval_rows.append(full_row)
                logger.emit(
                    "FULL",
                    ratio=args.ratio,
                    step=step,
                    candidate=candidate_type,
                    alpha=full_row["alpha"],
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

        experiment.write_csv(
            os.path.join(args.output_dir, "checkpoint_manifest.csv"),
            checkpoint_manifest_rows,
            CHECKPOINT_MANIFEST_FIELDS,
        )
        experiment.write_csv(
            os.path.join(args.output_dir, "endpoint_metrics.csv"),
            endpoint_rows,
            ENDPOINT_METRICS_FIELDS,
        )
        experiment.write_csv(
            os.path.join(args.output_dir, "path_scan.csv"),
            path_scan_rows,
            PATH_SCAN_FIELDS,
        )
        experiment.write_csv(
            os.path.join(args.output_dir, "step_summary.csv"),
            step_summary_rows,
            STEP_SUMMARY_FIELDS,
        )
        experiment.write_csv(
            os.path.join(args.output_dir, "full_eval.csv"),
            full_eval_rows,
            FULL_EVAL_FIELDS,
        )
        if not args.skip_plots:
            plot_outputs(args.output_dir, step_summary_rows, full_eval_rows)
        print(f"Wrote step connectivity artifacts to {os.path.abspath(args.output_dir)}")
    finally:
        logger.close()


if __name__ == "__main__":
    main()
