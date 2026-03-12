import os

import arg_parser
import experiment_helpers as experiment
import numpy as np
import torch
import torch.nn as nn
import utils


def build_parser():
    parser = arg_parser.build_parser()
    parser.description = "Evaluate linear interpolation between two checkpoint trajectories"
    parser.add_argument("--run_a_dir", required=True, type=str, help="first run directory")
    parser.add_argument("--run_b_dir", required=True, type=str, help="second run directory")
    parser.add_argument(
        "--curve_epochs",
        required=True,
        type=str,
        help="comma-separated checkpoint epochs to interpolate, e.g. 0,1,3,5,10",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=str,
        help="directory where interpolation CSVs will be written",
    )
    parser.add_argument(
        "--lambda_step",
        default=0.05,
        type=float,
        help="step size for lambda in [0, 1]",
    )
    parser.add_argument(
        "--retrain_metrics_path",
        default=None,
        type=str,
        help="optional endpoint_metrics.csv for the retrain oracle",
    )
    parser.add_argument(
        "--label_a",
        default="A",
        type=str,
        help="label for run A in the retrain gap summary",
    )
    parser.add_argument(
        "--label_b",
        default="B",
        type=str,
        help="label for run B in the retrain gap summary",
    )
    return parser


def load_epoch_state(run_dir, checkpoint_dir, epoch, device):
    checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch}.pth.tar")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint for epoch {epoch} not found in {run_dir}. Expected {checkpoint_path}"
        )
    checkpoint = experiment.load_checkpoint_file(checkpoint_path, device)
    state_dict, metadata = experiment.extract_state_dict(checkpoint)
    return checkpoint_path, state_dict, metadata


def compute_barrier(curve_rows, key):
    start_value = float(curve_rows[0][key])
    end_value = float(curve_rows[-1][key])
    barrier = []
    valley = []
    for row in curve_rows:
        lambda_value = float(row["lambda"])
        linear_value = (1.0 - lambda_value) * start_value + lambda_value * end_value
        current_value = float(row[key])
        barrier.append(current_value - linear_value)
        valley.append(linear_value - current_value)
    return max(barrier), max(valley)


def append_gap_rows(gap_rows, epoch, endpoint_row, oracle_row, run_label):
    gap_rows.append(
        {
            "run": run_label,
            "epoch": epoch,
            "gap_acc_retain": float(endpoint_row["acc_retain"]) - experiment.numeric_metric(oracle_row, "acc_retain"),
            "gap_acc_forget": float(endpoint_row["acc_forget"]) - experiment.numeric_metric(oracle_row, "acc_forget"),
            "gap_acc_val": float(endpoint_row["acc_val"]) - experiment.numeric_metric(oracle_row, "acc_val"),
            "gap_acc_test": float(endpoint_row["acc_test"]) - experiment.numeric_metric(oracle_row, "acc_test"),
            "gap_loss_retain": float(endpoint_row["loss_retain"]) - experiment.numeric_metric(oracle_row, "loss_retain"),
            "gap_loss_forget": float(endpoint_row["loss_forget"]) - experiment.numeric_metric(oracle_row, "loss_forget"),
            "gap_loss_val": float(endpoint_row["loss_val"]) - experiment.numeric_metric(oracle_row, "loss_val"),
            "gap_loss_test": float(endpoint_row["loss_test"]) - experiment.numeric_metric(oracle_row, "loss_test"),
            "gap_ua": float(endpoint_row["ua"]) - experiment.numeric_metric(oracle_row, "ua"),
            "gap_mia": float(endpoint_row["mia"]) - experiment.numeric_metric(oracle_row, "mia"),
        }
    )


def main():
    args = build_parser().parse_args()
    args = experiment.prepare_experiment_args(args)

    if not 0.0 < args.lambda_step <= 1.0:
        raise ValueError("--lambda_step must be in (0, 1]")

    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))
        device = torch.device(f"cuda:{int(args.gpu)}")
    else:
        device = torch.device("cpu")

    utils.setup_seed(args.forget_seed)
    (
        model,
        _train_loader_full,
        val_loader,
        test_loader,
        marked_loader,
    ) = utils.setup_model_dataset(args)
    model.to(device)
    utils.setup_seed(args.unlearn_seed)
    data_loaders, forget_dataset, retain_dataset = experiment.build_unlearn_data_loaders(
        marked_loader, val_loader, test_loader, args
    )
    criterion = nn.CrossEntropyLoss()

    checkpoint_dir_a = experiment.resolve_checkpoint_dir(args, base_dir=args.run_a_dir)
    checkpoint_dir_b = experiment.resolve_checkpoint_dir(args, base_dir=args.run_b_dir)
    oracle_row = (
        experiment.load_oracle_metrics(args.retrain_metrics_path)
        if args.retrain_metrics_path is not None
        else None
    )
    lambda_values = np.round(
        np.arange(0.0, 1.0 + args.lambda_step / 2.0, args.lambda_step), 10
    )

    os.makedirs(args.output_dir, exist_ok=True)
    barrier_rows = []
    gap_rows = []

    for epoch in sorted(experiment.parse_epoch_spec(args.curve_epochs)):
        path_a, state_a, _ = load_epoch_state(args.run_a_dir, checkpoint_dir_a, epoch, device)
        path_b, state_b, _ = load_epoch_state(args.run_b_dir, checkpoint_dir_b, epoch, device)

        curve_rows = []
        for lambda_value in lambda_values:
            interpolated_state = experiment.interpolate_state_dict(
                state_a, state_b, float(lambda_value)
            )
            model.load_state_dict(interpolated_state, strict=False)
            metrics = experiment.evaluate_model(
                model, data_loaders, forget_dataset, retain_dataset, criterion, args
            )
            curve_rows.append(
                {
                    "epoch": epoch,
                    "lambda": float(lambda_value),
                    "checkpoint_path_a": path_a,
                    "checkpoint_path_b": path_b,
                    "loss_retain": metrics["loss_retain"],
                    "loss_forget": metrics["loss_forget"],
                    "loss_val": metrics["loss_val"],
                    "loss_test": metrics["loss_test"],
                    "acc_retain": metrics["acc_retain"],
                    "acc_forget": metrics["acc_forget"],
                    "acc_val": metrics["acc_val"],
                    "acc_test": metrics["acc_test"],
                    "ua": metrics["ua"],
                    "mia": metrics["mia"],
                }
            )

        experiment.write_csv(
            os.path.join(args.output_dir, f"epoch_{epoch}_curve.csv"),
            curve_rows,
            fieldnames=[
                "epoch",
                "lambda",
                "checkpoint_path_a",
                "checkpoint_path_b",
                "loss_retain",
                "loss_forget",
                "loss_val",
                "loss_test",
                "acc_retain",
                "acc_forget",
                "acc_val",
                "acc_test",
                "ua",
                "mia",
            ],
        )

        barrier_retain, _ = compute_barrier(curve_rows, "loss_retain")
        barrier_val, _ = compute_barrier(curve_rows, "loss_val")
        barrier_test, _ = compute_barrier(curve_rows, "loss_test")
        barrier_forget, valley_forget = compute_barrier(curve_rows, "loss_forget")
        barrier_rows.append(
            {
                "epoch": epoch,
                "barrier_retain": barrier_retain,
                "barrier_val": barrier_val,
                "barrier_test": barrier_test,
                "barrier_forget": barrier_forget,
                "valley_forget": valley_forget,
            }
        )

        if oracle_row is not None:
            append_gap_rows(gap_rows, epoch, curve_rows[0], oracle_row, args.label_a)
            append_gap_rows(gap_rows, epoch, curve_rows[-1], oracle_row, args.label_b)

    experiment.write_csv(
        os.path.join(args.output_dir, "barrier_summary.csv"),
        barrier_rows,
        fieldnames=[
            "epoch",
            "barrier_retain",
            "barrier_val",
            "barrier_test",
            "barrier_forget",
            "valley_forget",
        ],
    )

    if gap_rows:
        experiment.write_csv(
            os.path.join(args.output_dir, "retrain_gap_summary.csv"),
            gap_rows,
            fieldnames=[
                "run",
                "epoch",
                "gap_acc_retain",
                "gap_acc_forget",
                "gap_acc_val",
                "gap_acc_test",
                "gap_loss_retain",
                "gap_loss_forget",
                "gap_loss_val",
                "gap_loss_test",
                "gap_ua",
                "gap_mia",
            ],
        )

    print(f"Wrote interpolation artifacts to {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
