import os

import arg_parser
import experiment_helpers as experiment
import torch
import torch.nn as nn
import utils


def build_parser():
    parser = arg_parser.build_parser()
    parser.description = "Evaluate one or more checkpoints on retain/forget/val/test splits"
    parser.add_argument(
        "--run_dir",
        default=None,
        type=str,
        help="run directory that contains intermediate checkpoints and optionally the final checkpoint",
    )
    parser.add_argument(
        "--checkpoint_paths",
        nargs="*",
        default=None,
        help="explicit checkpoint paths to evaluate",
    )
    parser.add_argument(
        "--output_path",
        default=None,
        type=str,
        help="CSV path for endpoint metrics (defaults to <run_dir>/endpoint_metrics.csv)",
    )
    parser.add_argument(
        "--include_final_checkpoint",
        action="store_true",
        help="include the standard final checkpoint from <run_dir>/<unlearn>checkpoint.pth.tar",
    )
    parser.add_argument(
        "--label",
        default=None,
        type=str,
        help="optional label written to the metrics CSV",
    )
    return parser


def collect_checkpoint_paths(args):
    paths = []

    if args.run_dir is not None:
        checkpoint_dir = experiment.resolve_checkpoint_dir(args, base_dir=args.run_dir)
        if os.path.isdir(checkpoint_dir):
            epoch_paths = [
                os.path.join(checkpoint_dir, name)
                for name in os.listdir(checkpoint_dir)
                if name.startswith("epoch_") and name.endswith(".pth.tar")
            ]
            epoch_paths.sort(key=lambda path: experiment.parse_epoch_from_path(path) or -1)
            paths.extend(epoch_paths)

        if args.include_final_checkpoint:
            final_path = os.path.join(args.run_dir, f"{args.unlearn}checkpoint.pth.tar")
            if os.path.exists(final_path):
                paths.append(final_path)

    if args.checkpoint_paths:
        paths.extend(args.checkpoint_paths)

    deduped_paths = []
    seen = set()
    for path in paths:
        normalized = os.path.abspath(path)
        if normalized in seen:
            continue
        deduped_paths.append(normalized)
        seen.add(normalized)

    if not deduped_paths:
        raise ValueError("No checkpoints found to evaluate")
    return deduped_paths


def resolve_epoch(path, checkpoint, args):
    if isinstance(checkpoint, dict) and checkpoint.get("epoch") is not None:
        return int(checkpoint["epoch"])

    parsed_epoch = experiment.parse_epoch_from_path(path)
    if parsed_epoch is not None:
        return parsed_epoch

    return args.unlearn_epochs


def main():
    args = build_parser().parse_args()
    args = experiment.prepare_experiment_args(args)

    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))
        device = torch.device(f"cuda:{int(args.gpu)}")
    else:
        device = torch.device("cpu")

    checkpoint_paths = collect_checkpoint_paths(args)
    if args.output_path is None:
        if args.run_dir is None:
            raise ValueError("--output_path is required when --run_dir is not set")
        args.output_path = os.path.join(args.run_dir, "endpoint_metrics.csv")

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
    data_loaders, forget_dataset, retain_dataset = experiment.build_eval_data_loaders(
        marked_loader, val_loader, test_loader, args
    )
    criterion = nn.CrossEntropyLoss()

    rows = []
    for checkpoint_path in checkpoint_paths:
        checkpoint = experiment.load_checkpoint_file(checkpoint_path, device)
        state_dict, metadata = experiment.extract_state_dict(checkpoint)
        model.load_state_dict(state_dict, strict=False)

        metrics = experiment.evaluate_model(
            model, data_loaders, forget_dataset, retain_dataset, criterion, args
        )
        row = {
            "label": args.label or os.path.basename(args.run_dir or checkpoint_path),
            "epoch": resolve_epoch(checkpoint_path, metadata, args),
            "global_step": metadata.get("global_step"),
            "epoch_float": metadata.get("epoch_float"),
            "checkpoint_path": checkpoint_path,
            "runtime_seconds": metadata.get("cumulative_runtime_seconds"),
            "valid_run": metrics["valid_run"],
            "acc_retain": metrics["acc_retain"],
            "acc_forget": metrics["acc_forget"],
            "acc_val": metrics["acc_val"],
            "acc_test": metrics["acc_test"],
            "ua": metrics["ua"],
            "mia": metrics["mia"],
            "loss_retain": metrics["loss_retain"],
            "loss_forget": metrics["loss_forget"],
            "loss_val": metrics["loss_val"],
            "loss_test": metrics["loss_test"],
        }
        rows.append(row)

    experiment.write_csv(
        args.output_path,
        rows,
        fieldnames=[
            "label",
            "epoch",
            "global_step",
            "epoch_float",
            "checkpoint_path",
            "runtime_seconds",
            "valid_run",
            "acc_retain",
            "acc_forget",
            "acc_val",
            "acc_test",
            "ua",
            "mia",
            "loss_retain",
            "loss_forget",
            "loss_val",
            "loss_test",
        ],
    )
    print(f"Wrote endpoint metrics to {os.path.abspath(args.output_path)}")


if __name__ == "__main__":
    main()
