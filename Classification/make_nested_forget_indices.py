import os

import arg_parser
import experiment_helpers as experiment
import utils


def main():
    parser = arg_parser.build_parser()
    parser.description = "Generate nested fixed forget index files for ratio sweeps"
    parser.add_argument(
        "--output_root",
        required=True,
        type=str,
        help="root directory where <ratio>/forget_indices.npy files will be written",
    )
    parser.add_argument(
        "--percentages",
        required=True,
        type=str,
        help="comma-separated forget percentages, e.g. 10,20,30,40,50",
    )
    parser.add_argument(
        "--file_name",
        default="forget_indices.npy",
        type=str,
        help="file name to use inside each ratio directory",
    )
    parser.add_argument(
        "--permutation_output_path",
        default=None,
        type=str,
        help="optional path to save the underlying shared permutation",
    )
    args = parser.parse_args()
    args = experiment.prepare_experiment_args(args)

    if args.dataset == "imagenet":
        raise ValueError("Nested forget index generation is only implemented for finite index datasets")

    percentages = experiment.parse_percentage_spec(args.percentages)
    utils.setup_seed(args.forget_seed)
    _, train_loader_full, _, _, _ = utils.setup_model_dataset(args)
    nested_sets, permutation = experiment.generate_nested_forget_index_sets(
        train_loader_full.dataset,
        args.class_to_replace,
        percentages,
        args.forget_seed - 1,
    )

    os.makedirs(args.output_root, exist_ok=True)
    if args.permutation_output_path is not None:
        experiment.save_forget_indices(permutation, args.permutation_output_path)

    for percentage, indexes in nested_sets.items():
        ratio_dir = os.path.join(args.output_root, str(percentage))
        output_path = os.path.join(ratio_dir, args.file_name)
        experiment.save_forget_indices(indexes, output_path)
        print(f"Saved {len(indexes)} forget indexes to {os.path.abspath(output_path)}")


if __name__ == "__main__":
    main()
