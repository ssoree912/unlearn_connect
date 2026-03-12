import os

import arg_parser
import experiment_helpers as experiment
import utils


def main():
    parser = arg_parser.build_parser()
    parser.description = "Generate a fixed forget index file for SalUn experiments"
    parser.add_argument(
        "--output_path",
        required=True,
        type=str,
        help="path to the .npy file that will store the fixed forget indexes",
    )
    args = parser.parse_args()
    args = experiment.prepare_experiment_args(args)

    if args.num_indexes_to_replace is None:
        raise ValueError("--num_indexes_to_replace is required to generate fixed forget indexes")
    if args.dataset == "imagenet":
        raise ValueError("Fixed forget index generation is only implemented for finite index datasets")

    utils.setup_seed(args.forget_seed)
    _, train_loader_full, _, _, _ = utils.setup_model_dataset(args)
    indexes = experiment.sample_forget_indices(
        train_loader_full.dataset,
        args.class_to_replace,
        args.num_indexes_to_replace,
        args.forget_seed - 1,
    )
    experiment.save_forget_indices(indexes, args.output_path)

    print(f"Saved {len(indexes)} forget indexes to {os.path.abspath(args.output_path)}")


if __name__ == "__main__":
    main()
