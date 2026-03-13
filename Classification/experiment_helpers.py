import copy
import csv
import os
import re
from collections import OrderedDict

import evaluation
import numpy as np
import torch
import utils
from imagenet import get_x_y_from_data_dict


def prepare_experiment_args(args):
    if getattr(args, "forget_seed", None) is None:
        args.forget_seed = args.seed
    if getattr(args, "unlearn_seed", None) is None:
        args.unlearn_seed = args.seed
    if getattr(args, "train_seed", None) is None:
        args.train_seed = args.seed

    explicit_indexes = None
    if getattr(args, "forget_index_path", None):
        if getattr(args, "class_to_replace", -1) not in (None, -1):
            raise ValueError(
                "--forget_index_path cannot be combined with a class-specific forget request"
            )
        explicit_indexes = load_forget_indices(args.forget_index_path)
    elif getattr(args, "indexes_to_replace", None) is not None:
        explicit_indexes = np.asarray(args.indexes_to_replace, dtype=np.int64).reshape(-1)

    if explicit_indexes is not None:
        args.indexes_to_replace = explicit_indexes.tolist()
        args.num_indexes_to_replace = int(explicit_indexes.shape[0])
        args.class_to_replace = None
        if getattr(args, "save_forget_index_path", None):
            save_forget_indices(explicit_indexes, args.save_forget_index_path)

    return args


def load_forget_indices(path):
    loaded = np.load(path, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        if "indexes" in loaded.files:
            indexes = loaded["indexes"]
        else:
            indexes = loaded[loaded.files[0]]
    else:
        indexes = loaded

    indexes = np.asarray(indexes, dtype=np.int64).reshape(-1)
    if indexes.size == 0:
        raise ValueError(f"No forget indexes found in {path}")
    if len(np.unique(indexes)) != len(indexes):
        raise ValueError(f"Forget indexes in {path} contain duplicates")
    return indexes


def save_forget_indices(indexes, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.save(path, np.asarray(indexes, dtype=np.int64))


def sample_forget_indices(dataset, class_to_replace, num_indexes_to_replace, seed):
    labels = get_dataset_labels(dataset)
    if class_to_replace in (None, -1):
        candidate_indexes = np.arange(len(labels))
    else:
        candidate_indexes = np.flatnonzero(labels == class_to_replace)

    if num_indexes_to_replace is None:
        return np.asarray(candidate_indexes, dtype=np.int64)

    if num_indexes_to_replace > len(candidate_indexes):
        raise ValueError(
            f"Requested {num_indexes_to_replace} forget samples but only {len(candidate_indexes)} are available"
        )

    rng = np.random.RandomState(seed)
    sampled = rng.choice(candidate_indexes, size=num_indexes_to_replace, replace=False)
    return np.sort(np.asarray(sampled, dtype=np.int64))


def parse_percentage_spec(spec):
    if spec is None or str(spec).strip() == "":
        return []

    percentages = []
    for token in str(spec).split(","):
        token = token.strip()
        if not token:
            continue
        percentage = int(token)
        if percentage <= 0 or percentage >= 100:
            raise ValueError("Forget percentages must be integers in (0, 100)")
        percentages.append(percentage)

    return sorted(set(percentages))


def generate_nested_forget_index_sets(dataset, class_to_replace, percentages, seed):
    if not percentages:
        raise ValueError("At least one forget percentage is required")

    labels = get_dataset_labels(dataset)
    if class_to_replace in (None, -1):
        candidate_indexes = np.arange(len(labels))
    else:
        candidate_indexes = np.flatnonzero(labels == class_to_replace)

    candidate_indexes = np.asarray(candidate_indexes, dtype=np.int64)
    if candidate_indexes.size == 0:
        raise ValueError("No candidate samples available for nested forget index generation")

    rng = np.random.RandomState(seed)
    permutation = rng.permutation(candidate_indexes)

    nested_sets = {}
    previous_count = 0
    total_candidates = candidate_indexes.size
    for percentage in percentages:
        count = int(round(total_candidates * (percentage / 100.0)))
        if count <= 0 or count > total_candidates:
            raise ValueError(
                f"Forget percentage {percentage} produced an invalid count {count}"
            )
        if count < previous_count:
            raise ValueError("Forget percentages must be non-decreasing")

        nested_sets[percentage] = np.sort(permutation[:count].copy())
        previous_count = count

    return nested_sets, permutation


def get_dataset_labels(dataset):
    if hasattr(dataset, "targets"):
        return np.asarray(dataset.targets)
    if hasattr(dataset, "labels"):
        return np.asarray(dataset.labels)
    if hasattr(dataset, "_labels"):
        return np.asarray(dataset._labels)
    raise AttributeError("Dataset does not expose targets/labels/_labels")


def set_dataset_labels(dataset, labels):
    if hasattr(dataset, "targets"):
        dataset.targets = labels
        return
    if hasattr(dataset, "labels"):
        dataset.labels = labels
        return
    if hasattr(dataset, "_labels"):
        dataset._labels = labels
        return
    raise AttributeError("Dataset does not expose targets/labels/_labels")


def _select_dataset_storage(dataset, mask):
    if hasattr(dataset, "data"):
        dataset.data = dataset.data[mask]
        return
    if hasattr(dataset, "imgs"):
        imgs = np.asarray(dataset.imgs, dtype=object)
        dataset.imgs = imgs[mask]
        if hasattr(dataset, "samples"):
            dataset.samples = dataset.imgs
        return
    raise AttributeError("Dataset does not expose data/imgs storage")


def _build_subset_dataset(dataset, is_forget):
    labels = get_dataset_labels(dataset)
    mask = labels < 0 if is_forget else labels >= 0

    subset = copy.deepcopy(dataset)
    subset_labels = labels[mask]
    subset_labels = -subset_labels - 1 if is_forget else subset_labels
    _select_dataset_storage(subset, mask)
    set_dataset_labels(subset, subset_labels)
    return subset


def build_seeded_dataloader(dataset, batch_size, seed, shuffle):
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True,
        shuffle=shuffle,
        generator=generator,
    )


def build_unlearn_data_loaders(marked_loader, val_loader, test_loader, args):
    forget_dataset = _build_subset_dataset(marked_loader.dataset, is_forget=True)
    retain_dataset = _build_subset_dataset(marked_loader.dataset, is_forget=False)

    forget_loader = build_seeded_dataloader(
        forget_dataset, args.batch_size, args.unlearn_seed, shuffle=True
    )
    retain_loader = build_seeded_dataloader(
        retain_dataset, args.batch_size, args.unlearn_seed, shuffle=True
    )

    print(f"number of retain dataset {len(retain_dataset)}")
    print(f"number of forget dataset {len(forget_dataset)}")

    data_loaders = OrderedDict(
        retain=retain_loader,
        forget=forget_loader,
        val=val_loader,
        test=test_loader,
    )
    return data_loaders, forget_dataset, retain_dataset


def parse_epoch_spec(spec):
    if spec is None or str(spec).strip() == "":
        return set()

    epochs = set()
    for token in str(spec).split(","):
        token = token.strip()
        if not token:
            continue
        epoch = int(token)
        if epoch < 0:
            raise ValueError("Checkpoint epochs must be non-negative")
        epochs.add(epoch)
    return epochs


def resolve_checkpoint_dir(args, base_dir=None):
    checkpoint_dir = getattr(args, "checkpoint_dir", "checkpoints")
    if os.path.isabs(checkpoint_dir):
        return checkpoint_dir

    root_dir = base_dir if base_dir is not None else args.save_dir
    if root_dir is None:
        raise ValueError("A save/run directory is required to resolve checkpoint_dir")
    return os.path.join(root_dir, checkpoint_dir)


def save_requested_checkpoint(model, args, epoch, evaluation_result=None, extra_state=None):
    if epoch not in parse_epoch_spec(getattr(args, "checkpoint_epochs", None)):
        return None

    checkpoint_dir = resolve_checkpoint_dir(args)
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch}.pth.tar")

    state = {
        "state_dict": model.state_dict(),
        "evaluation_result": evaluation_result,
        "epoch": epoch,
        "forget_seed": getattr(args, "forget_seed", None),
        "unlearn_seed": getattr(args, "unlearn_seed", None),
        "unlearn": getattr(args, "unlearn", None),
    }
    if extra_state:
        state.update(extra_state)

    torch.save(state, checkpoint_path)
    return checkpoint_path


def parse_epoch_from_path(path):
    match = re.search(r"epoch_(\d+)\.pth\.tar$", os.path.basename(path))
    if match is None:
        return None
    return int(match.group(1))


def load_checkpoint_file(path, device):
    return torch.load(path, map_location=device)


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"], checkpoint
    if isinstance(checkpoint, dict):
        return checkpoint, {"state_dict": checkpoint}
    raise ValueError("Unsupported checkpoint format")


def interpolate_state_dict(state_a, state_b, alpha):
    interpolated = {}
    for name, tensor_a in state_a.items():
        tensor_b = state_b[name]
        if torch.is_floating_point(tensor_a):
            interpolated[name] = (1.0 - alpha) * tensor_a + alpha * tensor_b
        else:
            interpolated[name] = tensor_a.clone()
    return interpolated


def evaluate_loader(loader, model, criterion, args):
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        if args.imagenet_arch:
            for data in loader:
                image, target = get_x_y_from_data_dict(data, device)
                output = model(image)
                loss = criterion(output, target)

                prec1 = utils.accuracy(output.float().data, target)[0]
                losses.update(loss.float().item(), image.size(0))
                top1.update(prec1.item(), image.size(0))
        else:
            for image, target in loader:
                image = image.to(device)
                target = target.to(device)
                output = model(image)
                loss = criterion(output, target)

                prec1 = utils.accuracy(output.float().data, target)[0]
                losses.update(loss.float().item(), image.size(0))
                top1.update(prec1.item(), image.size(0))

    return {"loss": losses.avg, "acc": top1.avg}


def evaluate_model(model, data_loaders, forget_dataset, retain_dataset, criterion, args):
    split_metrics = {}
    for loader in data_loaders.values():
        utils.dataset_convert_to_test(loader.dataset, args)

    for split_name, loader in data_loaders.items():
        split_metrics[split_name] = evaluate_loader(loader, model, criterion, args)

    test_loader = data_loaders["test"]
    forget_loader = data_loaders["forget"]

    shadow_train = torch.utils.data.Subset(retain_dataset, list(range(len(test_loader.dataset))))
    shadow_train_loader = torch.utils.data.DataLoader(
        shadow_train, batch_size=args.batch_size, shuffle=False
    )
    mia_result = safe_svc_mia(
        shadow_train=shadow_train_loader,
        shadow_test=test_loader,
        target_train=None,
        target_test=forget_loader,
        model=model,
    )

    return {
        "acc_retain": split_metrics["retain"]["acc"],
        "acc_forget": split_metrics["forget"]["acc"],
        "acc_val": split_metrics["val"]["acc"],
        "acc_test": split_metrics["test"]["acc"],
        "loss_retain": split_metrics["retain"]["loss"],
        "loss_forget": split_metrics["forget"]["loss"],
        "loss_val": split_metrics["val"]["loss"],
        "loss_test": split_metrics["test"]["loss"],
        "ua": 100.0 - split_metrics["forget"]["acc"],
        "mia": 100.0 * float(mia_result["confidence"]),
        "svc_mia_forget_efficacy": mia_result,
    }


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_oracle_metrics(path):
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No oracle metrics found in {path}")

    def _epoch_value(row):
        value = row.get("epoch")
        return int(value) if value not in (None, "", "None") else -1

    return max(rows, key=_epoch_value)


def numeric_metric(row, key):
    return float(row[key])


def nan_mia_result():
    return {
        "correctness": np.nan,
        "confidence": np.nan,
        "entropy": np.nan,
        "m_entropy": np.nan,
        "prob": np.nan,
    }


def safe_svc_mia(shadow_train, shadow_test, target_train, target_test, model):
    try:
        return evaluation.SVC_MIA(
            shadow_train=shadow_train,
            shadow_test=shadow_test,
            target_train=target_train,
            target_test=target_test,
            model=model,
        )
    except ValueError as error:
        print(f"Skipping SVC_MIA due to non-finite features: {error}")
        return nan_mia_result()
