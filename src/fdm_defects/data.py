"""Data loading exactly as in the notebooks: ImageFolder + one shared transform + seeded 80/20 random_split.

`grouped_split` is an optional leakage-aware alternative: frames from the same capture session
(5-class data: same class + capture date; binary data: same class + filename prefix) never end up
on both sides of the split.
"""
import os
import random
import re
from collections import defaultdict
from itertools import combinations

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def make_transform(input_size):
    """No augmentation, same as the notebooks: resize (aspect ratio not kept), tensor, ImageNet normalisation."""
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def load_folder(data_dir, input_size):
    return datasets.ImageFolder(root=data_dir, transform=make_transform(input_size))


def random_split_80_20(dataset, seed=42):
    n = len(dataset)
    n_train = int(0.8 * n)
    return random_split(dataset, [n_train, n - n_train], generator=torch.Generator().manual_seed(seed))


def session_key(path, class_name):
    """'Image_20231128195336980.jpg' -> class@2023-11-28 ; 'scratch_2_17.jpg' -> class@scratch_2."""
    name = os.path.basename(path)
    m = re.match(r"Image_(\d{4})(\d{2})(\d{2})", name)
    if m:
        return f"{class_name}@{m.group(1)}-{m.group(2)}-{m.group(3)}"
    prefix = re.sub(r"_\d+\.\w+$", "", name)
    return f"{class_name}@{prefix}"


def grouped_split(dataset, val_fraction=0.2, seed=42):
    """Hold out whole sessions *per class*, so every class appears in validation (a plain GroupShuffleSplit can drop
    classes that have only 2-4 sessions). For each class, every non-empty proper subset of its sessions is tried and
    the one whose size is closest to val_fraction of the class goes to validation (ties broken by a seeded shuffle),
    so each class keeps at least one session on each side. A class with a single session stays in training."""
    groups = defaultdict(list)
    for i, (path, target) in enumerate(dataset.samples):
        groups[(target, session_key(path, dataset.classes[target]))].append(i)
    rng = random.Random(seed)
    train_idx, val_idx = [], []
    for target in range(len(dataset.classes)):
        keys = sorted(k for k in groups if k[0] == target)
        rng.shuffle(keys)
        goal = val_fraction * sum(len(groups[k]) for k in keys)
        best = ()
        if len(keys) > 1:
            subsets = (s for r in range(1, len(keys)) for s in combinations(keys, r))
            if len(keys) <= 16:
                best = min(subsets, key=lambda s: abs(sum(len(groups[k]) for k in s) - goal))
            else:                                            # many sessions: greedy, still at least one held out
                best, taken = [], 0
                for k in keys[:-1]:
                    if not best or abs(taken + len(groups[k]) - goal) < abs(taken - goal):
                        best.append(k)
                        taken += len(groups[k])
        for k in keys:
            (val_idx if k in best else train_idx).extend(groups[k])
    return Subset(dataset, sorted(train_idx)), Subset(dataset, sorted(val_idx))


def make_loaders(train_ds, val_ds, batch_size=32, num_workers=2):
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
            DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers))
