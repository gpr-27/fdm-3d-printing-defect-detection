"""Re-score a saved checkpoint on the notebooks' validation split (no training).

    python -m fdm_defects.evaluate --data-dir data/fdm_5class --model alexnet \
        --checkpoint models/alexnet_best_model.pth
"""
import argparse

import torch
import torch.nn as nn

from .data import grouped_split, load_folder, make_loaders, random_split_80_20
from .engine import evaluate, pick_device
from .metrics import summarize
from .models import DEFAULTS, build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--model", required=True, choices=sorted(DEFAULTS))
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", choices=["random", "grouped"], default="random")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = pick_device()
    full = load_folder(args.data_dir, DEFAULTS[args.model]["input_size"])
    _, val_ds = random_split_80_20(full, args.seed) if args.split == "random" else grouped_split(full, 0.2, args.seed)
    _, val_loader = make_loaders(val_ds, val_ds, num_workers=0)
    model = build_model(args.model, len(full.classes), pretrained=False)
    # weights_only=True: state_dict tensors only, no arbitrary pickle code is executed
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model.to(device)
    _, _, labels, preds, _ = evaluate(model, val_loader, nn.CrossEntropyLoss(), device)
    s = summarize(labels, preds, full.classes)
    print(f"{args.model}: accuracy {100 * s['accuracy']:.2f}% ({s['correct']}/{s['n']})")
    print("confusion matrix (rows = actual):", full.classes)
    for row in s["confusion_matrix"]:
        print("  ", row)
    print(s["report_text"])


if __name__ == "__main__":
    main()
