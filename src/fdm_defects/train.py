"""Train one model with the notebooks' recipe and write checkpoint, figure, report and JSON.

    python -m fdm_defects.train --data-dir data/fdm_5class --model alexnet --tag fdm5
    python -m fdm_defects.train --data-dir data/binary --model resnet18 --tag binary
"""
import argparse
import json
import os

import torch
import torch.nn as nn
import torch.optim as optim

from .data import grouped_split, load_folder, make_loaders, random_split_80_20
from .engine import evaluate, fit, pick_device
from .metrics import plot_confusion, summarize, write_report
from .models import DEFAULTS, build_model, count_trainable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--model", required=True, choices=sorted(DEFAULTS))
    ap.add_argument("--tag", default="fdm5", help="name prefix for output files, e.g. fdm5 or binary")
    ap.add_argument("--epochs", type=int, default=None, help="default: per-model budget used in the notebooks")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--split", choices=["random", "grouped"], default="random",
                    help="random = notebooks' 80/20 split; grouped = whole capture sessions held out")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device()
    cfg = DEFAULTS[args.model]
    epochs = args.epochs or cfg["epochs"]
    full = load_folder(args.data_dir, cfg["input_size"])
    train_ds, val_ds = (random_split_80_20(full, args.seed) if args.split == "random"
                        else grouped_split(full, 0.2, args.seed))
    train_loader, val_loader = make_loaders(train_ds, val_ds, args.batch_size, args.num_workers)

    model = build_model(args.model, len(full.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    for d in ("models", "figures", "reports"):
        os.makedirs(os.path.join(args.out_dir, d), exist_ok=True)
    stem = f"{args.tag}_{args.model}_{args.split}"
    ckpt = os.path.join(args.out_dir, "models", f"{stem}_best.pth")

    print(f"{args.model} | classes {full.classes} | train {len(train_ds)} val {len(val_ds)} | "
          f"trainable params {count_trainable(model):,} | device {device}")
    run = fit(model, train_loader, val_loader, criterion, optimizer, device, epochs, args.patience, ckpt)

    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    _, _, labels, preds, _ = evaluate(model, val_loader, criterion, device)
    s = summarize(labels, preds, full.classes)
    fig = os.path.join(args.out_dir, "figures", f"confusion_{stem}.png")
    plot_confusion(s["confusion_matrix"], full.classes, f"Confusion Matrix - {args.model} ({args.tag}, {args.split} split)", fig)
    peak = max(h["val_acc"] for h in run["history"])
    write_report(os.path.join(args.out_dir, "reports", f"{stem}_report.md"), f"{args.model} — {args.tag}", s,
                 full.classes, os.path.relpath(fig, os.path.join(args.out_dir, "reports")),
                 [f"- Split: {args.split}, seed {args.seed}; best epoch (lowest val loss): {run['best_epoch']}",
                  f"- Peak validation accuracy during training: {peak:.2f}%"])
    json.dump({"args": vars(args), "run": run, "summary": {k: v for k, v in s.items() if k != "report_text"},
               "peak_val_acc": peak}, open(os.path.join(args.out_dir, "reports", f"{stem}.json"), "w"), indent=2)
    print(f"best-checkpoint accuracy {100 * s['accuracy']:.2f}% ({s['correct']}/{s['n']}) | peak val acc {peak:.2f}%")


if __name__ == "__main__":
    main()
