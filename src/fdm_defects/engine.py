"""Training / evaluation loops shared by every model (one copy instead of one per notebook)."""
import time

import torch


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_one_epoch(model, loader, criterion, optimizer, device):
    """InceptionV3 returns (logits, aux_logits) in train mode: loss = main + 0.4 * aux."""
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        if isinstance(outputs, tuple):
            main_out, aux_out = outputs
            loss = criterion(main_out, labels) + 0.4 * criterion(aux_out, labels)
            outputs = main_out
        else:
            loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * inputs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += inputs.size(0)
    return loss_sum / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Returns (loss, accuracy %, labels, predictions, softmax probabilities)."""
    model.eval()
    loss_sum, total = 0.0, 0
    labels_all, preds_all, probs_all = [], [], []
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss_sum += criterion(outputs, labels).item() * inputs.size(0)
        total += inputs.size(0)
        probs_all += torch.softmax(outputs, 1).cpu().tolist()
        preds_all += outputs.argmax(1).cpu().tolist()
        labels_all += labels.cpu().tolist()
    acc = 100.0 * sum(p == y for p, y in zip(preds_all, labels_all)) / total
    return loss_sum / total, acc, labels_all, preds_all, probs_all


def fit(model, train_loader, val_loader, criterion, optimizer, device, epochs, patience, ckpt_path, log=print):
    """Early stopping on validation loss; the best-val-loss weights are saved to ckpt_path."""
    best_loss, best_epoch, bad, history, t0 = float("inf"), -1, 0, [], time.time()
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc, *_ = evaluate(model, val_loader, criterion, device)
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc, "val_loss": va_loss, "val_acc": va_acc})
        log(f"epoch {epoch}: train loss {tr_loss:.4f} acc {tr_acc:.2f}% | val loss {va_loss:.4f} acc {va_acc:.2f}%")
        if va_loss < best_loss:
            best_loss, best_epoch, bad = va_loss, epoch, 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            bad += 1
            if bad >= patience:
                log(f"early stopping at epoch {epoch}")
                break
    return {"history": history, "best_epoch": best_epoch, "best_val_loss": best_loss,
            "train_seconds": round(time.time() - t0, 1)}
