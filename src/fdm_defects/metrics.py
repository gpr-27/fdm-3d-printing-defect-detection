"""Confusion matrix, classification report, per-class accuracy and the auto-generated markdown report."""
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


def summarize(labels, preds, class_names):
    cm = confusion_matrix(labels, preds, labels=list(range(len(class_names))))
    per_class = {c: {"correct": int(cm[i, i]), "total": int(cm[i].sum()),
                     "accuracy": float(cm[i, i] / cm[i].sum()) if cm[i].sum() else 0.0}
                 for i, c in enumerate(class_names)}
    return {
        "accuracy": float(np.trace(cm) / cm.sum()), "correct": int(np.trace(cm)), "n": int(cm.sum()),
        "confusion_matrix": cm.tolist(), "per_class": per_class,
        "report_text": classification_report(labels, preds, target_names=class_names, digits=4, zero_division=0),
        "report": classification_report(labels, preds, target_names=class_names, output_dict=True, zero_division=0),
    }


def plot_confusion(cm, class_names, title, path):
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(10, 8) if len(class_names) > 2 else (6, 5))
    sns.heatmap(np.array(cm), annot=True, fmt="d", cmap="Blues", xticklabels=class_names,
                yticklabels=class_names, cbar_kws={"label": "Number of Samples"})
    plt.title(title)
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def write_report(path, title, summary, class_names, figure_path=None, extra_lines=()):
    with open(path, "w") as f:
        f.write(f"# {title}\n\n")
        for line in extra_lines:
            f.write(f"{line}\n")
        f.write(f"\n**Accuracy:** {100 * summary['accuracy']:.2f}% ({summary['correct']}/{summary['n']})\n\n")
        f.write("## Confusion matrix (rows = actual, columns = predicted)\n\n")
        f.write("| | " + " | ".join(class_names) + " |\n|---|" + "---|" * len(class_names) + "\n")
        for c, row in zip(class_names, summary["confusion_matrix"]):
            f.write(f"| **{c}** | " + " | ".join(str(v) for v in row) + " |\n")
        f.write("\n## Classification report\n\n```\n" + summary["report_text"] + "```\n\n## Per-class accuracy\n\n")
        for c, v in summary["per_class"].items():
            f.write(f"- {c}: {v['correct']}/{v['total']} = {100 * v['accuracy']:.2f}%\n")
        if figure_path:
            f.write(f"\n![confusion matrix]({figure_path})\n")
