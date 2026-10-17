# FDM 3D Printing Defect Detection

**PyTorch · Computer Vision · Transfer Learning (ImageNet) · AlexNet, ResNet18, InceptionV3**

CNN models that detect defects in FDM (fused deposition modelling) 3D printing from camera images, evaluated on two tasks:

1. **Binary defect detection**: `defected` vs `no_defected` frames of prints in progress.
2. **5-class defect classification**: Cracking, Layer shifting, Off platform, Stringing, Warping.

Every model is an ImageNet-pretrained backbone with a new classification head, trained with the same recipe:
Adam (lr 1e-4), batch 32, cross-entropy, early stopping on validation loss (patience 3), 80/20 `random_split` with seed 42.

## Results

**Binary defect detection** (validation split: 312 images, 153 defected / 159 no_defected), notebook `4_binary_defect_detection.ipynb`

| Model | Trainable params | Best epoch | Accuracy | Correct | ROC-AUC | Defected recall |
|---|---:|:---:|---:|---:|---:|---:|
| **AlexNet** | 54,542,338 | 4 | **99.68%** | 311/312 | 1.0000 | 99.35% |
| InceptionV3 | 2,563,332 | 10 | 91.03% | 284/312 | 0.9801 | 88.24% |
| ResNet18 | 1,026 | 4 | 90.38% | 282/312 | 0.9733 | 91.50% |

**5-class defect classification** (validation split: 383 images), notebooks 1–3

| Model | Trainable params | Epochs | Best validation accuracy | Saved checkpoint (lowest val loss) | Macro-F1 (checkpoint) |
|---|---:|:---:|---:|---:|---:|
| **AlexNet** | 54,554,629 | 4 | **99.22%** (epoch 4, 380/383) | 98.96% (epoch 2, 379/383) | 0.99 |
| InceptionV3 | 2,571,786 | 10 | 87.47% (epoch 10) | 87.47% (335/383) | 0.82 |
| ResNet18 | 2,565 | 4 | 66.58% (epoch 4) | 66.58% (255/383) | 0.51 |

AlexNet's advantage comes mostly from what is trained: its whole 3-layer classifier (~54.6M parameters) adapts to the
new images, while the ResNet18 and InceptionV3 runs train only the final layer(s) on top of a frozen backbone, and were
still improving when their epoch budget ran out.

Confusion matrices are in `figures/`, per-model reports in `reports/`, and every number above is printed in the
executed notebooks.

## Data

| Task | Source | Images |
|---|---|---|
| Binary | Kaggle `justin900429/3d-printer-defected-dataset` (CC BY-SA 4.0) | 759 defected + 798 no_defected, 640×480 |
| 5-class | Kaggle `wengmhu/fdm-3d-printing-defect-dataset` (MIT; Hu et al., *Int J Adv Manuf Technol* 2024) | 1,912 images, 3072×2048 |

Download instructions and the expected folder layout are in [`DATA.md`](DATA.md).

## Evaluation caveat: near-duplicate frames

Both datasets are time-lapse frames from a handful of print sessions, so a random frame-level split puts neighbouring
frames of the same print into both train and validation:

- 5-class: for 361 of 383 validation images (94%), a same-class training frame was captured within 30 seconds.
- Binary: for 299 of 312 validation images (95.8%), the previous or next frame of the same session is in the training set.

The scores above are therefore best read as performance on prints the model has seen during training. A stricter test
holds out whole sessions: `python -m fdm_defects.train --split grouped …`.

## Repository layout

```
notebooks/
  1_alexnet_fdm3d_defects.ipynb      5-class, AlexNet      (executed)
  2_resnet18_fdm3d_defects.ipynb     5-class, ResNet18     (executed)
  3_inception_fdm3d_defects.ipynb    5-class, InceptionV3  (executed)
  4_binary_defect_detection.ipynb    binary, all three models (executed)
src/fdm_defects/   data.py · models.py · engine.py · metrics.py · train.py · evaluate.py
tests/             test_fdm.py   (split logic, session grouping, trainable-parameter counts)
figures/           confusion matrices and training curves
reports/           per-model reports and binary_results_summary.json
models/            checkpoints are not committed (see models/README.md)
```

## Reproduce

```bash
pip install -r requirements.txt
# put the images in data/ and data_binary/ (see DATA.md), then either run the notebooks, or:
PYTHONPATH=src python -m fdm_defects.train --data-dir data --model alexnet --tag fdm5
PYTHONPATH=src python -m fdm_defects.train --data-dir data_binary --model alexnet --tag binary
# re-score a saved checkpoint without training
PYTHONPATH=src python -m fdm_defects.evaluate --data-dir data --model alexnet --checkpoint models/alexnet_best_model.pth
PYTHONPATH=src pytest -q
```

Runs used PyTorch on an Apple Silicon GPU (MPS). The code picks MPS, then CUDA, then CPU automatically.

## References

- Krizhevsky, Sutskever, Hinton. *ImageNet Classification with Deep Convolutional Neural Networks*. NeurIPS 2012.
- He, Zhang, Ren, Sun. *Deep Residual Learning for Image Recognition*. CVPR 2016. https://arxiv.org/abs/1512.03385
- Szegedy et al. *Rethinking the Inception Architecture for Computer Vision*. CVPR 2016. https://arxiv.org/abs/1512.00567
- Hu, W., Chen, C., Su, S. *et al.* Real-time defect detection for FFF 3D printing using lightweight model deployment. *Int J Adv Manuf Technol* 134, 4871–4885 (2024). https://doi.org/10.1007/s00170-024-14452-4
