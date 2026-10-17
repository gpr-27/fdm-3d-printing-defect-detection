# models/

Checkpoints are not committed: GitHub rejects files over 100 MB, and the AlexNet checkpoint is ~228 MB.
Attach them to a GitHub Release (or use Git LFS) if you want to share them. Each one is a `state_dict` and loads with
`torch.load(path, weights_only=True)`.

| File | Produced by | Size |
|---|---|---|
| `alexnet_best_model.pth` (= `alexnet_fdm_5class_model.pth`, identical tensors) | notebook 1 | 228 MB |
| `resnet_best_model.pth` (= `resnet_fdm_5class_model.pth`) | notebook 2 (ResNet18) | 45 MB |
| `inception_v3_best_model.pth` (= `inception_v3_fdm_5class_model.pth`, identical tensors) | notebook 3 | 98 MB |
| `binary_alexnet_best_model.pth`, `binary_resnet18_best_model.pth`, `binary_inception_v3_best_model.pth` | notebook 4 | 228 / 45 / 98 MB |

Re-score any checkpoint on the validation split without training:

```bash
PYTHONPATH=src python -m fdm_defects.evaluate --data-dir data --model alexnet --checkpoint models/alexnet_best_model.pth
```
