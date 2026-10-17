# Data

Images are not committed (6.3 GB). Download them from Kaggle and place the class folders exactly here, because the
notebooks read `../data` (notebooks 1–3) and `../data_binary` (notebook 4):

```
data/                    5-class task (notebooks 1–3)
├── Cracking/            472 images
├── Layer_shifting/      364
├── Off_platform/         91
├── Stringing/           447
└── Warping/             538      (1,912 JPGs, all 3072×2048)
data_binary/             binary task (notebook 4)
├── defected/            759      (bed_not_stick 94, leg_broken 214, no_bottom 96, no_support 355)
└── no_defected/         798      (scratch 184, scratch_2 282, scratch_3 210, scratch_4 122; all 640×480)
```

| Task | Kaggle dataset | Author / licence | Citation |
|---|---|---|---|
| 5-class | [`wengmhu/fdm-3d-printing-defect-dataset`](https://www.kaggle.com/datasets/wengmhu/fdm-3d-printing-defect-dataset) | Hu Wenjing, Chen Chang — MIT | Hu, W., Chen, C., Su, S. *et al.* Real-time defect detection for FFF 3D printing using lightweight model deployment. *Int J Adv Manuf Technol* 134, 4871–4885 (2024). https://doi.org/10.1007/s00170-024-14452-4 |
| Binary | [`justin900429/3d-printer-defected-dataset`](https://www.kaggle.com/datasets/justin900429/3d-printer-defected-dataset) | Justin Ruan — CC BY-SA 4.0 | https://github.com/Justin900429/3d-printer-anomaly-detect |

Notes
- The 5-class dataset card lists Cracking as 427 images; the downloaded folder contains **472**
  (472 + 364 + 91 + 447 + 538 = 1,912, which matches the card's total).
- Both datasets are time-lapse frames from a few print sessions, and consecutive frames are near-duplicates. A random
  frame-level split is therefore optimistic. `python -m fdm_defects.train --split grouped` holds out whole sessions instead.
- `figures/kb_sample_images_5class.jpg`, `figures/kb_sample_images_binary.jpg` and `figures/kb_leakage_pairs.jpg`
  show frames from these two datasets (credited above). Because the binary dataset is CC BY-SA 4.0, those figures are
  shared under CC BY-SA 4.0 as well.
