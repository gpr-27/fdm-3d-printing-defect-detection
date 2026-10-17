# AlexNet - FDM 3D Printer Defect Classification

## Confusion Matrix (Numeric)

Classes: Cracking, Layer_shifting, Off_platform, Stringing, Warping

Confusion matrix (rows = actual, columns = predicted):
[[ 89   0   0   0   3]
 [  1  61   0   0   0]
 [  0   0  14   0   0]
 [  0   0   0 101   0]
 [  0   0   0   0 114]]

## Classification Report

                precision    recall  f1-score   support

      Cracking       0.99      0.97      0.98        92
Layer_shifting       1.00      0.98      0.99        62
  Off_platform       1.00      1.00      1.00        14
     Stringing       1.00      1.00      1.00       101
       Warping       0.97      1.00      0.99       114

      accuracy                           0.99       383
     macro avg       0.99      0.99      0.99       383
  weighted avg       0.99      0.99      0.99       383


## Per-class Accuracy

Cracking: 89/92 = 96.74%
Layer_shifting: 61/62 = 98.39%
Off_platform: 14/14 = 100.00%
Stringing: 101/101 = 100.00%
Warping: 114/114 = 100.00%

Confusion matrix figure saved at: ../figures/confusion_matrix_alexnet_fdm_defects.png
