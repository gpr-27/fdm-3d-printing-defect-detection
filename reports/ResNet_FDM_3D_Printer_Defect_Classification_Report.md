# ResNet - FDM 3D Printer Defect Classification

## Confusion Matrix (Numeric)

Classes: Cracking, Layer_shifting, Off_platform, Stringing, Warping

Confusion matrix (rows = actual, columns = predicted):
[[62 12  0  1 17]
 [26 17  0  1 18]
 [ 0  1  0  2 11]
 [ 9  2  0 80 10]
 [13  1  0  4 96]]

## Classification Report

                precision    recall  f1-score   support

      Cracking       0.56      0.67      0.61        92
Layer_shifting       0.52      0.27      0.36        62
  Off_platform       0.00      0.00      0.00        14
     Stringing       0.91      0.79      0.85       101
       Warping       0.63      0.84      0.72       114

      accuracy                           0.67       383
     macro avg       0.52      0.52      0.51       383
  weighted avg       0.65      0.67      0.64       383


## Per-class Accuracy

Cracking: 62/92 = 67.39%
Layer_shifting: 17/62 = 27.42%
Off_platform: 0/14 = 0.00%
Stringing: 80/101 = 79.21%
Warping: 96/114 = 84.21%

Confusion matrix figure saved at: ../figures/confusion_matrix_resnet_fdm_defects.png
