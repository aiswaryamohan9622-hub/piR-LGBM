# DeepMDA-piRNA --- Autoencoder + LightGBM for piRNA--Disease Association Prediction

This project implements a deep learning + gradient boosting framework
for predicting **piRNA--disease associations**.

The pipeline combines:

-   Deep Autoencoder (TensorFlow/Keras) for nonlinear feature extraction
-   LightGBM for classification
-   5-Fold Cross Validation
-   ROC and Precision--Recall evaluation

Designed and tested in Google Colab.

------------------------------------------------------------------------

## Project Overview

Workflow:

1.  Load disease features, piRNA similarity features, and adjacency
    matrix
2.  Construct positive and negative samples
3.  Balance dataset via negative sampling
4.  Normalize features (train-only normalization to avoid data leakage)
5.  Train Autoencoder on training fold
6.  Extract encoded features
7.  Concatenate raw + encoded features
8.  Train LightGBM classifier
9.  Evaluate using:

-   Accuracy
-   Precision
-   Sensitivity (Recall)
-   Specificity
-   MCC
-   AUC
-   AUPR
-   F1-score

------------------------------------------------------------------------

## Dataset Structure

Place the following files in:

My Drive/AISH/

Files:

-   d2d_do.csv → Disease feature matrix
-   half_p2p_smith.csv → piRNA similarity feature matrix
-   adj.csv → piRNA--disease interaction matrix

------------------------------------------------------------------------

## How to Run (Google Colab)

### 1. Mount Google Drive

``` python
from google.colab import drive
drive.mount('/content/drive')
```

### 2. Install Dependencies

``` python
!pip install lightgbm
```

(TensorFlow, NumPy, Pandas, sklearn, matplotlib are pre-installed in
Colab.)

### 3. Run

``` python
if __name__ == "__main__":
    DeepMDA()
```

------------------------------------------------------------------------

## Model Architecture

Autoencoder Encoder:

Input → 512 → 256 → 128 → 64

Decoder:

64 → 128 → 256 → 512 → Output

-   Activation: ReLU
-   Output Activation: Sigmoid
-   Loss: MSE
-   Optimizer: Adam (1e-3)

LightGBM is trained on concatenated raw + encoded features.

------------------------------------------------------------------------

## Evaluation

-   5-Fold Cross Validation
-   Mean ROC curve
-   Mean AUC and AUPR
-   Class balancing through random negative sampling

------------------------------------------------------------------------

## Requirements

-   Python 3.8+
-   TensorFlow 2.x
-   LightGBM
-   NumPy
-   Pandas
-   scikit-learn
-   Matplotlib

------------------------------------------------------------------------

## Notes

-   Normalization is applied inside each fold to prevent data leakage.
-   Autoencoder is trained only on training data in each fold.
-   Early stopping is enabled in LightGBM.

------------------------------------------------------------------------

## Author

Your Name\
Your Institution\
Year
