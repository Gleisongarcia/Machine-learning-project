# CYP2C19 Inhibition Prediction Using Machine Learning

## Overview

This project explores the use of machine learning to predict whether a molecule inhibits **CYP2C19**, a cytochrome P450 enzyme involved in the metabolism of several therapeutic compounds.

The objective is to investigate how different molecular representations and classification algorithms can be used to predict CYP2C19 inhibition from chemical structures encoded as SMILES strings.

Two molecular representations are compared:

- **Morgan fingerprints**, which encode local molecular environments.
- **SMILES character n-grams**, which extract sequential patterns directly from molecular strings.

Both representations are evaluated using Logistic Regression and Random Forest classifiers. Hyperparameter optimization and validation-based model selection are then performed to identify the final model.

## Dataset

The analysis uses the `CYP2C19_Veith` dataset from [Therapeutics Data Commons (TDC)](https://tdcommons.ai/).

The dataset contains **12,665 molecules**, each described by:

| Variable | Description |
|---|---|
| `Drug_ID` | Molecule identifier |
| `Drug` | Molecular structure encoded as a SMILES string |
| `Y` | Binary CYP2C19 inhibition label |

The prediction task is defined as:

- `0`: Non-inhibitor
- `1`: Inhibitor

The dataset is divided into three subsets using a fixed random seed (`42`):

| Subset | Proportion | Number of molecules |
|---|---:|---:|
| Training | 70% | 8,866 |
| Validation | 10% | 1,266 |
| Test | 20% | 2,533 |

The training set is used for model fitting and cross-validation, the validation set for model comparison and selection, and the test set for the final evaluation.

## Methodology

The project follows a structured machine learning workflow.

### 1. Exploratory Data Analysis

- Dataset inspection and data quality assessment.
- Missing value and duplicate checks.
- SMILES validity assessment using RDKit.
- Target class distribution analysis.
- SMILES string length distribution.

### 2. Molecular Feature Engineering

Two approaches are investigated:

**Morgan fingerprints**

Molecular structures are converted into fixed-length binary fingerprints using RDKit. The baseline representation uses radius 2 and 2,048 bits.

**SMILES character n-grams**

SMILES strings are represented using character sequences extracted with scikit-learn's `CountVectorizer`. The baseline configuration considers n-grams of lengths 2 to 4, with a maximum vocabulary size of 30,000 features.

### 3. Baseline Models

Four baseline configurations are evaluated:

| Molecular representation | Classifier |
|---|---|
| Morgan fingerprints | Logistic Regression |
| Morgan fingerprints | Random Forest |
| SMILES character n-grams | Logistic Regression |
| SMILES character n-grams | Random Forest |

### 4. Hyperparameter Optimization

Hyperparameter optimization is performed using **five-fold stratified cross-validation** on the training set.

The search explores molecular representation parameters and classifier hyperparameters. ROC-AUC is used as the primary optimization metric.

For the text-based models, `CountVectorizer` is integrated into scikit-learn pipelines to ensure that the vocabulary is learned independently within each cross-validation training fold, preventing data leakage.

### 5. Model Comparison and Final Evaluation

Baseline and optimized candidates are compared on the same validation set using:

- Accuracy
- Balanced accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

The final model is selected using validation results and evaluated once on the held-out test set.

## Results

### Final model

The selected model is **Logistic Regression using optimized Morgan fingerprints**.

| Parameter | Selected configuration |
|---|---|
| Molecular representation | Morgan fingerprints |
| Radius | 2 |
| Fingerprint size | 4,096 bits |
| Classifier | Logistic Regression |
| Regularization strength (`C`) | 0.1 |

### Final test set performance

The model achieved the following results on the 2,533 molecules reserved for final evaluation:

| Metric | Test score |
|---|---:|
| Accuracy | **0.817** |
| Balanced accuracy | **0.817** |
| Precision | **0.795** |
| Recall | **0.811** |
| F1-score | **0.803** |
| ROC-AUC | **0.884** |

The confusion matrix shows that the model correctly identified 1,127 non-inhibitors and 943 inhibitors, with 243 false positives and 220 false negatives.

These results indicate that the selected model achieves a reasonably balanced performance across both classes.

## Technologies and Libraries

- **Python**
- **Pandas** and **NumPy** — data manipulation and numerical operations.
- **Matplotlib** — data visualization.
- **RDKit** — molecular processing and Morgan fingerprint generation.
- **scikit-learn** — feature extraction, classification, cross-validation, hyperparameter optimization and evaluation.
- **SciPy** — sparse matrix operations.
- **Therapeutics Data Commons (TDC)** — dataset access.
- **Jupyter Notebook** — analysis and experimentation.
- **Conda** — environment management.

## Repository Contents

The main deliverable is the Jupyter notebook:

- `cyp2c19_ml_analysis.ipynb` — complete analysis, from dataset exploration and feature engineering to model optimization, evaluation and future perspectives.
- `environment.yml` — Conda environment specification.

## Installation and Reproducibility

Clone the repository:

```bash
git clone https://github.com/Gleisongarcia/Machine-learning-project.git
cd Machine-learning-project
```

Create and activate the Conda environment:

```bash
conda env create -f environment.yml
conda activate smilesproject
```

Launch Jupyter:

```bash
jupyter notebook
```

Open `cyp2c19_ml_analysis.ipynb` and execute the cells sequentially, from beginning to end.

An internet connection may be required to retrieve the dataset through TDC. The notebook uses a fixed random seed for reproducibility of the data split and cross-validation setup.

## Limitations and Future Perspectives

The project provides a baseline for molecular machine learning applied to CYP2C19 inhibition prediction. Several extensions could be investigated:

- Incorporating physicochemical descriptors calculated from SMILES, such as molecular weight, LogP and topological polar surface area.
- Exploring additional molecular representations, including MACCS keys and alternative fingerprints.
- Testing more advanced algorithms, such as gradient boosting and graph neural networks.
- Expanding hyperparameter optimization using alternative search strategies.
- Assessing generalization with scaffold-based splits and independent external datasets.
- Investigating model interpretability and the molecular features associated with prediction errors.

Adding more descriptors or using more complex models does not necessarily improve predictive performance. Further experiments would be required to assess whether these approaches provide reproducible improvements.

## Disclaimer

This project is an academic machine learning study. Predictions represent computational estimates based on the dataset labels and should not be interpreted as experimental confirmation of CYP2C19 inhibition or as a substitute for pharmacological testing.