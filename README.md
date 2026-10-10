# Predicting CYP2C19 Inhibition from Molecular Structures

## Overview

This project explores the application of machine learning to molecular property prediction, with a focus on drug metabolism and pharmacokinetics (ADME).

The objective is to investigate whether molecular representations derived from SMILES strings can be used to predict whether a compound inhibits **CYP2C19**, an enzyme involved in the metabolism of many drugs.

The project will compare molecular featurization strategies and supervised learning approaches to evaluate their predictive performance and generalization to unseen compounds.

## Dataset

The project uses the **CYP2C19_Veith** dataset from [Therapeutics Data Commons (TDC)](https://tdcommons.ai/).

- **Task:** Binary classification
- **Target:** CYP2C19 inhibition
- **Input:** Molecular structures encoded as SMILES strings
- **Application:** Drug metabolism and pharmacokinetics (ADME)

The dataset is accessed programmatically through the PyTDC Python package.

## Methodology

The project will be developed through the following steps:

1. **Data exploration:** inspect the dataset, target labels, class distribution and molecular structures.

## Environment Setup

The project uses Conda to manage its Python environment.

### 1. Create the environment

```bash
conda env create -f environment.yaml
```

### 2. Activate the environment

```bash
conda activate cyp2c19-ml
```

### 3. Run the dataset loading script

```bash
python load_dataset.py
```

The dataset is retrieved through TDC. Depending on the local cache, it may be downloaded or loaded from an existing local copy.

## Project Status

**Current stage:** Environment setup and dataset access.

The Conda environment has been tested, and the dataset loading script executes successfully. Data exploration, model development and evaluation will be documented as they are completed.

## Tools and Technologies

- Python
- Conda
- PyTDC — dataset access
- Machine learning and molecular data analysis

## References

- [Therapeutics Data Commons](https://tdcommons.ai/)
- [CYP2C19_Veith dataset](https://tdcommons.ai/benchmark/admet_group/overview/)

