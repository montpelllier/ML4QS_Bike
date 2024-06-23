# ML4QS
This project involves a series of data preprocessing and machine learning tasks for the lesson "Machine Learning for Quantified Self" at VU Amsterdam.

## File Descriptions
- **datasets**: This directory contains the raw datasets.
- **results**: This directory stores the intermediate results and outputs from each step of the execution process.
- **figures**: Contains visualizations and figures generated during the analysis.
- **Python3Code**: Contains Python 3 code from textbook.
- **LSTM**: Contains scripts for LSTM models.

- **classification_RandomForest.py**: Script to run Random Forest classification.
- **classification_Regression.py**: Script to run LogisticRegression.
- **create_dataset.py**: Script to create and preprocess the initial dataset.
- **feature.py**: Script for feature engineering and selection.
- **merge_dataset.py**: Script to merge different datasets.
- **noise_miss.py**: Script to handle noise and missing data.
- **outlier.py**: Script for outlier detection and removal.
- **requirements.txt**: Lists the dependencies and libraries required for the project.

## Instructions

1. Ensure all dependencies listed in `requirements.txt` are installed.
2. Execute the scripts in the specified order:
   - Start with `create_dataset.py` to prepare your dataset.
   - Run `outlier.py` to handle outliers.
   - Execute `noise_miss.py` to manage missing values and noise.
   - Use `feature.py` for feature engineering.
   - Run `classification_RandomForest.py` for Random Forest classification.
   - Execute `classification_Regression.py` for LogisticRegression analysis.
   - Finally, navigate to the `LSTM` directory to handle any LSTM related models.
