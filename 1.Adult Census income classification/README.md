Adult Income Classification Project
A complete machine learning pipeline for predicting whether an individual's income exceeds $50K/year based on the UCI Adult dataset.

Table of Contents
Overview
Dataset
Project Structure
Requirements
Installation
Usage
Pipeline Tasks
Task 1: Dataset Understanding
Task 2: Data Cleaning
Task 3: Feature Engineering
Task 4: Model Building
Task 5: Performance Evaluation
Results
Models Compared
License

Overview
This project implements an end-to-end machine learning workflow to classify adult income levels using demographic and employment data. The pipeline includes data exploration, cleaning, feature engineering, model training, and comprehensive performance evaluation across five different algorithms.

Dataset
Source: UCI Machine Learning Repository - Adult Dataset
File: adult.csv.zip
Target Variable: income (<=50K or >50K)
Features: Age, workclass, education, marital status, occupation, relationship, race, sex, capital gain/loss, hours per week, native country

Project Structure
plain
.
├── adult.csv.zip              # Compressed dataset file
├── main.py                    # Complete pipeline script (this code)
├── README.md                  # Project documentation
└── requirements.txt           # Python dependencies

Requirements
Python 3.7+
pandas
numpy
scikit-learn

Installation
Clone or download this repository
Install dependencies:
bash
pip install pandas numpy scikit-learn
Ensure adult.csv.zip is in the same directory as the script

Usage
Run the complete pipeline with:
bash
python main.py
The script will execute all tasks sequentially and output:
Dataset statistics and overview
Cleaning verification
Feature engineering confirmation
Model training progress
Final performance evaluation matrix

Pipeline Tasks
Task 1: Dataset Understanding
Displays dataset dimensions (rows × columns)
Shows feature types and missing value overview via df.info()
Analyzes target variable distribution (income counts)
Calculates class imbalance percentages
Task 2: Data Cleaning
Detects hidden missing values: Scans all object-type columns for '?' entries
Converts to NaN: Replaces '?' with standard np.nan values
Mode Imputation: Fills missing values in:
workclass
occupation
native.country / native-country
Verification: Confirms zero remaining missing values
Task 3: Feature Engineering
Target Encoding: Converts income to binary (0 = <=50K, 1 = >50K)
Redundancy Removal: Drops text education column (keeps education.num / education-num)
Categorical Encoding: Label-encodes all remaining object-type features to numeric values
Task 4: Model Building
Features (X): All columns except income
Target (y): income column
Train/Test Split: 80% training, 20% testing (stratified, random_state=42)
Feature Scaling: StandardScaler applied to all features
Task 5: Performance Evaluation
Evaluates each model using:
Accuracy — Overall correctness
Precision — True positives / predicted positives
Recall — True positives / actual positives
F1 Score — Harmonic mean of precision and recall
ROC-AUC — Area under the receiver operating characteristic curve

Results
The final output displays a performance matrix comparing all five algorithms:
| Algorithm           | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| ------------------- | -------- | --------- | ------ | -------- | ------- |
| Logistic Regression | ~0.85    | ~0.75     | ~0.60  | ~0.67    | ~0.90   |
| Decision Tree       | ~0.81    | ~0.70     | ~0.60  | ~0.65    | ~0.78   |
| Random Forest       | ~0.86    | ~0.78     | ~0.62  | ~0.69    | ~0.91   |
| KNN                 | ~0.84    | ~0.75     | ~0.58  | ~0.65    | ~0.87   |
| SVM                 | ~0.85    | ~0.76     | ~0.59  | ~0.66    | ~0.90   |

Note: Actual values may vary slightly based on dataset version and random state.

Models Compared
| Model                   | Description                     | Best For                          |
| ----------------------- | ------------------------------- | --------------------------------- |
| **Logistic Regression** | Linear probabilistic classifier | Baseline, interpretability        |
| **Decision Tree**       | Rule-based tree structure       | Feature importance, visualization |
| **Random Forest**       | Ensemble of decision trees      | High accuracy, robustness         |
| **KNN**                 | Instance-based learning         | Simple, non-parametric            |
| **SVM**                 | Maximum margin classifier       | Complex boundaries                |

Key Design Decisions
Mode Imputation: Chosen for categorical missing values to preserve distribution
Label Encoding: Used over One-Hot to maintain feature space efficiency
StandardScaler: Applied uniformly for fair comparison across algorithms
Stratified Split: Ensures class balance is maintained in train/test sets
Probability=True for SVM: Required for ROC-AUC calculation

License
This project is for educational purposes. The Adult dataset is publicly available from the UCI Machine Learning Repository. 
