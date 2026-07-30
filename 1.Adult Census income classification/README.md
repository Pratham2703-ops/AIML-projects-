Adult Census Income Classification

 Objective
Build a machine learning model to predict whether an individual's annual income exceeds $50,000 based on demographic and employment data from the 1994 U.S. Census.

 Dataset
Source: UCI Machine Learning Repository / Kaggle
Size: ~32,000 records
Features: Age, Education, Occupation, Marital Status, Race, Sex, Capital Gain/Loss, Hours per week, etc.
Target: Income — >50K or <=50K
 Methodology
Data Cleaning — Handle missing values, fix inconsistencies
Exploratory Data Analysis (EDA) — Visualize distributions, correlations
Feature Engineering — Encode categorical variables, scale numerical features
Model Selection — Compare Logistic Regression, Random Forest, XGBoost, SVM
Evaluation — Accuracy, Precision, Recall, F1-Score, ROC-AUC

 Results
Table
Model	Accuracy	F1-Score
Logistic Regression	~82%	~0.65
Random Forest	~86%	~0.72
XGBoost	~87%	~0.74
 How to Run
bash
cd Adult-Census-Income-Classification/
pip install -r requirements.txt
python train.py
# or open notebook.ipynb
 Files
data/ — Raw and processed datasets
notebook.ipynb — Full analysis and modeling notebook
train.py — Training script
model.pkl — Saved trained model
requirements.txt — Python dependencies
 Key Takeaways
Feature engineering (especially education & occupation encoding) significantly impacts performance
Tree-based models outperform linear models on this tabular dataset
Class imbalance handling (SMOTE/undersampling) improves recall
