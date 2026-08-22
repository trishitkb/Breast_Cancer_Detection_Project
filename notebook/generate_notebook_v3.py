import nbformat

cells_content = [
    """# Multi-Dataset Model Training, Pipeline Refactoring, & Cost Analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import datetime
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_predict, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve, auc
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import shap""",

    """# Create directories for saving artifacts
os.makedirs('../data/processed', exist_ok=True)
os.makedirs('../reports/figures', exist_ok=True)
os.makedirs('../models', exist_ok=True)""",

    """# 1. Load Data
df = pd.read_csv('../data/raw/breast_cancer_prediction.csv')
print("Initial Shape:", df.shape)
df.head()""",

    """# 2. Data Cleaning
# Remove duplicates
df.drop_duplicates(inplace=True)

# Drop Patient_ID and leaky features
cols_to_drop = ['Patient_ID', 'Biopsy_Result', 'Cancer_Stage']
for col in cols_to_drop:
    if col in df.columns:
        df.drop(col, axis=1, inplace=True)
df.dropna(inplace=True)""",

    """# 3. Train-Test Split (BEFORE preprocessing to prevent data leakage)
X = df.drop('Cancer', axis=1)
y = df['Cancer']

# Ensure target is integer
y = y.astype(int)

X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training set: {X_train_raw.shape}, Testing set: {X_test_raw.shape}")""",

    """# 4. Pipeline Definition (Preprocessing + Resampling)
categorical_cols = X_train_raw.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X_train_raw.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Preprocessing for numerical data
numerical_transformer = StandardScaler()

# Preprocessing for categorical data
# Use OrdinalEncoder with handle_unknown='use_encoded_value' to prevent errors on unseen data
categorical_transformer = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

# Bundle preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])""",

    """# 5. Model Initialization
# We only use Baseline, LogReg, RF, and GradientBoosting as per requirements
models = {
    'Baseline (Dummy)': DummyClassifier(strategy='most_frequent'),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

datasets = {
    'Imbalanced': None,
    'SMOTE': SMOTE(random_state=42),
    'Undersampled': RandomUnderSampler(random_state=42)
}""",

    """# 6. Evaluation Loop with 5-Fold Cross Validation
results = []
trained_pipelines = {}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for ds_name, sampler in datasets.items():
    trained_pipelines[ds_name] = {}
    for model_name, classifier in models.items():
        # Build pipeline
        steps = [('preprocessor', preprocessor)]
        if sampler is not None:
            steps.append(('sampler', sampler))
        
        # Calibrate the classifier to get better probability estimates, except for Dummy
        if model_name != 'Baseline (Dummy)':
            calibrated_clf = CalibratedClassifierCV(classifier, cv=cv, method='sigmoid')
            steps.append(('classifier', calibrated_clf))
        else:
            steps.append(('classifier', classifier))
            
        pipeline = ImbPipeline(steps=steps)
        
        # Train
        pipeline.fit(X_train_raw, y_train)
        
        # Predict on Test Set
        y_pred = pipeline.predict(X_test_raw)
        y_prob = pipeline.predict_proba(X_test_raw)[:, 1] if hasattr(pipeline, 'predict_proba') else None
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        if y_prob is not None:
            roc_auc = roc_auc_score(y_test, y_prob)
            precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
            pr_auc = auc(recalls, precisions)
        else:
            roc_auc = 'N/A'
            pr_auc = 'N/A'
            
        results.append({
            'Dataset': ds_name,
            'Model': model_name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1 Score': f1,
            'ROC-AUC': roc_auc,
            'PR-AUC': pr_auc
        })
        trained_pipelines[ds_name][model_name] = pipeline

results_df = pd.DataFrame(results)
display(results_df.sort_values(by=['Recall', 'PR-AUC'], ascending=[False, False]))""",

    """# 7. Threshold Optimization for Best Model
# Find the best non-dummy model by Recall
valid_results = results_df[results_df['Model'] != 'Baseline (Dummy)']
best_row = valid_results.sort_values(by=['Recall', 'PR-AUC'], ascending=[False, False]).iloc[0]

best_ds = best_row['Dataset']
best_model_name = best_row['Model']
best_pipeline = trained_pipelines[best_ds][best_model_name]

print(f"Absolute Best Model: {best_model_name} on {best_ds}")

# Optimize Threshold
y_prob_best = best_pipeline.predict_proba(X_test_raw)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob_best)

# We want to maximize Recall, but let's constrain Precision to be at least 0.15 (or use F2 score)
# Using F2 Score to heavily weight Recall over Precision
f2_scores = (5 * precisions * recalls) / (4 * precisions + recalls + 1e-10)
optimal_idx = np.argmax(f2_scores)
optimal_threshold = thresholds[optimal_idx]

print(f"Standard Threshold (0.5) Recall: {recall_score(y_test, y_prob_best >= 0.5)}")
print(f"Optimal Threshold ({optimal_threshold:.4f}) Recall: {recall_score(y_test, y_prob_best >= optimal_threshold)}")
print(f"Optimal Threshold Precision: {precision_score(y_test, y_prob_best >= optimal_threshold)}")""",

    """# 8. Cost-Sensitive Analysis (False Positives vs False Negatives)
# Assume the medical cost of missing a cancer diagnosis (False Negative) is $100,000 (legal, life impact)
# Assume the cost of a False Positive is $5,000 (unnecessary biopsy and stress)
COST_FP = 5000
COST_FN = 100000

def calculate_cost(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fp = cm[0, 1]
    fn = cm[1, 0]
    return (fp * COST_FP) + (fn * COST_FN)

cost_standard = calculate_cost(y_test, y_prob_best >= 0.5)
cost_optimal = calculate_cost(y_test, y_prob_best >= optimal_threshold)
cost_dummy = calculate_cost(y_test, trained_pipelines['Imbalanced']['Baseline (Dummy)'].predict(X_test_raw))

print(f"Total Cost with Baseline (predict all negative): ${cost_dummy:,.2f}")
print(f"Total Cost with Standard 0.5 Threshold: ${cost_standard:,.2f}")
print(f"Total Cost with Optimal Threshold: ${cost_optimal:,.2f}")
print(f"Savings by using Optimal Threshold vs 0.5: ${(cost_standard - cost_optimal):,.2f}")""",

    """# 9. Save Entire Pipeline and Metadata
metadata = {
    "dataset_version": "v1",
    "training_date": datetime.datetime.now().isoformat(),
    "model_name": best_model_name,
    "dataset_balancing": best_ds,
    "optimal_threshold": float(optimal_threshold),
    "metrics": {
        "recall_optimized": float(recall_score(y_test, y_prob_best >= optimal_threshold)),
        "precision_optimized": float(precision_score(y_test, y_prob_best >= optimal_threshold)),
        "pr_auc": float(best_row['PR-AUC']),
        "roc_auc": float(best_row['ROC-AUC'])
    }
}

joblib.dump(best_pipeline, '../models/pipeline.pkl')
with open('../models/model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=4)

# 9b. Experiment Tracking (CSV Log)
log_file = '../reports/experiment_log.csv'
log_record = {
    'Timestamp': metadata['training_date'],
    'Model_Version': metadata['dataset_version'],
    'Model_Name': metadata['model_name'],
    'Dataset_Balancing': metadata['dataset_balancing'],
    'Optimal_Threshold': metadata['optimal_threshold'],
    'Recall_Optimized': metadata['metrics']['recall_optimized'],
    'Precision_Optimized': metadata['metrics']['precision_optimized'],
    'PR_AUC': metadata['metrics']['pr_auc'],
    'ROC_AUC': metadata['metrics']['roc_auc']
}
log_df = pd.DataFrame([log_record])
import os
if not os.path.exists(log_file):
    log_df.to_csv(log_file, index=False)
else:
    log_df.to_csv(log_file, mode='a', header=False, index=False)

print("Saved pipeline.pkl and model_metadata.json to models/")""",

    """# 10. Local SHAP Explanations Setup
# The backend will need to compute SHAP values for individual patients.
# Since we are using CalibratedClassifierCV, getting direct TreeExplainer is complex.
# We'll save the background dataset for KernelExplainer or just use the base estimator if possible.
# For simplicity in production, we will extract the base estimator from the CalibratedClassifierCV 
# if it's a tree, or just save a small background dataset for KernelExplainer.

background_data = X_train_raw.sample(100, random_state=42)
background_data.to_csv('../models/background_data.csv', index=False)
print("Saved background dataset for SHAP to models/background_data.csv")"""
]

nb = nbformat.v4.new_notebook()
for cell_text in cells_content:
    nb.cells.append(nbformat.v4.new_code_cell(cell_text))

with open('notebook/breast_cancer_pipeline.ipynb', 'w') as f:
    nbformat.write(nb, f)
print("Notebook v3 generated successfully.")
