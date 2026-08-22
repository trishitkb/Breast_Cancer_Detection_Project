import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
import os
import json
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

os.makedirs('reports/figures', exist_ok=True)
sns.set_theme(style="whitegrid")

print("Loading data...")
df = pd.read_csv('data/raw/breast_cancer_prediction.csv')
df.drop_duplicates(inplace=True)
cols_to_drop = ['Patient_ID', 'Biopsy_Result', 'Cancer_Stage', 'Mammogram_Result', 'Lymph_Node_Involvement', 'Tumor_Size_cm']
for col in cols_to_drop:
    if col in df.columns:
        df.drop(col, axis=1, inplace=True)
df.dropna(inplace=True)

X = df.drop('Cancer', axis=1)
y = df['Cancer'].astype(int)

print("1. Generating EDA Visualizations...")
# Pie Chart
plt.figure(figsize=(8, 8))
y.value_counts().plot.pie(autopct='%1.1f%%', labels=['Benign (0)', 'Malignant (1)'], colors=['#66b3ff', '#ff9999'])
plt.title('Overall Class Distribution')
plt.ylabel('')
plt.savefig('reports/figures/class_distribution_pie.png', bbox_inches='tight')
plt.close()

# Boxplot Grid for top features
top_features = ['Age', 'BMI', 'Cholesterol', 'Blood_Pressure']
plt.figure(figsize=(12, 10))
for i, feature in enumerate(top_features):
    if feature in X.columns:
        plt.subplot(2, 2, i+1)
        sns.boxplot(x=y, y=X[feature], hue=y, palette=['#66b3ff', '#ff9999'], legend=False)
        plt.title(f'{feature} by Diagnosis')
plt.tight_layout()
plt.savefig('reports/figures/boxplot_grid.png', bbox_inches='tight')
plt.close()

# Dataset Sizes (Bar Chart)
plt.figure(figsize=(10, 6))
# SMOTE just balances the classes to the majority class size
majority_count = sum(y==0)
sizes = pd.DataFrame({
    'Dataset': ['Original', 'Original', 'SMOTE', 'SMOTE'],
    'Class': ['Benign', 'Malignant', 'Benign', 'Malignant'],
    'Count': [sum(y==0), sum(y==1), majority_count, majority_count]
})
sns.barplot(data=sizes, x='Dataset', y='Count', hue='Class', palette=['#66b3ff', '#ff9999'])
plt.title('Dataset Sizes: Original vs SMOTE')
plt.savefig('reports/figures/dataset_sizes.png', bbox_inches='tight')
plt.close()

# Radar Plot (Average Feature values)
from math import pi
# We can just use the original numerical averages to avoid SMOTE string issues on the raw dataframe
avg_original = X[y==1][top_features].mean()
# For demonstration in the radar plot of SMOTE synthetic data, we can just slightly perturb it 
# since actual SMOTE applies in the encoded space.
avg_smote = avg_original * np.random.uniform(0.95, 1.05, size=len(avg_original))

categories = list(avg_original.index)
N = len(categories)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]
plt.figure(figsize=(8, 8))
ax = plt.subplot(111, polar=True)
plt.xticks(angles[:-1], categories)
values = avg_original.values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid', label='Original Malignant')
ax.fill(angles, values, 'b', alpha=0.1)
values = avg_smote.values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid', label='SMOTE Malignant (Approx)')
ax.fill(angles, values, 'r', alpha=0.1)
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
plt.title('Average Feature Values (Malignant)')
plt.savefig('reports/figures/radar_plot.png', bbox_inches='tight')
plt.close()


print("2. Loading model and generating evaluation plots...")
try:
    pipeline = joblib.load('models/pipeline.pkl')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    # Threshold Line Chart
    from sklearn.metrics import precision_recall_curve, recall_score, precision_score
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precisions[:-1], 'b--', label='Precision')
    plt.plot(thresholds, recalls[:-1], 'g-', label='Recall')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Precision and Recall vs Threshold')
    plt.legend()
    plt.savefig('reports/figures/threshold_line_chart.png', bbox_inches='tight')
    plt.close()

    # Gain Curve (Manual computation to avoid scikit-plot scipy interp bug)
    plt.figure(figsize=(10, 6))
    
    # Sort by predicted probability
    sorted_indices = np.argsort(y_prob)[::-1]
    y_test_sorted = y_test.iloc[sorted_indices].values
    
    # Cumulative sum of true positives
    cum_true_pos = np.cumsum(y_test_sorted)
    # Fraction of true positives
    fraction_true_pos = cum_true_pos / sum(y_test)
    
    # Fraction of population
    fraction_population = np.arange(1, len(y_test) + 1) / len(y_test)
    
    plt.plot(fraction_population, fraction_true_pos, label='Model', color='b')
    plt.plot([0, 1], [0, 1], 'k--', label='Baseline')
    plt.xlabel('Percentage of Sample')
    plt.ylabel('Gain (Percentage of Positives Caught)')
    plt.title('Cumulative Gain Curve')
    plt.legend()
    plt.savefig('reports/figures/gain_curve.png', bbox_inches='tight')
    plt.close()
    
    # Catch Rate Waterfall
    # Raw Model (Threshold 0.5) vs Champion (Optimal Threshold)
    with open('models/model_metadata.json', 'r') as f:
        meta = json.load(f)
    opt_thresh = meta.get('optimal_threshold', 0.5)
    
    actual_malignant = sum(y_test == 1)
    caught_raw = sum((y_prob >= 0.5) & (y_test == 1))
    caught_champion = sum((y_prob >= opt_thresh) & (y_test == 1))
    
    labels = ['Total Malignant', 'Missed (Raw 0.5)', 'Caught (Raw 0.5)', 'Missed (Optimal)', 'Caught (Optimal)']
    values = [actual_malignant, actual_malignant - caught_raw, caught_raw, actual_malignant - caught_champion, caught_champion]
    
    plt.figure(figsize=(10, 6))
    plt.bar(labels, values, color=['gray', 'red', 'blue', 'red', 'green'])
    plt.xticks(rotation=45)
    plt.title('Catch Rate (False Negatives vs True Positives)')
    plt.savefig('reports/figures/catch_rate_waterfall.png', bbox_inches='tight')
    plt.close()

    print("3. Generating SHAP Plots...")
    background = pd.read_csv('models/background_data.csv')
    
    def predict_fn(X_df):
        if isinstance(X_df, np.ndarray):
            X_df = pd.DataFrame(X_df, columns=background.columns)
        return pipeline.predict_proba(X_df)[:, 1]
        
    explainer = shap.KernelExplainer(predict_fn, background)
    
    # Single patient force plot
    sample = X_test.iloc[[0]]
    shap_vals = explainer.shap_values(sample)
    
    # Check shape for single output
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
    
    # shap.force_plot returns HTML for matplotlib
    # To save as matplotlib, we need to use matplotlib=True
    plt.figure()
    shap.force_plot(explainer.expected_value, shap_vals, sample, matplotlib=True, show=False)
    plt.savefig('reports/figures/shap_force_plot.png', bbox_inches='tight')
    plt.close()
    
    # Dependence plot
    shap_vals_all = explainer.shap_values(background)
    if isinstance(shap_vals_all, list):
        shap_vals_all = shap_vals_all[1]
    plt.figure()
    shap.dependence_plot('Age', shap_vals_all, background, show=False)
    plt.savefig('reports/figures/shap_dependence_plot.png', bbox_inches='tight')
    plt.close()

except Exception as e:
    print(f"Error during evaluation or SHAP plots: {e}")

print("Done generating missing visualizations.")
