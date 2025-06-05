#!/usr/bin/env python
# coding: utf-8

"""
Simplified Diabetic Classification Case Study
===============================================
This script implements the diabetic classification workflow
with simpler visualization code to work with Python 3.12
"""

# 1. Loading libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import pickle
from pathlib import Path
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Set style for visualizations - for newer matplotlib/seaborn versions 
try: 
    plt.style.use('seaborn-v0_8-whitegrid') 
except: 
    plt.style.use('default') 
    sns.set_theme(style='whitegrid') 

# Set figure aesthetics
sns.set(font_scale=1.2)
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14

# Create output directory
output_dir = Path('output')
output_dir.mkdir(exist_ok=True)

print("=" * 50)
print("DIABETIC CLASSIFICATION CASE STUDY")
print("=" * 50)

# Load dataset
print("\nLoading dataset...")
data = pd.read_csv('diabetes.csv')
print(f"Dataset shape: {data.shape}")
print("\nFirst 5 rows:")
print(data.head())

# Data visualizations
print("\nCreating data visualizations...")
# Distribution of target variable
plt.figure(figsize=(10, 6))
outcome_counts = data['Outcome'].value_counts()
sns.barplot(x=outcome_counts.index, y=outcome_counts.values, palette='viridis')
plt.title('Distribution of Diabetes Outcome')
plt.xlabel('Outcome (0: Non-diabetic, 1: Diabetic)')
plt.ylabel('Count')
# Add count labels on top of bars
for i, v in enumerate(outcome_counts.values):
    plt.text(i, v + 5, str(v), ha='center')
plt.savefig(output_dir / 'outcome_distribution.png')
plt.close()

# Feature distributions by outcome
plt.figure(figsize=(15, 12))
for i, feature in enumerate(data.columns[:-1]):  # Exclude 'Outcome'
    plt.subplot(3, 3, i+1)
    sns.histplot(data=data, x=feature, hue='Outcome', kde=True, palette='viridis')
    plt.title(f'Distribution of {feature}')
plt.tight_layout()
plt.savefig(output_dir / 'feature_distributions.png')
plt.close()

# Correlation heatmap
plt.figure(figsize=(12, 10))
correlation_matrix = data.corr()
mask = np.triu(correlation_matrix)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', mask=mask)
plt.title('Correlation Matrix of Features')
plt.savefig(output_dir / 'correlation_heatmap.png')
plt.close()

# Data preprocessing
print("\nPreprocessing data...")
# Handle zero values that should be considered as missing
features_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for feature in features_with_zeros:
    # Count zeros before replacement
    zeros_count = (data[feature] == 0).sum()
    if zeros_count > 0:
        print(f"- {feature}: {zeros_count} zeros found")
    
    # Replace zeros with NaN and fill with median
    data[feature] = data[feature].replace(0, np.nan)
    median_value = data[feature].median()
    data[feature].fillna(median_value, inplace=True)

# Separate features and target
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
print("\nSplitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
print(f"Training set: {X_train.shape}, Testing set: {X_test.shape}")

# Train models
print("\nTraining models...")
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(probability=True)
}

results = {}
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Metrics
    results[name] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }

# Display results
print("\nModel Results:")
results_df = pd.DataFrame(results).T
print(results_df)

# Visualize model performance metrics
metrics = ['accuracy', 'precision', 'recall', 'f1']
plt.figure(figsize=(14, 10))
for i, metric in enumerate(metrics):
    plt.subplot(2, 2, i+1)
    sns.barplot(x=results_df.index, y=results_df[metric], palette='viridis')
    plt.title(f'Model Comparison: {metric.capitalize()}')
    plt.xticks(rotation=45)
    plt.ylim(0, 1.0)
    # Add value labels on bars
    for j, v in enumerate(results_df[metric]):
        plt.text(j, v + 0.02, f'{v:.3f}', ha='center')
plt.tight_layout()
plt.savefig(output_dir / 'model_comparison.png')
plt.close()

# Create confusion matrix for best model
y_pred = models[str(results_df['f1'].idxmax())].predict(X_test)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title(f'Confusion Matrix - {results_df["f1"].idxmax()}')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig(output_dir / 'confusion_matrix.png')
plt.close()

# Select best model
best_model_name = results_df['f1'].idxmax()
best_model = models[str(best_model_name)]
print(f"\nBest model: {best_model_name} (F1 Score: {results_df.loc[best_model_name, 'f1']:.4f})")

# Save model
print("\nSaving best model...")
pickle.dump(best_model, open(output_dir / 'diabetes_model_tuned.pkl', 'wb'))
pickle.dump(scaler, open(output_dir / 'scaler.pkl', 'wb'))

# Visualize feature importance if best model is Random Forest
if isinstance(best_model, RandomForestClassifier):
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x=importances[indices], y=[data.columns[i] for i in indices if i < len(data.columns)-1], palette='viridis')
    plt.title('Feature Importance in Random Forest Model')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_importance.png')
    plt.close()

# Create a pairplot of key features
print("\nGenerating pairplot of key features...")
# Select the most important features based on correlation with outcome
corr_with_outcome = abs(data.corr()['Outcome']).sort_values(ascending=False)
top_features = corr_with_outcome.index[1:5]  # Skip 'Outcome' itself
selected_data = data[list(top_features) + ['Outcome']]

# Create pairplot
sns.pairplot(selected_data, hue='Outcome', palette='viridis')
plt.suptitle('Pairplot of Top Correlated Features', y=1.02)
plt.savefig(output_dir / 'feature_pairplot.png')
plt.close()

# Example prediction
print("\nMaking a sample prediction...")
new_data = pd.DataFrame({
    'Pregnancies': [6],
    'Glucose': [148],
    'BloodPressure': [72],
    'SkinThickness': [35],
    'Insulin': [0],
    'BMI': [33.6],
    'DiabetesPedigreeFunction': [0.627],
    'Age': [50]
})

# Handle missing values if any
for feature in features_with_zeros:
    if any(new_data[feature] == 0):
        new_data[feature] = new_data[feature].replace(0, np.nan)
        new_data[feature].fillna(data[feature].median(), inplace=True)

# Make prediction
new_data_scaled = scaler.transform(new_data)
prediction = best_model.predict(new_data_scaled)
prediction_probability = best_model.predict_proba(new_data_scaled)

print(f"\nSample data: {new_data.iloc[0].to_dict()}")
print(f"Prediction: {'Diabetic' if prediction[0] == 1 else 'Non-Diabetic'}")
print(f"Confidence: {max(prediction_probability[0]) * 100:.2f}%")

# Visualize prediction probability
plt.figure(figsize=(10, 6))
probabilities = prediction_probability[0]
classes = ['Non-Diabetic', 'Diabetic']
colors = ['skyblue', 'coral']
sns.barplot(x=classes, y=probabilities, palette=colors)
plt.title('Prediction Probability')
plt.ylabel('Probability')
plt.ylim(0, 1)
for i, prob in enumerate(probabilities):
    plt.text(i, prob + 0.02, f'{prob:.4f}', ha='center')
plt.savefig(output_dir / 'prediction_probability.png')
plt.close()

print("\n" + "=" * 50)
print("CASE STUDY COMPLETE")
print("=" * 50)
print(f"Model saved to {output_dir}")
