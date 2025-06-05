# Case Study: Diabetics Classification Problem

## 1. Loading Libraries

This case study utilizes the following Python libraries:
- **Data Management**: pandas for data manipulation and analysis
- **Mathematical Computation**: numpy for numerical operations
- **Data Transformation**: z-score standardization
- **Data Visualization**: seaborn and matplotlib.pyplot for creating informative visualizations
- **Machine Learning**: scikit-learn for model selection, training, evaluation, and splitting datasets

```python
# Import required libraries
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
```

## 2. Loading Dataset

The study uses the Pima Indians Diabetes Database, which contains diagnostic measurements and diabetes status for a population of female Pima Indians.

### Dataset Description

The Pima Indians Diabetes Dataset was originally from the National Institute of Diabetes and Digestive and Kidney Diseases. It contains the following features:
- **Pregnancies**: Number of pregnancies the patient has had
- **Glucose**: Plasma glucose concentration at 2 hours in an oral glucose tolerance test
- **BloodPressure**: Diastolic blood pressure (mm Hg)
- **SkinThickness**: Triceps skin fold thickness (mm)
- **Insulin**: 2-Hour serum insulin (mu U/ml)
- **BMI**: Body mass index (weight in kg/(height in m)²)
- **DiabetesPedigreeFunction**: A function that scores likelihood of diabetes based on family history
- **Age**: Age in years
- **Outcome**: Class variable (0: non-diabetic, 1: diabetic)

The dataset consists of 768 observations, making it a manageable size for demonstration purposes while still providing enough data for meaningful analysis.

```python
# Load the dataset
data = pd.read_csv('pima-indians-diabetes.csv')

# Display the first few rows
print(data.head())
```

## 3. Data Summary

### Descriptive Statistics

```python
# Summary of data
print(data.info())

# Descriptive statistics
print(data.describe())

# Count of samples by class
print(data['Outcome'].value_counts())
```

This provides:
- Overview of dataset structure
- Statistical summary including mean, median, min, max values
- Count of diabetic vs. non-diabetic samples

## 4. Data Visualization

### Class Distribution

```python
# Plot the class distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='Outcome', data=data)
plt.title('Class Distribution: Diabetic vs Non-Diabetic')
plt.xlabel('Outcome (0 = Non-Diabetic, 1 = Diabetic)')
plt.ylabel('Count')
plt.show()
```

### Outlier Detection

```python
# Box plots for feature distribution and outlier detection
plt.figure(figsize=(15, 10))
data.drop('Outcome', axis=1).boxplot()
plt.title('Box Plots for Feature Distribution')
plt.show()

# Scatter plots for selected features
plt.figure(figsize=(12, 10))
sns.pairplot(data, hue='Outcome')
plt.suptitle('Scatter Plot Matrix for Feature Relationships')
plt.show()
```

### Feature Correlation Analysis

Understanding the relationships between features is critical for feature selection and interpretation:

```python
# Compute correlation matrix
corr_matrix = data.corr()

# Plot heatmap of correlations
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.show()

# Correlation with target variable
target_corr = corr_matrix['Outcome'].sort_values(ascending=False)
print("\nFeature correlation with diabetes outcome:")
print(target_corr)

# Plot features by correlation with target
plt.figure(figsize=(10, 6))
target_corr.drop('Outcome').plot(kind='bar')
plt.title('Correlation with Diabetes Outcome')
plt.ylabel('Correlation Coefficient')
plt.tight_layout()
plt.show()
```

This analysis reveals which features are most strongly associated with diabetes outcomes, showing that:
1. Glucose level has the strongest positive correlation with diabetes
2. Several features have moderate correlations, indicating their predictive value
3. Some features show correlation with each other, suggesting potential multicollinearity

## 5. Data Preprocessing

### Data Cleaning

```python
# Check for missing values
print(data.isnull().sum())

# Handle zero values that should be considered as missing values
features_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for feature in features_with_zeros:
    # Replace zeros with NaN for selected columns
    data[feature] = data[feature].replace(0, np.nan)
    # Fill NaN with median values
    data[feature].fillna(data[feature].median(), inplace=True)

# Check for outliers and handle them using IQR method
for feature in X.columns:
    Q1 = data[feature].quantile(0.25)
    Q3 = data[feature].quantile(0.75)
    IQR = Q3 - Q1
    
    # Define bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Cap outliers
    data[feature] = np.where(data[feature] < lower_bound, lower_bound, data[feature])
    data[feature] = np.where(data[feature] > upper_bound, upper_bound, data[feature])

print("After outlier treatment - descriptive statistics:")
print(data.describe())
```

### Feature Standardization

```python
# Separate features and target
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Z-score standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Display the result
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
print(X_scaled_df.head())
```

## 6. Split the Dataset

```python
# Split the data into training and testing sets (80:20)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape}")
print(f"Testing set size: {X_test.shape}")
```

## 7. Choosing ML Algorithms

For this classification problem, we'll evaluate several common algorithms:

1. **Logistic Regression**: A linear model that estimates the probability of a binary outcome. It's simple, interpretable, and often serves as a good baseline for binary classification tasks.

2. **Decision Tree**: A non-parametric method that creates a model resembling a tree structure. Decision trees can capture non-linear relationships and are easy to interpret, making them valuable in medical contexts where explainability is important.

3. **Random Forest**: An ensemble method that builds multiple decision trees and merges their predictions. This approach typically improves accuracy and reduces overfitting compared to single decision trees.

4. **Support Vector Machine (SVM)**: A powerful algorithm that finds the optimal hyperplane to separate classes. SVMs can handle high-dimensional data effectively and can capture complex relationships through kernel functions.

Each algorithm has strengths and weaknesses that make them suitable for different aspects of medical data analysis. By comparing their performance, we can select the most appropriate model for diabetes prediction.

## 8. Training the Model

```python
# Dictionary to store models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(probability=True)
}

# Train and evaluate each model
results = {}

for name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate evaluation metrics
    results[name] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }
    
    # Print confusion matrix and classification report
    print(f"\nResults for {name}:")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    print(f"Classification Report:\n{classification_report(y_test, y_pred)}")

# Cross-validation for more robust performance estimation
from sklearn.model_selection import cross_val_score, KFold

print("\nPerforming cross-validation:")
cv_results = {}
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    cv_scores = cross_val_score(model, X_scaled, y, cv=kf, scoring='f1')
    cv_results[name] = {
        'mean_f1': cv_scores.mean(),
        'std_f1': cv_scores.std()
    }
    print(f"{name} - Mean F1: {cv_scores.mean():.4f}, Std: {cv_scores.std():.4f}")

# Create DataFrame with cross-validation results
cv_df = pd.DataFrame(cv_results).T
print("\nCross-validation results:")
print(cv_df)

# Compare models
results_df = pd.DataFrame(results).T
print("\nModel Comparison:")
print(results_df)

# Select the best model based on F1 score
best_model_name = results_df['f1'].idxmax()
best_model = models[best_model_name]
print(f"\nBest model: {best_model_name}")
```

### Feature Importance Analysis

Understanding which features contribute most to the prediction is crucial for both model refinement and medical insights:

```python
# Extract feature importance if the best model supports it
if best_model_name in ['Random Forest', 'Decision Tree']:
    # For tree-based models
    importances = best_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    # Plot feature importances
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
    plt.title(f'Feature Importance for {best_model_name}')
    plt.tight_layout()
    plt.show()
    
    print("Top features by importance:")
    print(feature_importance_df)
elif best_model_name == 'Logistic Regression':
    # For Logistic Regression, coefficients indicate feature importance
    coefficients = best_model.coef_[0]
    feature_importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': coefficients
    }).sort_values('Coefficient', ascending=False)
    
    # Plot coefficients
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Coefficient', y='Feature', data=feature_importance_df)
    plt.title('Logistic Regression Coefficients')
    plt.tight_layout()
    plt.show()
    
    print("Feature coefficients:")
    print(feature_importance_df)
```

This analysis reveals which medical factors are most predictive of diabetes, providing valuable clinical insights alongside the machine learning implementation.

### Save the Trained Model

```python
# Save the best model and scaler for future inference
pickle.dump(best_model, open('diabetes_model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))
```

## 9. Inferencing

Inference in machine learning refers to using a trained model to make predictions on new, unseen data.

### Hyperparameter Tuning

Before final deployment, we can optimize the best model's hyperparameters:

```python
from sklearn.model_selection import GridSearchCV

# Choose hyperparameters to tune based on the best model
if best_model_name == 'Random Forest':
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
elif best_model_name == 'Logistic Regression':
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga']
    }
elif best_model_name == 'SVM':
    param_grid = {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto', 0.1, 1],
        'kernel': ['linear', 'rbf']
    }
elif best_model_name == 'Decision Tree':
    param_grid = {
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['gini', 'entropy']
    }

# Perform grid search
grid_search = GridSearchCV(
    estimator=best_model,
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    verbose=1,
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

# Print best parameters and score
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best F1 score: {grid_search.best_score_:.4f}")

# Update the best model with tuned hyperparameters
best_model_tuned = grid_search.best_estimator_

# Evaluate tuned model
y_pred_tuned = best_model_tuned.predict(X_test)
print("\nTuned model performance:")
print(f"F1 Score: {f1_score(y_test, y_pred_tuned):.4f}")
print(f"Classification Report:\n{classification_report(y_test, y_pred_tuned)}")

# Save the tuned model for deployment
pickle.dump(best_model_tuned, open('diabetes_model_tuned.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))
```

### Making Predictions on New Data

```python
# Load the saved model and scaler
loaded_model = pickle.load(open('diabetes_model.pkl', 'rb'))
loaded_scaler = pickle.load(open('scaler.pkl', 'rb'))

# Define new, unseen data (example)
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

# Standardize the new data using the saved scaler
new_data_scaled = loaded_scaler.transform(new_data)

# Make a prediction
prediction = loaded_model.predict(new_data_scaled)
prediction_probability = loaded_model.predict_proba(new_data_scaled)

print(f"Prediction: {'Diabetic' if prediction[0] == 1 else 'Non-Diabetic'}")
print(f"Confidence: {max(prediction_probability[0]) * 100:.2f}%")
```

## Conclusion

This case study demonstrates a complete machine learning workflow for diabetic classification:

1. Data loading and exploration
2. Data preprocessing, including handling missing values and standardization
3. Model training and evaluation using multiple algorithms
4. Model selection based on performance metrics
5. Saving the model for future use
6. Making predictions on new data

The best performing model can be used in healthcare settings to assist in early diabetes diagnosis, potentially leading to better patient outcomes through timely intervention.

## 10. Model Deployment and Implementation

Taking the model from development to production involves several additional considerations:

### Deployment Options

```python
# Example of packaging the model for a REST API using Flask
from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model and scaler
model = pickle.load(open('diabetes_model_tuned.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
features_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from POST request
    data = request.get_json()
    
    # Convert to DataFrame
    df = pd.DataFrame(data, index=[0])
    
    # Process missing values
    for feature in features_with_zeros:
        if any(df[feature] == 0):
            df[feature] = df[feature].replace(0, np.nan)
            # Note: In production we would use pre-computed medians from training data
            df[feature].fillna(75, inplace=True)  # Example median value
    
    # Standardize the data
    scaled_data = scaler.transform(df)
    
    # Make prediction
    prediction = model.predict(scaled_data)[0]
    probability = model.predict_proba(scaled_data)[0].max()
    
    # Return prediction
    return jsonify({
        'prediction': int(prediction),
        'probability': float(probability),
        'label': 'Diabetic' if prediction == 1 else 'Non-Diabetic'
    })

if __name__ == '__main__':
    app.run(debug=True)
```

### Implementation Considerations

1. **Monitoring Performance**: Establish a system to monitor model performance over time as medical understanding and patient populations evolve.

2. **Retraining Strategy**: Define criteria for when to retrain the model with new data to maintain accuracy.

3. **Interpretability Tools**: Implement tools like SHAP (SHapley Additive exPlanations) values for explaining individual predictions to healthcare providers.

4. **Feedback Loop**: Create a mechanism for clinicians to provide feedback on predictions to improve future iterations.

5. **Regulatory Compliance**: Ensure the deployed model complies with healthcare regulations like HIPAA in the US or similar regulations in other countries.

6. **Documentation**: Maintain thorough documentation of the model development process, performance metrics, and limitations for regulatory and maintenance purposes.

By addressing these considerations, the diabetes classification model can be effectively integrated into clinical workflows, providing valuable decision support while maintaining appropriate clinical governance.
