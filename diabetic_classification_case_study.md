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

# Check for outliers and handle them if necessary
# For example, we can use IQR method to detect and cap outliers
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

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Support Vector Machine (SVM)

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

# Compare models
results_df = pd.DataFrame(results).T
print("\nModel Comparison:")
print(results_df)

# Select the best model based on F1 score
best_model_name = results_df['f1'].idxmax()
best_model = models[best_model_name]
print(f"\nBest model: {best_model_name}")
```

### Save the Trained Model

```python
# Save the best model and scaler for future inference
pickle.dump(best_model, open('diabetes_model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))
```

## 9. Inferencing

Inference in machine learning refers to using a trained model to make predictions on new, unseen data.

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
