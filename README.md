# Diabetes Classification Project

This project implements machine learning models to predict diabetes based on various health indicators. It includes data visualization, model training, and a prediction tool for new patient data.

## Project Overview

The project uses the Pima Indians Diabetes dataset to train and evaluate several machine learning models to predict diabetes. The models implemented include:
- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)

## Directory Structure

```
.
├── diabetes.csv                   # Dataset file
├── diabetic_classification_simple.py  # Main training script
├── predict_diabetes.py            # Prediction script for new patients
├── run_diabetes_study.bat         # Batch file to run the scripts
├── output/                        # Output directory for models and visualizations
│   ├── confusion_matrix.png       # Confusion matrix visualization
│   ├── correlation_heatmap.png    # Feature correlation heatmap
│   ├── diabetes_model_tuned.pkl   # Trained best model file
│   ├── feature_distributions.png  # Feature distribution plots
│   ├── feature_importance.png     # Feature importance visualization
│   ├── feature_pairplot.png       # Pairplot of top features
│   ├── model_comparison.png       # Model performance comparison
│   ├── outcome_distribution.png   # Target variable distribution
│   ├── prediction_probability.png # Sample prediction probability
│   └── scaler.pkl                 # Feature scaler object
└── README.md                      # This file
```

## Requirements

- Python 3.8 or higher (compatible with Python 3.12)
- Required Python libraries:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
  - pickle

## How to Run

### Option 1: Using the Batch File (Windows)

1. Open Command Prompt in the project directory
2. Run the batch file:
   ```
   run_diabetes_study.bat
   ```
3. Choose from the menu options:
   - Option 1: Run the diabetes classification study (trains models)
   - Option 2: Make predictions using the trained model
   - Option 3: Exit

### Option 2: Running Python Scripts Directly

1. To train models and generate visualizations:
   ```
   python diabetic_classification_simple.py
   ```

2. To make predictions on new patient data:
   ```
   python predict_diabetes.py
   ```

## Model Performance

The script evaluates multiple machine learning models and selects the best one based on F1 score. Performance metrics include:
- Accuracy
- Precision
- Recall
- F1 Score

## Visualizations

The project generates several visualizations to help understand the data and model performance:
- Feature distributions
- Correlation heatmap
- Model comparison
- Confusion matrix
- Feature importance (for Random Forest)
- Pairplot of key features

## Making Predictions

The `predict_diabetes.py` script allows you to input patient data and receive a prediction about diabetes risk. The prediction includes:
- Classification outcome (Diabetic or Non-Diabetic)
- Prediction confidence

## Data Preprocessing

The dataset undergoes several preprocessing steps:
- Handling of zero values in certain features
- Standardization of features
- Train-test split for model evaluation

## Author

This project was developed as a case study for diabetes prediction using machine learning techniques.
