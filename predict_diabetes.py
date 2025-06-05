#!/usr/bin/env python
# coding: utf-8

"""
Diabetes Prediction Tool
=======================
This script loads the trained model and makes predictions on new patient data.
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path

def load_model_and_scaler(model_path, scaler_path):
    """Load the trained model and scaler from disk"""
    model = pickle.load(open(model_path, 'rb'))
    scaler = pickle.load(open(scaler_path, 'rb'))
    return model, scaler

def preprocess_data(data, features_with_zeros, reference_data=None):
    """Preprocess new data using the same steps as training data"""
    # Handle zero values
    for feature in features_with_zeros:
        if any(data[feature] == 0):
            data[feature] = data[feature].replace(0, np.nan)
            if reference_data is not None:
                median_value = reference_data[feature].median()
            else:
                # Default medians from the original dataset (pre-computed)
                default_medians = {
                    'Glucose': 117.0,
                    'BloodPressure': 72.0,
                    'SkinThickness': 23.0,
                    'Insulin': 30.5,
                    'BMI': 32.0
                }
                median_value = default_medians.get(feature, 0)
            data[feature].fillna(median_value, inplace=True)
    return data

def predict_diabetes(model, scaler, patient_data, features_with_zeros, reference_data=None):
    """Make diabetes prediction on new patient data"""
    # Preprocess the data
    processed_data = preprocess_data(
        patient_data.copy(), features_with_zeros, reference_data
    )
    
    # Standardize using the pre-fitted scaler
    scaled_data = scaler.transform(processed_data)
    
    # Make predictions
    prediction = model.predict(scaled_data)[0]
    probabilities = model.predict_proba(scaled_data)[0]
    
    return {
        'is_diabetic': bool(prediction),
        'prediction_label': 'Diabetic' if prediction else 'Non-Diabetic',
        'confidence': float(max(probabilities) * 100)
    }

def main():
    """Main function to demonstrate prediction"""
    # Create output directory if it doesn't exist
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    
    # Paths to saved model and scaler
    model_path = output_dir / 'diabetes_model_tuned.pkl'
    scaler_path = output_dir / 'scaler.pkl'
    
    # Features that might contain zeros
    features_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    
    # Check if model and scaler exist
    if not model_path.exists() or not scaler_path.exists():
        print("Error: Model files not found. Please run the training script first.")
        print(f"Looking for files at: {model_path} and {scaler_path}")
        return
    
    # Load the model and scaler
    model, scaler = load_model_and_scaler(model_path, scaler_path)
    
    # Example patient data
    patient_data = pd.DataFrame({
        'Pregnancies': [1],                  # Number of pregnancies
        'Glucose': [140],                    # Plasma glucose concentration
        'BloodPressure': [70],               # Diastolic blood pressure (mm Hg)
        'SkinThickness': [20],               # Triceps skin fold thickness (mm)
        'Insulin': [80],                     # 2-Hour serum insulin (mu U/ml)
        'BMI': [30.5],                       # Body mass index
        'DiabetesPedigreeFunction': [0.5],   # Diabetes pedigree function
        'Age': [45]                          # Age in years
    })
    
    # Interactive input option
    use_custom_data = input("Do you want to enter custom patient data? (y/n): ").lower() == 'y'
    
    if use_custom_data:
        try:
            print("\nPlease enter patient information:")
            patient_data = pd.DataFrame({
                'Pregnancies': [float(input("Number of pregnancies: "))],
                'Glucose': [float(input("Plasma glucose concentration: "))],
                'BloodPressure': [float(input("Diastolic blood pressure (mm Hg): "))],
                'SkinThickness': [float(input("Triceps skin fold thickness (mm): "))],
                'Insulin': [float(input("2-Hour serum insulin (mu U/ml): "))],
                'BMI': [float(input("Body mass index: "))],
                'DiabetesPedigreeFunction': [float(input("Diabetes pedigree function: "))],
                'Age': [float(input("Age in years: "))]
            })
        except ValueError:
            print("Invalid input. Using example data instead.")
    
    # Print the patient data
    print("\nPatient Data:")
    for col in patient_data.columns:
        print(f"{col}: {patient_data[col].values[0]}")
    
    # Get prediction
    result = predict_diabetes(model, scaler, patient_data, features_with_zeros)
    
    # Print results
    print("\nPrediction Results:")
    print(f"Prediction: {result['prediction_label']}")
    print(f"Confidence: {result['confidence']:.2f}%")
    
    # Provide additional context
    if result['is_diabetic']:
        print("\nThis indicates the patient is at risk for diabetes based on the provided features.")
        print("Recommendation: Further clinical evaluation is advised.")
    else:
        print("\nThis indicates the patient is likely not diabetic based on the provided features.")
        print("Recommendation: Continue regular health monitoring.")
    
    print("\nNote: This is a machine learning prediction and should not replace professional medical advice.")

if __name__ == "__main__":
    main()
