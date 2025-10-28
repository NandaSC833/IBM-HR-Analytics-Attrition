# scripts/data_preprocessing.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

def load_data(path):
    df = pd.read_csv(path)
    print(f"✅ Data loaded successfully with shape {df.shape}")
    return df

def clean_data(df):
    df = df.drop(['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours'], axis=1)
    
    le = LabelEncoder()
    for col in df.select_dtypes('object').columns:
        df[col] = le.fit_transform(df[col])
    
    print("✅ Data cleaning & encoding completed.")
    return df

if __name__ == "__main__":
    input_path = r"C:\Users\Nanda Chowgle\OneDrive\Desktop\IBM HR Analytics Employee Attrition & Performance\data\WA_Fn-UseC_-HR-Employee-Attrition (1).csv"
    output_path = r"C:\Users\Nanda Chowgle\OneDrive\Desktop\IBM HR Analytics Employee Attrition & Performance\data\cleaned_hr_data.csv"
    
    # Ensure output folder exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df = load_data(input_path)
    df = clean_data(df)
    df.to_csv(output_path, index=False)
    print(f"✅ Cleaned data saved successfully to:\n{output_path}")
