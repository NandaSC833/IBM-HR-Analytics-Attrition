# scripts/predict.py
import pandas as pd
import joblib

def predict_attrition(input_data):
    # Load the trained model
    model = joblib.load(r"C:\Users\Nanda Chowgle\OneDrive\Desktop\IBM HR Analytics Employee Attrition & Performance\models\hr_attrition_model.pkl")
    
    # Load the same cleaned dataset to get correct feature names
    df = pd.read_csv(r"C:\Users\Nanda Chowgle\OneDrive\Desktop\IBM HR Analytics Employee Attrition & Performance\data\cleaned_hr_data.csv")
    X = df.drop('Attrition', axis=1)

    # Create a DataFrame with correct column names
    input_df = pd.DataFrame([input_data], columns=X.columns)

    # Predict
    prediction = model.predict(input_df)
    return "Likely to Leave" if prediction[0] == 1 else "Likely to Stay"

# Example test run
if __name__ == "__main__":
    # ⚠️ You must pass data with same order & number of features as model expects
    df_ref = pd.read_csv(r"C:\Users\Nanda Chowgle\OneDrive\Desktop\IBM HR Analytics Employee Attrition & Performance\data\cleaned_hr_data.csv")
    feature_names = df_ref.drop('Attrition', axis=1).columns

    # Example: take the first row from dataset to ensure correct shape
    sample = df_ref.drop('Attrition', axis=1).iloc[0].to_dict()
    print("🔍 Predicting for sample:", sample)
    
    result = predict_attrition(sample)
    print("✅ Prediction:", result)
