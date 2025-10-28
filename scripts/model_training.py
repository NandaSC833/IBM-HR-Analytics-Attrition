# scripts/model_training.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

def train_model():
    df = pd.read_csv(r"C:\Users\Nanda Chowgle\OneDrive\Desktop\IBM HR Analytics Employee Attrition & Performance\data\cleaned_hr_data.csv")

    # Ensure data is numeric
    df = df.apply(pd.to_numeric, errors='ignore')
    if df.select_dtypes('object').shape[1] > 0:
        raise ValueError("❌ Dataset still contains categorical columns. Run data_preprocessing.py first.")

    X = df.drop('Attrition', axis=1)
    y = df['Attrition']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("✅ Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

    joblib.dump(model, r"C:\Users\Nanda Chowgle\OneDrive\Desktop\IBM HR Analytics Employee Attrition & Performance\models\hr_attrition_model.pkl")
    print("✅ Model saved successfully!")

if __name__ == "__main__":
    train_model()
