# app/app.py
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# --------------------------
# Streamlit Config
# --------------------------
st.set_page_config(page_title="IBM HR Analytics Dashboard", page_icon="💼", layout="wide")

# --------------------------
# Custom CSS Styling
# --------------------------
st.markdown("""
    <style>
        .main {
            background: linear-gradient(135deg, #e3f2fd 0%, #fce4ec 100%);
        }
        .metric-container {
            background-color: white;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.1);
            text-align: center;
        }
        .title {
            text-align: center;
            font-size: 36px !important;
            color: #2c3e50;
            font-weight: 700;
            margin-bottom: 10px;
        }
        .subtitle {
            text-align: center;
            color: #616161;
            font-size: 18px;
        }
        .stTabs [data-baseweb="tab-list"] {
            justify-content: center;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: #f8f9fa;
            border-radius: 8px;
            margin: 2px;
            padding: 10px 20px;
        }
    </style>
""", unsafe_allow_html=True)

# --------------------------
# Load Data & Model
# --------------------------
@st.cache_data
def load_data():
    try:
        # Load cleaned data with same columns used in model training
        df = pd.read_csv(
            r"C:\Users\Nanda Chowgle\OneDrive\Desktop\IBM HR Analytics Employee Attrition & Performance\data\cleaned_hr_data.csv"
        )
    except FileNotFoundError:
        # Fallback to original dataset if cleaned file not found
        df = pd.read_csv(
            r"C:\Users\Nanda Chowgle\OneDrive\Desktop\IBM HR Analytics Employee Attrition & Performance\data\WA_Fn-UseC_-HR-Employee-Attrition (1).csv"
        )
    return df


@st.cache_resource
def load_model():
    model = joblib.load(
        r"C:\Users\Nanda Chowgle\OneDrive\Desktop\IBM HR Analytics Employee Attrition & Performance\models\hr_attrition_model.pkl"
    )
    # Load the same training columns (from cleaned data)
    train_df = pd.read_csv(
        r"C:\Users\Nanda Chowgle\OneDrive\Desktop\IBM HR Analytics Employee Attrition & Performance\data\cleaned_hr_data.csv"
    )
    feature_columns = [col for col in train_df.columns if col != 'Attrition']
    return model, feature_columns


df = load_data()
model, feature_columns = load_model()


# --------------------------
# Header
# --------------------------
st.markdown("<h1 class='title'>💼 IBM HR Analytics Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Analyze, Predict & Understand Employee Attrition</p>", unsafe_allow_html=True)
st.markdown("---")

# --------------------------
# KPIs Section
# --------------------------
col1, col2, col3, col4 = st.columns(4)

total_employees = len(df)

# Handle both text and numeric Attrition values
if 'Attrition' in df.columns:
    attr_values = df['Attrition'].value_counts(normalize=True)
    if 'Yes' in attr_values.index:
        attrition_rate = (attr_values['Yes'] * 100).round(2)
    elif 1 in attr_values.index:
        attrition_rate = (attr_values[1] * 100).round(2)
    else:
        attrition_rate = 0
else:
    attrition_rate = 0

avg_income = int(df['MonthlyIncome'].mean()) if 'MonthlyIncome' in df.columns else 0
avg_satisfaction = int(df['JobSatisfaction'].mean()) if 'JobSatisfaction' in df.columns else 0

with col1:
    st.markdown("<div class='metric-container'><h4>👥 Total Employees</h4><h2>"
                f"{total_employees}</h2></div>", unsafe_allow_html=True)
with col2:
    st.markdown("<div class='metric-container'><h4>📉 Attrition Rate</h4><h2>"
                f"{attrition_rate}%</h2></div>", unsafe_allow_html=True)
with col3:
    st.markdown("<div class='metric-container'><h4>💰 Avg Monthly Income</h4><h2>"
                f"${avg_income}</h2></div>", unsafe_allow_html=True)
with col4:
    st.markdown("<div class='metric-container'><h4>😊 Job Satisfaction</h4><h2>"
                f"{avg_satisfaction}/4</h2></div>", unsafe_allow_html=True)

st.markdown("---")

# --------------------------
# Tabs Layout
# --------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Insights Dashboard",
    "🔍 Correlation Heatmap",
    "🎯 Predict Attrition",
    "🧠 Explainability (Feature Importance)"
])

# ==========================
# TAB 1 - Insights
# ==========================
with tab1:
    st.subheader("Employee Attrition Insights")
    t1a, t1b = st.columns(2)

    with t1a:
        fig, ax = plt.subplots()
        sns.countplot(x="Department", hue="Attrition", data=df, palette="Set2", ax=ax)
        plt.title("Attrition by Department")
        plt.xticks(rotation=20)
        st.pyplot(fig)

        fig, ax = plt.subplots()
        sns.histplot(df['Age'], kde=True, color="#42a5f5")
        plt.title("Employee Age Distribution")
        st.pyplot(fig)

    with t1b:
        fig, ax = plt.subplots()
        sns.countplot(x="Gender", hue="Attrition", data=df, palette="husl", ax=ax)
        plt.title("Attrition by Gender")
        st.pyplot(fig)

        fig, ax = plt.subplots(figsize=(8,4))
        sns.countplot(x="JobRole", hue="Attrition", data=df, palette="coolwarm", ax=ax)
        plt.title("Attrition by Job Role")
        plt.xticks(rotation=45)
        st.pyplot(fig)

# ==========================
# TAB 2 - Correlation
# ==========================
with tab2:
    st.subheader("Correlation Heatmap")
    corr = df.select_dtypes(include=['int64', 'float64']).corr()
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(corr, cmap="YlGnBu", ax=ax)
    plt.title("Feature Correlation Matrix")
    st.pyplot(fig)

# ==========================
# TAB 3 - Prediction
# ==========================
with tab3:
    st.subheader("🎯 Predict Employee Attrition")
    st.info("Enter details to predict whether the employee will stay or leave.")

    colA, colB, colC = st.columns(3)

    age = colA.slider("Age", 18, 60, 30)
    distance = colA.slider("Distance From Home", 1, 30, 5)
    monthly_income = colB.number_input("Monthly Income ($)", 1000, 20000, 5000)
    job_satisfaction = colB.slider("Job Satisfaction (1-Low → 4-Very High)", 1, 4, 3)
    work_life_balance = colC.slider("Work Life Balance (1-Bad → 4-Best)", 1, 4, 3)
    overtime = colC.selectbox("OverTime", ["Yes", "No"])

    # Create base input
    user_input = {
        'Age': age,
        'DistanceFromHome': distance,
        'MonthlyIncome': monthly_income,
        'JobSatisfaction': job_satisfaction,
        'WorkLifeBalance': work_life_balance,
        'OverTime': 1 if overtime == "Yes" else 0
    }

    # Convert to DataFrame
    input_data = pd.DataFrame([user_input])

    # ✅ Align user input with training features
    for col in feature_columns:
        if col not in input_data.columns:
            input_data[col] = 0  # fill missing features with 0

    input_data = input_data[feature_columns]  # ensure correct column order

    if st.button("🚀 Predict Now"):
        prediction = model.predict(input_data)
        if prediction[0] == 1:
            st.error("⚠️ Employee is **Likely to Leave** the company.")
        else:
            st.success("✅ Employee is **Likely to Stay** with the company.")

# ==========================
# TAB 4 - Explainability
# ==========================
with tab4:
    st.subheader("🧠 Feature Importance & SHAP Explainability")

    try:
        # Match feature columns used during model training
        X = df.drop('Attrition', axis=1, errors='ignore')

        # Feature Importance
        st.markdown("### 🔝 Top 10 Influencing Features")
        importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(8,5))
        importances.head(10).plot(kind='bar', color='#26a69a', ax=ax)
        plt.title("Top Features Affecting Attrition")
        st.pyplot(fig)

        # SHAP Values
        st.markdown("### 🌈 SHAP Summary Plot (Why Attrition Happens)")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X.sample(100, random_state=42))
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values[1], X.sample(100, random_state=42), plot_type="bar", show=False)
        st.pyplot(fig)
        st.caption("Higher SHAP values indicate stronger influence on employee leaving.")
    except Exception as e:
        st.warning(f"Explainability not available: {e}")

# --------------------------
# Footer
# --------------------------
st.markdown("---")
st.markdown("<p style='text-align:center;color:gray;'>👩‍💻 Created by <b>Nanda S.C</b> | IBM HR Analytics Project | Data Science</p>", unsafe_allow_html=True)
