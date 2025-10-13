# streamlit_app.py
import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# ==========================================================
# Load Dataset
# ==========================================================
@st.cache_data
def load_data():
    df = pd.read_excel("Employee-Attrition.xlsx")
    print("Initial Shape:", df.shape)
    print(df.info())
    print("Columns:", df.columns.tolist())
    drop_cols = ['EmployeeCount','Over18','StandardHours','EmployeeNumber']
    df = df.drop(columns=drop_cols)
    return df

df = load_data()

# Encode categorical columns globally
label_encoders = {}
for col in ['BusinessTravel','Department','EducationField',
            'Gender','JobRole','MaritalStatus','OverTime','Education']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# ==========================================================
# Sidebar Navigation
# ==========================================================
st.sidebar.title("📊 HR Analytics Dashboard")
page = st.sidebar.radio("Go to:", ["🏠 Home", "🔮 Attrition Prediction", "⭐ Performance Prediction"])

# ==========================================================
# Home Page
# ==========================================================
if page == "🏠 Home":
    st.title("Employee Attrition & Performance Dashboard")
    st.markdown("""
    **Business Objectives**
    - Predict attrition and performance ratings.
    - Identify key influencing factors.
    - Provide actionable HR insights.
    """)
    st.dataframe(df.head())

# ==========================================================
# Attrition Prediction (Model 1)
# ==========================================================
elif page == "🔮 Attrition Prediction":
    st.title("Attrition Risk Prediction")

    df['Attrition'] = df['Attrition'].map({'Yes':1, 'No':0})

    features = [
        'OverTime','MaritalStatus','DistanceFromHome','JobRole','Department',
        'TotalWorkingYears','JobLevel','YearsInCurrentRole','MonthlyIncome',
        'Age','YearsWithCurrManager','StockOptionLevel','YearsAtCompany',
        'JobInvolvement','Education'
    ]
    X = df[features]
    y = df['Attrition']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_scaled, y)
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
    )

    rf_model = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
    rf_model.fit(X_train, y_train)

    # Input form (dynamic) – 15 fields
    st.subheader("Enter Employee Details")
    user_input = {}
    for f in features:
        if f in ['OverTime','MaritalStatus','JobRole','Department','Education']:
            user_input[f] = st.selectbox(f, label_encoders[f].classes_)
        elif f in ['JobLevel','StockOptionLevel','JobInvolvement']:
            max_val = 5 if f == "JobLevel" else 3 if f == "StockOptionLevel" else 4
            user_input[f] = st.slider(f, 0, max_val, 1)
        else:  # Age, DistanceFromHome, YearsAtCompany, YearsInCurrentRole, YearsWithCurrManager, TotalWorkingYears, MonthlyIncome
            user_input[f] = st.number_input(f, min_value=0, max_value=1000000, step=1)

    # Encode categoricals
    for col in ['OverTime','MaritalStatus','JobRole','Department','Education']:
        user_input[col] = label_encoders[col].transform([user_input[col]])[0]

    if st.button("Predict Attrition Risk"):
        X_new = pd.DataFrame([user_input], columns=features)
        X_new = scaler.transform(X_new)
        pred = rf_model.predict(X_new)[0]
        prob = rf_model.predict_proba(X_new)[0,1]

        label = "Attrition Risk" if pred == 1 else "No Attrition"
        st.write(f"**Prediction Possibility:** {label} (Probability: {prob:.2f})")
        st.write(f"**Prediction:** {pred}  (1 = Leave, 0 = Stay)")

        if pred == 1:
            st.warning("⚠️ Employee is at risk of leaving. Consider immediate retention actions.")
            st.markdown("""
            **HR Suggestions (Automatic):**
            - Review workload and reduce excessive overtime.
            - Ensure competitive compensation (especially for lower salary bands).
            - Provide career growth opportunities (promotions, learning programs).
            - Strengthen manager–employee relationship (coaching, recognition).
            """)
        else:
            st.success("✅ Employee is predicted to stay. Continue normal development & recognition programs.")
            st.markdown("""
            **Business Recommendations Summary**
            - Engagement score is predictive — invest in engagement initiatives (surveys, mentorship, recognition).
            - Overtime is a strong driver — monitor overtime hours and ensure proper work-life balance.
            - Compensation impacts attrition — review salary bands and align pay with market standards.
            - Use targeted retention (manager coaching, promotions, pay adjustments) for high-risk groups.
            """)

# ==========================================================
# Performance Rating Prediction (Model 2)
# ==========================================================
elif page == "⭐ Performance Prediction":
    st.title("Performance Rating Prediction")

    features = [
        'YearsInCurrentRole','YearsWithCurrManager','YearsSinceLastPromotion',
        'TotalWorkingYears','DistanceFromHome','RelationshipSatisfaction',
        'EnvironmentSatisfaction','JobInvolvement','PercentSalaryHike',
        'Age','JobLevel','Education','StockOptionLevel','MonthlyIncome','OverTime'
    ]
    X = df[features]
    y = df['PerformanceRating']

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_scaled, y)
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.25, random_state=42)

    rf_model = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
    rf_model.fit(X_train, y_train)

    # Input form (dynamic) – 15 fields
    st.subheader("Enter Employee Details")
    user_input = {}
    for f in features:
        if f in ['RelationshipSatisfaction','EnvironmentSatisfaction','JobInvolvement','JobLevel','StockOptionLevel']:
            max_val = 4 if f in ['RelationshipSatisfaction','EnvironmentSatisfaction','JobInvolvement'] else 5
            user_input[f] = st.slider(f, 1, max_val, 2)
        elif f in ['PercentSalaryHike','DistanceFromHome','YearsSinceLastPromotion','YearsInCurrentRole',
                   'YearsWithCurrManager','TotalWorkingYears','Age','MonthlyIncome']:
            user_input[f] = st.number_input(f, min_value=0, max_value=1000000, step=1)
        elif f in ['Education','OverTime']:
            user_input[f] = st.selectbox(f, label_encoders[f].classes_)

    # Encode categoricals before scaling
    for col in ['Education','OverTime']:
        user_input[col] = label_encoders[col].transform([user_input[col]])[0]

    if st.button("Predict Performance Rating"):
        X_new = pd.DataFrame([user_input], columns=features)
        X_new = scaler.transform(X_new)
        pred = rf_model.predict(X_new)[0]

        st.write(f"**Predicted Performance Rating:** {pred}")

        if pred <= 2:
            st.warning("⚠️ Low/Below Average Performance detected.")
            st.markdown("""
            **HR Suggestions (Automatic):**
            - Provide targeted training & skill development.
            - Increase mentoring and coaching support.
            - Review role fit and workload balance.
            - Create performance improvement plans with clear milestones.
            """)
        elif pred == 3:
            st.info("ℹ️ Average Performance detected.")
            st.markdown("""
            **HR Suggestions (Automatic):**
            - Encourage employee through recognition & feedback.
            - Offer moderate learning and upskilling opportunities.
            - Maintain balanced workload and support career growth.
            """)
        else:
            st.success("🌟 High Performance detected.")
            st.markdown("""
            **HR Suggestions (Automatic):**
            - Recognize and reward with incentives/bonuses.
            - Provide leadership opportunities and promotions.
            - Retain top performers with career growth pathways.
            - Encourage them to mentor junior employees.
            """)

# python -m streamlit run streamlit_app.py
