import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Loan Eligibility Predictor",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Loan Eligibility Approval Predictor")

st.write(
    "Predict whether a loan application will be Approved or Rejected."
)

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("loan_data.csv")

# -----------------------------
# ENCODING
# -----------------------------
data = df.copy()

encoders = {}

for col in ["Gender", "Married", "Education", "Loan_Status"]:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    encoders[col] = le

# -----------------------------
# MODEL TRAINING
# -----------------------------
X = data.drop("Loan_Status", axis=1)
y = data["Loan_Status"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LogisticRegression()

model.fit(X_train, y_train)

pred = model.predict(X_test)

accuracy = accuracy_score(y_test, pred)

# -----------------------------
# KPI SECTION
# -----------------------------
col1, col2, col3 = st.columns(3)

col1.metric("Applications", len(df))
col2.metric("Features", len(X.columns))
col3.metric("Accuracy", f"{accuracy:.2f}")

st.divider()

# -----------------------------
# USER INPUT
# -----------------------------
st.subheader("Applicant Details")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    married = st.selectbox(
        "Married",
        ["Yes", "No"]
    )

    education = st.selectbox(
        "Education",
        ["Graduate", "Not Graduate"]
    )

with col2:
    income = st.number_input(
        "Applicant Income",
        min_value=0,
        value=5000
    )

    loan_amount = st.number_input(
        "Loan Amount",
        min_value=0,
        value=150
    )

    credit_history = st.selectbox(
        "Credit History",
        [1, 0]
    )

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("Predict Loan Status"):

    input_data = pd.DataFrame({
        "Gender": [1 if gender == "Male" else 0],
        "Married": [1 if married == "Yes" else 0],
        "ApplicantIncome": [income],
        "LoanAmount": [loan_amount],
        "Credit_History": [credit_history],
        "Education": [1 if education == "Not Graduate" else 0]
    })

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

# -----------------------------
# DATA PREVIEW
# -----------------------------
st.subheader("Dataset Preview")

st.dataframe(df)

# -----------------------------
# LOAN STATUS DISTRIBUTION
# -----------------------------
st.subheader("Loan Approval Distribution")

loan_counts = df["Loan_Status"].value_counts()

st.bar_chart(loan_counts)

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.caption(
    "Built using Streamlit, Pandas and Logistic Regression"
)