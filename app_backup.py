
import streamlit as st
import joblib
import pandas as pd


# =========================================================
# LOAD MODEL
# =========================================================

model_data = joblib.load("models/churn_model.pkl")

model = model_data["model"]
feature_columns = model_data["features"]

threshold = model_data.get("threshold", 0.65)

accuracy = model_data.get("accuracy", 0.7811)
precision = model_data.get("precision", 0.5750)
recall = model_data.get("recall", 0.6765)
f1 = model_data.get("f1", 0.6216)
roc_auc = model_data.get("roc_auc", 0.8350)


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="centered"
)


# =========================================================
# TITLE
# =========================================================

st.title("📊 Customer Churn Predictor")

st.write(
    "Enter customer details to predict their churn risk."
)

st.divider()


# =========================================================
# CUSTOMER INFORMATION
# =========================================================

st.subheader("👤 Customer Information")

col1, col2 = st.columns(2)


with col1:

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    senior_citizen = st.selectbox(
        "Senior Citizen",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )


with col2:

    dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"]
    )

    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=72,
        value=12
    )

    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )


# =========================================================
# INTERNET & SERVICES
# =========================================================

st.subheader("🌐 Internet & Services")

col1, col2 = st.columns(2)


with col1:

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["Yes", "No", "No phone service"]
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )


with col2:

    device_protection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )


# =========================================================
# BILLING INFORMATION
# =========================================================

st.subheader("💳 Billing Information")

col1, col2 = st.columns(2)


with col1:

    contract = st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )


with col2:

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )


# =========================================================
# CHARGES
# =========================================================

st.subheader("💰 Charges")

col1, col2 = st.columns(2)


with col1:

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0
    )


with col2:

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=840.0
    )


st.divider()


# =========================================================
# PREDICTION
# =========================================================

if st.button(
    "🔮 Predict Churn",
    use_container_width=True
):

    # -----------------------------------------------------
    # CREATE INPUT DATAFRAME
    # -----------------------------------------------------

    input_data = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [senior_citizen],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "PhoneService": [phone_service],
        "MultipleLines": [multiple_lines],
        "InternetService": [internet_service],
        "OnlineSecurity": [online_security],
        "OnlineBackup": [online_backup],
        "DeviceProtection": [device_protection],
        "TechSupport": [tech_support],
        "StreamingTV": [streaming_tv],
        "StreamingMovies": [streaming_movies],
        "Contract": [contract],
        "PaperlessBilling": [paperless_billing],
        "PaymentMethod": [payment_method],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges]
    })


    # -----------------------------------------------------
    # ONE-HOT ENCODING
    # -----------------------------------------------------

    input_data = pd.get_dummies(
        input_data,
        drop_first=True
    )


    # -----------------------------------------------------
    # MATCH TRAINING FEATURES
    # -----------------------------------------------------

    input_data = input_data.reindex(
        columns=feature_columns,
        fill_value=0
    )


    # -----------------------------------------------------
    # GET CHURN PROBABILITY
    # -----------------------------------------------------

    probability = model.predict_proba(
        input_data
    )[0][1]


    # -----------------------------------------------------
    # APPLY TRAINED THRESHOLD
    # -----------------------------------------------------

    prediction = int(
        probability >= threshold
    )


    # =====================================================
    # PREDICTION RESULT
    # =====================================================

    st.divider()

    st.subheader("📊 Prediction Result")


    if prediction == 1:

        st.error(
            "⚠️ HIGH CHURN RISK"
        )

    else:

        st.success(
            "✅ LOW CHURN RISK"
        )


    st.metric(
        "Churn Probability",
        f"{probability * 100:.2f}%"
    )


    # =====================================================
    # RISK GAUGE
    # =====================================================

    st.progress(
        min(int(probability * 100), 100),
        text=f"Churn Risk: {probability * 100:.1f}%"
    )


    # =====================================================
    # RISK LEVEL
    # =====================================================

    if probability < 0.30:

        risk = "LOW"

    elif probability < threshold:

        risk = "MEDIUM"

    else:

        risk = "HIGH"


    st.write(
        f"**Risk Level: {risk}**"
    )


    # =====================================================
    # WHY THIS PREDICTION?
    # =====================================================

    st.subheader(
        "🔍 Why this prediction?"
    )


    reasons = []


    # -----------------------------------------------------
    # TENURE
    # -----------------------------------------------------

    if tenure <= 12:

        reasons.append(
            (
                "🔴",
                "Short tenure",
                "Customers with short tenure have higher churn risk."
            )
        )

    elif tenure >= 36:

        reasons.append(
            (
                "🟢",
                "Long tenure",
                "Long-term customers are generally less likely to churn."
            )
        )


    # -----------------------------------------------------
    # CONTRACT
    # -----------------------------------------------------

    if contract == "Month-to-month":

        reasons.append(
            (
                "🔴",
                "Month-to-month contract",
                "Month-to-month customers have higher churn risk."
            )
        )

    elif contract == "Two year":

        reasons.append(
            (
                "🟢",
                "Two-year contract",
                "Long-term contracts are associated with lower churn."
            )
        )


    # -----------------------------------------------------
    # INTERNET SERVICE
    # -----------------------------------------------------

    if internet_service == "Fiber optic":

        reasons.append(
            (
                "🔴",
                "Fiber optic service",
                "Fiber-optic customers show relatively higher churn in this dataset."
            )
        )


    # -----------------------------------------------------
    # ONLINE SECURITY
    # -----------------------------------------------------

    if online_security == "Yes":

        reasons.append(
            (
                "🟢",
                "Online Security enabled",
                "Security services are associated with lower churn."
            )
        )

    elif online_security == "No":

        reasons.append(
            (
                "🔴",
                "No Online Security",
                "Customers without online security may have higher churn risk."
            )
        )


    # -----------------------------------------------------
    # TECH SUPPORT
    # -----------------------------------------------------

    if tech_support == "Yes":

        reasons.append(
            (
                "🟢",
                "Tech Support enabled",
                "Technical support is associated with better customer retention."
            )
        )

    elif tech_support == "No":

        reasons.append(
            (
                "🔴",
                "No Tech Support",
                "Customers without technical support may have higher churn risk."
            )
        )


    # -----------------------------------------------------
    # PAYMENT METHOD
    # -----------------------------------------------------

    if payment_method == "Electronic check":

        reasons.append(
            (
                "🔴",
                "Electronic check",
                "Electronic-check customers show relatively higher churn in this dataset."
            )
        )


    # -----------------------------------------------------
    # MONTHLY CHARGES
    # -----------------------------------------------------

    if monthly_charges >= 80:

        reasons.append(
            (
                "🔴",
                "High monthly charges",
                "Higher monthly charges can contribute to increased churn risk."
            )
        )

    elif monthly_charges <= 40:

        reasons.append(
            (
                "🟢",
                "Lower monthly charges",
                "Lower monthly charges are generally associated with lower churn."
            )
        )


    # -----------------------------------------------------
    # DISPLAY REASONS
    # -----------------------------------------------------

    if reasons:

        for icon, title, explanation in reasons[:5]:

            st.write(
                f"{icon} **{title}** — {explanation}"
            )

    else:

        st.write(
            "No strong individual factors identified."
        )


# =========================================================
# MODEL PERFORMANCE
# =========================================================

st.divider()

st.subheader("📈 Model Performance")


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Accuracy",
        f"{accuracy * 100:.2f}%"
    )


with col2:

    st.metric(
        "Precision",
        f"{precision * 100:.2f}%"
    )


with col3:

    st.metric(
        "Recall",
        f"{recall * 100:.2f}%"
    )


with col4:

    st.metric(
        "F1 Score",
        f"{f1 * 100:.2f}%"
    )


st.metric(
    "ROC-AUC",
    f"{roc_auc:.3f}"
)


st.info(
    f"The model uses a probability threshold of "
    f"{threshold:.2f}, selected using threshold analysis "
    f"to maximize F1 score."
)


# =========================================================
# MODEL COMPARISON
# =========================================================

st.subheader("🤖 Model Comparison")


comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest"
    ],

    "Accuracy": [
        accuracy,
        0.7868
    ],

    "Precision": [
        precision,
        0.6209
    ],

    "Recall": [
        recall,
        0.5080
    ],

    "F1 Score": [
        f1,
        0.5588
    ]
})


comparison_display = comparison.copy()

for column in [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score"
]:

    comparison_display[column] = (
        comparison_display[column] * 100
    ).round(2).astype(str) + "%"


st.dataframe(
    comparison_display,
    use_container_width=True,
    hide_index=True
)


# =========================================================
# CROSS VALIDATION
# =========================================================

st.subheader("🔄 Cross-Validation")


st.metric(
    "5-Fold Cross-Validation F1",
    "63.24%"
)


st.caption(
    "Average F1 score across 5 validation folds."
)


# =========================================================
# MODEL SUMMARY
# =========================================================

st.divider()

st.subheader("ℹ️ Model Summary")

st.write(
    "The system uses Logistic Regression with "
    "feature scaling and balanced class weights."
)

st.write(
    "The model was trained on 7,032 cleaned customer "
    "records with 30 encoded features."
)

st.write(
    f"The selected probability threshold is "
    f"{threshold:.2f}."
)

st.write(
    f"The model achieved a ROC-AUC score of "
    f"{roc_auc:.3f}."
)
# =========================================================
# ANALYTICS DASHBOARD
# =========================================================

st.divider()

st.header("📊 Customer Analytics Dashboard")

# Load original dataset
@st.cache_data
def load_data():

    data = pd.read_csv(
        "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    )

    data["TotalCharges"] = pd.to_numeric(
        data["TotalCharges"],
        errors="coerce"
    )

    data = data.dropna(
        subset=["TotalCharges"]
    )

    return data


df = load_data()


# =========================================================
# KPI CARDS
# =========================================================

total_customers = len(df)

churned_customers = (
    df["Churn"] == "Yes"
).sum()

churn_rate = (
    churned_customers / total_customers
) * 100

avg_monthly_charges = df[
    "MonthlyCharges"
].mean()


col1, col2, col3, col4 = st.columns(4)


with col1:
    st.metric(
        "👥 Total Customers",
        f"{total_customers:,}"
    )


with col2:
    st.metric(
        "⚠️ Churned Customers",
        f"{churned_customers:,}"
    )


with col3:
    st.metric(
        "📉 Churn Rate",
        f"{churn_rate:.2f}%"
    )


with col4:
    st.metric(
        "💰 Avg Monthly Charges",
        f"${avg_monthly_charges:.2f}"
    )


# =========================================================
# CHURN DISTRIBUTION
# =========================================================

st.subheader("📉 Churn Distribution")

churn_distribution = (
    df["Churn"]
    .value_counts()
    .rename_axis("Churn")
    .reset_index(name="Customers")
)

st.bar_chart(
    churn_distribution.set_index("Churn")
)


# =========================================================
# CHURN BY CONTRACT
# =========================================================

st.subheader("📄 Churn by Contract")


contract_churn = pd.crosstab(
    df["Contract"],
    df["Churn"],
    normalize="index"
) * 100


st.bar_chart(
    contract_churn
)


# =========================================================
# CHURN BY INTERNET SERVICE
# =========================================================

st.subheader("🌐 Churn by Internet Service")


internet_churn = pd.crosstab(
    df["InternetService"],
    df["Churn"],
    normalize="index"
) * 100


st.bar_chart(
    internet_churn
)


# =========================================================
# CHURN BY PAYMENT METHOD
# =========================================================

st.subheader("💳 Churn by Payment Method")


payment_churn = pd.crosstab(
    df["PaymentMethod"],
    df["Churn"],
    normalize="index"
) * 100


st.bar_chart(
    payment_churn
)


# =========================================================
# TENURE ANALYSIS
# =========================================================

st.subheader("⏳ Churn by Tenure")


df["TenureGroup"] = pd.cut(
    df["tenure"],
    bins=[0, 12, 24, 36, 48, 60, 72],
    labels=[
        "0-12 months",
        "13-24 months",
        "25-36 months",
        "37-48 months",
        "49-60 months",
        "61-72 months"
    ]
)


tenure_churn = pd.crosstab(
    df["TenureGroup"],
    df["Churn"],
    normalize="index"
) * 100


st.bar_chart(
    tenure_churn
)


# =========================================================
# BUSINESS INSIGHTS
# =========================================================

st.subheader("💡 Key Business Insights")


contract_rates = (
    df.groupby("Contract")["Churn"]
    .apply(lambda x: (x == "Yes").mean() * 100)
)


highest_contract = contract_rates.idxmax()
highest_contract_rate = contract_rates.max()


payment_rates = (
    df.groupby("PaymentMethod")["Churn"]
    .apply(lambda x: (x == "Yes").mean() * 100)
)


highest_payment = payment_rates.idxmax()
highest_payment_rate = payment_rates.max()


internet_rates = (
    df.groupby("InternetService")["Churn"]
    .apply(lambda x: (x == "Yes").mean() * 100)
)


highest_internet = internet_rates.idxmax()
highest_internet_rate = internet_rates.max()


st.info(
    f"📄 **Contract insight:** "
    f"{highest_contract} customers have the highest observed "
    f"churn rate at approximately "
    f"{highest_contract_rate:.1f}%."
)


st.info(
    f"💳 **Payment insight:** "
    f"{highest_payment} customers have the highest observed "
    f"churn rate at approximately "
    f"{highest_payment_rate:.1f}%."
)


st.info(
    f"🌐 **Internet insight:** "
    f"{highest_internet} customers have the highest observed "
    f"churn rate at approximately "
    f"{highest_internet_rate:.1f}%."
)

