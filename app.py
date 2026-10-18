import streamlit as st
import joblib
import pandas as pd


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    model_data = joblib.load(
        "models/churn_model.pkl"
    )

    return model_data


model_data = load_model()

model = model_data["model"]
feature_columns = model_data["features"]

threshold = model_data.get(
    "threshold",
    0.65
)

accuracy = model_data.get(
    "accuracy",
    0.7811
)

precision = model_data.get(
    "precision",
    0.5750
)

recall = model_data.get(
    "recall",
    0.6765
)

f1 = model_data.get(
    "f1",
    0.6216
)

roc_auc = model_data.get(
    "roc_auc",
    0.8350
)

cv_f1 = model_data.get(
    "cv_f1",
    0.6324
)


# =========================================================
# LOAD DATASET
# =========================================================

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
# SIDEBAR
# =========================================================

st.sidebar.title("📊 Churn Predictor")

st.sidebar.caption(
    "AI-powered customer retention analytics"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "🔮 Prediction",
        "📊 Analytics",
        "🤖 Model Performance",
        "💡 Business Insights",
        "ℹ️ About Project"
    ]
)

st.sidebar.divider()

st.sidebar.caption(
    "Customer Churn Prediction System"
)

st.sidebar.caption(
    "Machine Learning + Streamlit"
)


# =========================================================
# HELPER FUNCTION
# =========================================================

def create_prediction_input(
    gender,
    senior_citizen,
    partner,
    dependents,
    tenure,
    phone_service,
    multiple_lines,
    internet_service,
    online_security,
    online_backup,
    device_protection,
    tech_support,
    streaming_tv,
    streaming_movies,
    contract,
    paperless_billing,
    payment_method,
    monthly_charges,
    total_charges
):

    input_data = pd.DataFrame({

        "gender": [gender],

        "SeniorCitizen": [
            senior_citizen
        ],

        "Partner": [
            partner
        ],

        "Dependents": [
            dependents
        ],

        "tenure": [
            tenure
        ],

        "PhoneService": [
            phone_service
        ],

        "MultipleLines": [
            multiple_lines
        ],

        "InternetService": [
            internet_service
        ],

        "OnlineSecurity": [
            online_security
        ],

        "OnlineBackup": [
            online_backup
        ],

        "DeviceProtection": [
            device_protection
        ],

        "TechSupport": [
            tech_support
        ],

        "StreamingTV": [
            streaming_tv
        ],

        "StreamingMovies": [
            streaming_movies
        ],

        "Contract": [
            contract
        ],

        "PaperlessBilling": [
            paperless_billing
        ],

        "PaymentMethod": [
            payment_method
        ],

        "MonthlyCharges": [
            monthly_charges
        ],

        "TotalCharges": [
            total_charges
        ]
    })

    input_data = pd.get_dummies(
        input_data,
        drop_first=True
    )

    input_data = input_data.reindex(
        columns=feature_columns,
        fill_value=0
    )

    return input_data


# =========================================================
# PAGE 1 — PREDICTION
# =========================================================

if page == "🔮 Prediction":

    st.title(
        "📊 Customer Churn Predictor"
    )

    st.write(
        "Enter customer information to "
        "predict their probability of churn."
    )

    st.divider()

    # -----------------------------------------------------
    # CUSTOMER INFORMATION
    # -----------------------------------------------------

    st.subheader(
        "👤 Customer Information"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        gender = st.selectbox(
            "Gender",
            ["Female", "Male"]
        )

        senior_citizen = st.selectbox(
            "Senior Citizen",
            [0, 1],
            format_func=lambda x:
                "Yes" if x == 1 else "No"
        )

    with col2:

        partner = st.selectbox(
            "Partner",
            ["Yes", "No"]
        )

        dependents = st.selectbox(
            "Dependents",
            ["Yes", "No"]
        )

    with col3:

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


    # -----------------------------------------------------
    # INTERNET & SERVICES
    # -----------------------------------------------------

    st.subheader(
        "🌐 Internet & Services"
    )

    col1, col2 = st.columns(2)

    with col1:

        multiple_lines = st.selectbox(
            "Multiple Lines",
            [
                "Yes",
                "No",
                "No phone service"
            ]
        )

        internet_service = st.selectbox(
            "Internet Service",
            [
                "DSL",
                "Fiber optic",
                "No"
            ]
        )

        online_security = st.selectbox(
            "Online Security",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

        online_backup = st.selectbox(
            "Online Backup",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

    with col2:

        device_protection = st.selectbox(
            "Device Protection",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

        tech_support = st.selectbox(
            "Tech Support",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

        streaming_tv = st.selectbox(
            "Streaming TV",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )


    # -----------------------------------------------------
    # BILLING
    # -----------------------------------------------------

    st.subheader(
        "💳 Billing Information"
    )

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


    # -----------------------------------------------------
    # CHARGES
    # -----------------------------------------------------

    st.subheader(
        "💰 Charges"
    )

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


    # -----------------------------------------------------
    # PREDICT
    # -----------------------------------------------------

    if st.button(
        "🔮 Predict Churn",
        type="primary",
        use_container_width=True
    ):

        input_data = create_prediction_input(

            gender,
            senior_citizen,
            partner,
            dependents,
            tenure,
            phone_service,
            multiple_lines,
            internet_service,
            online_security,
            online_backup,
            device_protection,
            tech_support,
            streaming_tv,
            streaming_movies,
            contract,
            paperless_billing,
            payment_method,
            monthly_charges,
            total_charges
        )

        probability = model.predict_proba(
            input_data
        )[0][1]

        prediction = int(
            probability >= threshold
        )

        # -------------------------------------------------
        # RESULT
        # -------------------------------------------------

        st.divider()

        st.subheader(
            "📊 Prediction Result"
        )

        result_col1, result_col2 = st.columns(2)

        with result_col1:

            if prediction == 1:

                st.error(
                    "⚠️ HIGH CHURN RISK"
                )

            else:

                st.success(
                    "✅ LOW CHURN RISK"
                )

        with result_col2:

            st.metric(
                "Churn Probability",
                f"{probability * 100:.2f}%"
            )


        # -------------------------------------------------
        # RISK
        # -------------------------------------------------

        st.progress(
            min(
                int(probability * 100),
                100
            ),
            text=(
                f"Churn Risk: "
                f"{probability * 100:.1f}%"
            )
        )

        if probability < 0.30:

            risk = "LOW"

        elif probability < threshold:

            risk = "MEDIUM"

        else:

            risk = "HIGH"

        st.write(
            f"**Risk Level: {risk}**"
        )


        # -------------------------------------------------
        # EXPLANATION
        # -------------------------------------------------

        st.subheader(
            "🔍 Why this prediction?"
        )

        reasons = []


        if tenure <= 12:

            reasons.append(
                (
                    "🔴",
                    "Short tenure",
                    "Customers with short tenure "
                    "have higher churn risk."
                )
            )

        elif tenure >= 36:

            reasons.append(
                (
                    "🟢",
                    "Long tenure",
                    "Long-term customers are "
                    "generally less likely to churn."
                )
            )


        if contract == "Month-to-month":

            reasons.append(
                (
                    "🔴",
                    "Month-to-month contract",
                    "Month-to-month customers "
                    "have higher churn risk."
                )
            )

        elif contract == "Two year":

            reasons.append(
                (
                    "🟢",
                    "Two-year contract",
                    "Long-term contracts are "
                    "associated with lower churn."
                )
            )


        if internet_service == "Fiber optic":

            reasons.append(
                (
                    "🔴",
                    "Fiber optic service",
                    "Fiber-optic customers show "
                    "relatively higher churn in "
                    "this dataset."
                )
            )


        if online_security == "Yes":

            reasons.append(
                (
                    "🟢",
                    "Online Security enabled",
                    "Security services are associated "
                    "with lower churn."
                )
            )

        elif online_security == "No":

            reasons.append(
                (
                    "🔴",
                    "No Online Security",
                    "Customers without online security "
                    "may have higher churn risk."
                )
            )


        if tech_support == "Yes":

            reasons.append(
                (
                    "🟢",
                    "Tech Support enabled",
                    "Technical support is associated "
                    "with better customer retention."
                )
            )

        elif tech_support == "No":

            reasons.append(
                (
                    "🔴",
                    "No Tech Support",
                    "Customers without technical support "
                    "may have higher churn risk."
                )
            )


        if payment_method == "Electronic check":

            reasons.append(
                (
                    "🔴",
                    "Electronic check",
                    "Electronic-check customers show "
                    "relatively higher churn in "
                    "this dataset."
                )
            )


        if monthly_charges >= 80:

            reasons.append(
                (
                    "🔴",
                    "High monthly charges",
                    "Higher monthly charges can "
                    "contribute to increased churn risk."
                )
            )

        elif monthly_charges <= 40:

            reasons.append(
                (
                    "🟢",
                    "Lower monthly charges",
                    "Lower monthly charges are "
                    "generally associated with lower churn."
                )
            )


        if reasons:

            for icon, title, explanation in reasons[:5]:

                st.write(
                    f"{icon} **{title}** — "
                    f"{explanation}"
                )

        else:

            st.write(
                "No strong individual factors identified."
            )


# =========================================================
# PAGE 2 — ANALYTICS
# =========================================================

elif page == "📊 Analytics":

    st.title(
        "📊 Customer Analytics"
    )

    st.write(
        "Explore customer churn patterns "
        "across the dataset."
    )

    st.divider()


    total_customers = len(df)

    churned_customers = (
        df["Churn"] == "Yes"
    ).sum()

    churn_rate = (
        churned_customers /
        total_customers
    ) * 100

    avg_monthly_charges = (
        df["MonthlyCharges"].mean()
    )


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


    # -----------------------------------------------------
    # CHURN DISTRIBUTION
    # -----------------------------------------------------

    st.subheader(
        "📉 Churn Distribution"
    )

    churn_distribution = (
        df["Churn"]
        .value_counts()
        .rename_axis("Churn")
        .reset_index(
            name="Customers"
        )
    )

    st.bar_chart(
        churn_distribution.set_index(
            "Churn"
        )
    )


    # -----------------------------------------------------
    # CONTRACT
    # -----------------------------------------------------

    st.subheader(
        "📄 Churn by Contract"
    )

    contract_churn = pd.crosstab(
        df["Contract"],
        df["Churn"],
        normalize="index"
    ) * 100

    st.bar_chart(
        contract_churn
    )


    # -----------------------------------------------------
    # INTERNET
    # -----------------------------------------------------

    st.subheader(
        "🌐 Churn by Internet Service"
    )

    internet_churn = pd.crosstab(
        df["InternetService"],
        df["Churn"],
        normalize="index"
    ) * 100

    st.bar_chart(
        internet_churn
    )


    # -----------------------------------------------------
    # PAYMENT
    # -----------------------------------------------------

    st.subheader(
        "💳 Churn by Payment Method"
    )

    payment_churn = pd.crosstab(
        df["PaymentMethod"],
        df["Churn"],
        normalize="index"
    ) * 100

    st.bar_chart(
        payment_churn
    )


    # -----------------------------------------------------
    # TENURE
    # -----------------------------------------------------

    st.subheader(
        "⏳ Churn by Tenure"
    )

    analytics_df = df.copy()

    analytics_df["TenureGroup"] = pd.cut(
        analytics_df["tenure"],
        bins=[
            0,
            12,
            24,
            36,
            48,
            60,
            72
        ],
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
        analytics_df["TenureGroup"],
        analytics_df["Churn"],
        normalize="index"
    ) * 100

    st.bar_chart(
        tenure_churn
    )


# =========================================================
# PAGE 3 — MODEL PERFORMANCE
# =========================================================

elif page == "🤖 Model Performance":

    st.title(
        "🤖 Model Performance"
    )

    st.write(
        "Evaluation metrics for the trained "
        "customer churn prediction model."
    )

    st.divider()


    col1, col2, col3, col4, col5 = st.columns(5)

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

    with col5:

        st.metric(
            "ROC-AUC",
            f"{roc_auc:.3f}"
        )


    st.divider()


    st.subheader(
        "🤖 Model Comparison"
    )

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
            comparison_display[column] *
            100
        ).round(2).astype(str) + "%"


    st.dataframe(
        comparison_display,
        use_container_width=True,
        hide_index=True
    )


    st.divider()


    st.subheader(
        "🔄 Cross-Validation"
    )

    st.metric(
        "5-Fold Cross-Validation F1",
        f"{cv_f1 * 100:.2f}%"
    )

    st.caption(
        "Average F1 score across five "
        "validation folds."
    )


    st.info(
        f"The production threshold is "
        f"{threshold:.2f}. It was selected "
        f"through threshold analysis to "
        f"maximize F1 score."
    )


    st.subheader(
        "📌 Model Configuration"
    )

    st.write(
        "• Algorithm: Logistic Regression"
    )

    st.write(
        "• Feature scaling: StandardScaler"
    )

    st.write(
        "• Class balancing: Balanced weights"
    )

    st.write(
        "• Encoded features: 30"
    )

    st.write(
        "• Training records: 7,032"
    )


# =========================================================
# PAGE 4 — BUSINESS INSIGHTS
# =========================================================

elif page == "💡 Business Insights":

    st.title(
        "💡 Business Insights"
    )

    st.write(
        "Key findings from customer churn "
        "analysis."
    )

    st.divider()


    contract_rates = (
        df.groupby("Contract")["Churn"]
        .apply(
            lambda x:
            (x == "Yes").mean() * 100
        )
    )

    highest_contract = (
        contract_rates.idxmax()
    )

    highest_contract_rate = (
        contract_rates.max()
    )


    payment_rates = (
        df.groupby("PaymentMethod")["Churn"]
        .apply(
            lambda x:
            (x == "Yes").mean() * 100
        )
    )

    highest_payment = (
        payment_rates.idxmax()
    )

    highest_payment_rate = (
        payment_rates.max()
    )


    internet_rates = (
        df.groupby("InternetService")["Churn"]
        .apply(
            lambda x:
            (x == "Yes").mean() * 100
        )
    )

    highest_internet = (
        internet_rates.idxmax()
    )

    highest_internet_rate = (
        internet_rates.max()
    )


    col1, col2 = st.columns(2)


    with col1:

        st.subheader(
            "📄 Contract"
        )

        st.metric(
            "Highest Churn Contract",
            highest_contract
        )

        st.write(
            f"Observed churn rate: "
            f"**{highest_contract_rate:.1f}%**"
        )


    with col2:

        st.subheader(
            "💳 Payment"
        )

        st.metric(
            "Highest Churn Payment",
            highest_payment
        )

        st.write(
            f"Observed churn rate: "
            f"**{highest_payment_rate:.1f}%**"
        )


    st.divider()


    st.subheader(
        "🌐 Internet Service"
    )

    st.metric(
        "Highest Churn Internet Service",
        highest_internet
    )

    st.write(
        f"Observed churn rate: "
        f"**{highest_internet_rate:.1f}%**"
    )


    st.divider()


    st.subheader(
        "🎯 Recommended Business Actions"
    )

    st.info(
        "1️⃣ Focus retention campaigns on "
        "month-to-month customers."
    )

    st.info(
        "2️⃣ Investigate why electronic-check "
        "customers show higher churn."
    )

    st.info(
        "3️⃣ Monitor fiber-optic customers "
        "for early churn signals."
    )

    st.info(
        "4️⃣ Offer incentives for customers "
        "to move toward longer contracts."
    )

    st.info(
        "5️⃣ Use the churn model to identify "
        "high-risk customers before they leave."
    )


# =========================================================
# PAGE 5 — ABOUT
# =========================================================

elif page == "ℹ️ About Project":

    st.title(
        "ℹ️ About This Project"
    )

    st.write(
        "### 📊 Customer Churn Predictor"
    )

    st.write(
        "This project uses machine learning "
        "to predict whether a telecom customer "
        "is likely to churn."
    )


    st.divider()


    st.subheader(
        "🧠 Machine Learning Pipeline"
    )

    st.write(
        "1. Dataset collection"
    )

    st.write(
        "2. Data cleaning"
    )

    st.write(
        "3. Missing-value handling"
    )

    st.write(
        "4. Categorical feature encoding"
    )

    st.write(
        "5. Train/test split"
    )

    st.write(
        "6. Feature scaling"
    )

    st.write(
        "7. Logistic Regression training"
    )

    st.write(
        "8. Random Forest comparison"
    )

    st.write(
        "9. Cross-validation"
    )

    st.write(
        "10. Hyperparameter tuning"
    )

    st.write(
        "11. Probability threshold optimization"
    )

    st.write(
        "12. Streamlit deployment"
    )


    st.divider()


    st.subheader(
        "📊 Dataset"
    )

    st.write(
        "Cleaned customer records: **7,032**"
    )

    st.write(
        "Encoded features: **30**"
    )

    st.write(
        "Target variable: **Customer Churn**"
    )


    st.divider()


    st.subheader(
        "🏆 Final Model"
    )

    st.write(
        "Algorithm: **Logistic Regression**"
    )

    st.write(
        f"ROC-AUC: **{roc_auc:.3f}**"
    )

    st.write(
        f"F1 Score: **{f1 * 100:.2f}%**"
    )

    st.write(
        f"Decision Threshold: **{threshold:.2f}**"
    )


    st.success(
        "The project combines machine learning, "
        "data analysis, visualization, and a "
        "real-time prediction interface."
    )
