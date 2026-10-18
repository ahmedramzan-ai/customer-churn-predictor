# 📊 Customer Churn Predictor

An end-to-end machine learning application that predicts whether a telecom customer is likely to churn based on customer information, services, contract details, payment method, tenure, and billing information.

The project combines **Machine Learning + Streamlit + Data Analytics** to provide both individual churn predictions and business-level customer churn insights.

---

## 🚀 Live Demo

Try the deployed application:

https://customer-churn-predictor-jdecwk3plakufdskycuvzh.streamlit.app/

---

## 🚀 Features

- 📊 Customer churn prediction
- 🤖 Logistic Regression model
- 🌲 Random Forest comparison
- ⚖️ Balanced class weights
- 🎯 Probability threshold optimization
- 📈 ROC-AUC evaluation
- 🔄 5-Fold Cross-Validation
- 🔍 Prediction explanation
- 📊 Interactive analytics dashboard
- 💡 Business insights
- 💾 Saved trained ML model
- 🌐 Streamlit web application

---

## 🧠 Machine Learning Workflow

```text
Raw Dataset
     ↓
Data Cleaning
     ↓
Missing Value Handling
     ↓
Feature Encoding
     ↓
Train/Test Split
     ↓
Feature Scaling
     ↓
Logistic Regression
     ↓
Model Evaluation
     ↓
Threshold Optimization
     ↓
Saved Model
     ↓
Streamlit Application
```

---

## 📂 Project Structure

```text
customer-churn-predictor/
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── models/
│   └── churn_model.pkl
│
├── notebooks/
│
├── src/
│   ├── eda.py
│   └── model.py
│
├── screenshots/
│   ├── prediction.png
│   ├── model-performance.png
│   └── analytics-dashboard.png
│
├── app.py
├── app_backup.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📊 Dataset

The project uses the **Telco Customer Churn dataset**.

After cleaning:

- **7,032 customer records**
- **30 encoded features**
- Target variable: `Churn`

Target classes:

```text
No  → 0
Yes → 1
```

The `customerID` column was removed because it does not provide useful predictive information.

`TotalCharges` was converted to numeric values and rows with missing values were removed.

---

## 🤖 Models Used

### Logistic Regression

The primary model uses:

- StandardScaler
- Logistic Regression
- Balanced class weights
- Maximum iterations: 1000

### Random Forest

A Random Forest model was also trained for comparison.

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 78.11% | 57.50% | 67.65% | 62.16% |
| Random Forest | 78.68% | 62.09% | 50.80% | 55.88% |

Logistic Regression was selected as the primary model because it achieved better recall and F1 performance at the selected probability threshold.

---

## 🎯 Probability Threshold Optimization

Instead of simply using the default probability threshold of `0.50`, multiple thresholds were evaluated.

The selected threshold is:

```text
0.65
```

At this threshold:

- Accuracy: **78.11%**
- Precision: **57.50%**
- Recall: **67.65%**
- F1 Score: **62.16%**
- ROC-AUC: **0.835**

The threshold was selected based on the highest F1 score found during threshold analysis.

---

## 🔄 Cross-Validation

The Logistic Regression model was evaluated using **5-Fold Cross-Validation**.

```text
Mean F1 Score: 63.24%
```

This provides an additional estimate of how consistently the model performs across different training and validation splits.

---

## 🔍 Important Features

Some of the important features identified by the Logistic Regression coefficients include:

- Tenure
- Monthly Charges
- Internet Service
- Contract type
- Total Charges
- Streaming services
- Payment Method
- Online Security
- Tech Support

These features help explain factors associated with customer churn in the dataset.

---

## 🌐 Streamlit Application

The Streamlit application allows users to enter customer information such as:

### 👤 Customer Information

- Gender
- Senior Citizen
- Partner
- Dependents
- Tenure
- Phone Service

### 🌐 Internet & Services

- Multiple Lines
- Internet Service
- Online Security
- Online Backup
- Device Protection
- Tech Support
- Streaming TV
- Streaming Movies

### 💳 Billing Information

- Contract
- Paperless Billing
- Payment Method
- Monthly Charges
- Total Charges

The application provides:

- Churn probability
- Churn prediction
- Risk level
- Risk gauge
- Prediction explanation
- Model performance
- Model comparison
- Cross-validation results
- Customer analytics

---

## 📊 Analytics Dashboard

The application provides business-level analytics including:

- Total customers
- Churned customers
- Overall churn rate
- Average monthly charges
- Churn distribution
- Churn by contract
- Churn by internet service
- Churn by payment method
- Churn by tenure

### 💡 Key Dataset Insights

- Overall churn rate: **26.58%**
- Month-to-month customers show the highest contract-based churn rate.
- Electronic-check customers show relatively high churn.
- Fiber-optic customers show relatively high churn in this dataset.

These insights can help businesses identify customer segments that may require targeted retention strategies.

---

## 📸 Application Screenshots

Screenshots of the deployed application can be added here.

### 🔮 Churn Prediction

![Churn Prediction](screenshots/prediction.png)

### 📈 Model Performance

![Model Performance](screenshots/model-performance.png)

### 📊 Analytics Dashboard

![Analytics Dashboard](screenshots/analytics-dashboard.png)

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/ahmedramzan-ai/customer-churn-predictor.git
cd customer-churn-predictor
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 🧪 Train the Model

To retrain the machine learning model:

```bash
python src/model.py
```

The trained model will be saved inside:

```text
models/churn_model.pkl
```

---

## 🛠️ Technologies Used

- **Python**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **Matplotlib**
- **Seaborn**
- **Joblib**
- **Streamlit**

---

## 📈 Performance Summary

```text
Accuracy   : 78.11%
Precision  : 57.50%
Recall     : 67.65%
F1 Score   : 62.16%
ROC-AUC    : 0.835
CV F1      : 63.24%
```

---

## 🔮 Future Improvements

Possible future improvements include:

- Advanced models such as XGBoost and LightGBM
- SHAP-based explainable AI
- Automated customer retention recommendations
- Customer segmentation
- Real-time database integration
- Authentication and user accounts
- Automated model retraining
- Model monitoring
- REST API integration
- Improved deployment infrastructure

---

## 🎯 Project Goal

The goal of this project is to demonstrate a complete machine learning workflow from **raw customer data to a deployed predictive application**.

The system can help businesses identify customers who may be at risk of leaving and provide insights that support customer retention strategies.

---

## 📜 License

This project is intended for educational and demonstration purposes.
# Application Screenshots

## 📸 Streamlit Application

### 1. Customer Churn Prediction
![Customer Churn Prediction](screenshots/Screenshot%202026-08-25%20165301.png)

### 2. Prediction Result
![Prediction Result](screenshots/Screenshot%202026-08-25%20165403.png)

### 3. Analytics Dashboard
![Analytics Dashboard](screenshots/Screenshot%202026-08-25%20165414.png)
