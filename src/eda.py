import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

print(df.head())

print("\nShape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

print("\nTotalCharges Data Type:")
print(df["TotalCharges"].dtype)

print("\nMissing TotalCharges after conversion:")
print(df["TotalCharges"].isnull().sum())

# Remove rows with missing TotalCharges
df = df.dropna(subset=["TotalCharges"])

print("\nShape after cleaning:")
print(df.shape)

print("\nMissing values after cleaning:")
print(df.isnull().sum().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Churn distribution
print("\nChurn Distribution After Cleaning:")
print(df["Churn"].value_counts())

# Churn percentage by contract type
contract_churn = pd.crosstab(
    df["Contract"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn Percentage by Contract Type:")
print(contract_churn)

# Plot
contract_churn.plot(kind="bar", figsize=(8, 5))

plt.title("Churn Percentage by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Percentage")
plt.xticks(rotation=0)
plt.legend(title="Churn")
plt.tight_layout()
plt.close()
# Create tenure groups
df["TenureGroup"] = pd.cut(
    df["tenure"],
    bins=[0, 12, 24, 48, 72],
    labels=["0-12", "13-24", "25-48", "49-72"]
)

# Calculate churn percentage by tenure group
tenure_churn = pd.crosstab(
    df["TenureGroup"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn Percentage by Tenure Group:")
print(tenure_churn)

# Plot
tenure_churn.plot(kind="bar", figsize=(8, 5))

plt.title("Churn Percentage by Tenure Group")
plt.xlabel("Tenure (Months)")
plt.ylabel("Percentage")
plt.xticks(rotation=0)
plt.legend(title="Churn")
plt.tight_layout()
plt.close()
# Create monthly charge groups
df["MonthlyChargeGroup"] = pd.qcut(
    df["MonthlyCharges"],
    q=4,
    labels=["Low", "Medium-Low", "Medium-High", "High"]
)

# Churn percentage by monthly charge group
charge_churn = pd.crosstab(
    df["MonthlyChargeGroup"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn Percentage by Monthly Charge Group:")
print(charge_churn)
