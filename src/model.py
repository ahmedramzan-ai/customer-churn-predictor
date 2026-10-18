
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV
)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)


# =========================================================
# 1. LOAD DATASET
# =========================================================

df = pd.read_csv(
    "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

print("Dataset shape:", df.shape)

print("\nTarget distribution:")
print(df["Churn"].value_counts())


# =========================================================
# 2. CLEAN DATA
# =========================================================

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

print("\nTotalCharges Data Type:")
print(df["TotalCharges"].dtype)

print("\nMissing TotalCharges:")
print(df["TotalCharges"].isna().sum())

# Remove rows with missing TotalCharges
df = df.dropna(subset=["TotalCharges"])

print("\nShape after cleaning:")
print(df.shape)

print("\nMissing values after cleaning:")
print(df.isnull().sum().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())


# =========================================================
# 3. REMOVE CUSTOMER ID
# =========================================================

df = df.drop(
    "customerID",
    axis=1
)


# =========================================================
# 4. CONVERT TARGET
# =========================================================

df["Churn"] = df["Churn"].map({
    "No": 0,
    "Yes": 1
})


# =========================================================
# 5. SEPARATE FEATURES AND TARGET
# =========================================================

X = df.drop(
    "Churn",
    axis=1
)

y = df["Churn"]


# =========================================================
# 6. ONE-HOT ENCODING
# =========================================================

X = pd.get_dummies(
    X,
    drop_first=True
)

feature_columns = X.columns.tolist()

print("\nFeatures after encoding:")
print(X.shape)

print("\nTarget:")
print(y.shape)


# =========================================================
# 7. TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining data:")
print(X_train.shape)

print("Testing data:")
print(X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())


# =========================================================
# 8. LOGISTIC REGRESSION PIPELINE
# =========================================================

model = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        )
    )
])


# =========================================================
# 9. TRAIN MODEL
# =========================================================

model.fit(
    X_train,
    y_train
)

print("\nModel training completed!")


# =========================================================
# 10. PREDICTIONS
# =========================================================

y_pred = model.predict(
    X_test
)

y_prob = model.predict_proba(
    X_test
)[:, 1]


print("\nFirst 20 predictions:")
print(y_pred[:20])

print("\nActual values:")
print(y_test.values[:20])


# =========================================================
# 11. MODEL EVALUATION - DEFAULT 0.50 THRESHOLD
# =========================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)

print("\nModel Evaluation:")
print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))


# =========================================================
# 12. CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)


# Confusion Matrix Visualization
plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Churn", "Churn"],
    yticklabels=["No Churn", "Churn"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Customer Churn Confusion Matrix")

plt.tight_layout()
plt.show()


# =========================================================
# 13. RANDOM FOREST COMPARISON
# =========================================================

rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

rf_model.fit(
    X_train,
    y_train
)

rf_pred = rf_model.predict(
    X_test
)

rf_accuracy = accuracy_score(
    y_test,
    rf_pred
)

rf_precision = precision_score(
    y_test,
    rf_pred
)

rf_recall = recall_score(
    y_test,
    rf_pred
)

rf_f1 = f1_score(
    y_test,
    rf_pred
)

print("\nRandom Forest Evaluation:")
print("Accuracy :", round(rf_accuracy, 4))
print("Precision:", round(rf_precision, 4))
print("Recall   :", round(rf_recall, 4))
print("F1 Score :", round(rf_f1, 4))


# =========================================================
# 14. LOGISTIC REGRESSION FEATURE IMPORTANCE
# =========================================================

feature_names = X.columns

coefficients = model.named_steps[
    "classifier"
].coef_[0]

feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients
})

feature_importance["AbsoluteCoefficient"] = (
    feature_importance["Coefficient"].abs()
)

feature_importance = feature_importance.sort_values(
    "AbsoluteCoefficient",
    ascending=False
)

print("\nTop 15 Important Features:")

print(
    feature_importance[
        ["Feature", "Coefficient"]
    ].head(15)
)


# =========================================================
# 15. FEATURE IMPORTANCE VISUALIZATION
# =========================================================

top_features = (
    feature_importance
    .head(10)
    .sort_values("Coefficient")
)

plt.figure(figsize=(10, 6))

plt.barh(
    top_features["Feature"],
    top_features["Coefficient"]
)

plt.xlabel(
    "Logistic Regression Coefficient"
)

plt.ylabel("Feature")

plt.title(
    "Top 10 Features Associated with Customer Churn"
)

plt.tight_layout()
plt.show()


# =========================================================
# 16. 5-FOLD CROSS VALIDATION
# =========================================================

cv_scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=5,
    scoring="f1"
)

print("\n5-Fold Cross Validation F1 Scores:")
print(cv_scores)

mean_cv_f1 = cv_scores.mean()

print("\nMean Cross Validation F1 Score:")
print(mean_cv_f1)


# =========================================================
# 17. HYPERPARAMETER TUNING
# =========================================================

param_grid = {
    "classifier__C": [
        0.01,
        0.1,
        1,
        10,
        100
    ]
}

grid_search = GridSearchCV(
    model,
    param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

grid_search.fit(
    X_train,
    y_train
)

print("\nBest Parameters:")
print(grid_search.best_params_)

print("\nBest Cross Validation F1 Score:")
print(grid_search.best_score_)

# Use the best model
best_model = grid_search.best_estimator_


# =========================================================
# 18. THRESHOLD ANALYSIS
# =========================================================

best_y_prob = best_model.predict_proba(
    X_test
)[:, 1]

print("\nThreshold Analysis:")

thresholds = [
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80
]

best_threshold = 0.65
best_threshold_f1 = 0

for threshold in thresholds:

    y_threshold = (
        best_y_prob >= threshold
    ).astype(int)

    threshold_f1 = f1_score(
        y_test,
        y_threshold
    )

    threshold_precision = precision_score(
        y_test,
        y_threshold
    )

    threshold_recall = recall_score(
        y_test,
        y_threshold
    )

    print(
        f"Threshold: {threshold:.2f} | "
        f"Precision: {threshold_precision:.4f} | "
        f"Recall: {threshold_recall:.4f} | "
        f"F1: {threshold_f1:.4f}"
    )

    # Automatically track best F1 threshold
    if threshold_f1 > best_threshold_f1:
        best_threshold_f1 = threshold_f1
        best_threshold = threshold


print("\nSelected Threshold:")
print(round(best_threshold, 2))

print("\nBest Threshold F1:")
print(round(best_threshold_f1, 4))


# =========================================================
# 19. FINAL THRESHOLD EVALUATION
# =========================================================

final_prediction = (
    best_y_prob >= best_threshold
).astype(int)

final_accuracy = accuracy_score(
    y_test,
    final_prediction
)

final_precision = precision_score(
    y_test,
    final_prediction
)

final_recall = recall_score(
    y_test,
    final_prediction
)

final_f1 = f1_score(
    y_test,
    final_prediction
)

final_cm = confusion_matrix(
    y_test,
    final_prediction
)

print("\nFinal Threshold Evaluation:")

print(
    "Accuracy :",
    round(final_accuracy, 4)
)

print(
    "Precision:",
    round(final_precision, 4)
)

print(
    "Recall   :",
    round(final_recall, 4)
)

print(
    "F1 Score :",
    round(final_f1, 4)
)

print("\nFinal Confusion Matrix:")
print(final_cm)


# =========================================================
# 20. ROC-AUC
# =========================================================

roc_auc = roc_auc_score(
    y_test,
    best_y_prob
)

print("\nROC-AUC Score:")
print(round(roc_auc, 4))


# =========================================================
# 21. ROC CURVE
# =========================================================

fpr, tpr, roc_thresholds = roc_curve(
    y_test,
    best_y_prob
)

plt.figure(figsize=(7, 5))

plt.plot(
    fpr,
    tpr,
    label=f"Logistic Regression (AUC = {roc_auc:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title(
    "ROC Curve - Customer Churn"
)

plt.legend()

plt.tight_layout()

plt.show()


# =========================================================
# 22. SAVE FINAL MODEL
# =========================================================

model_data = {
    "model": best_model,
    "features": feature_columns,
    "threshold": best_threshold,
    "roc_auc": roc_auc,
    "accuracy": final_accuracy,
    "precision": final_precision,
    "recall": final_recall,
    "f1": final_f1
}

joblib.dump(
    model_data,
    "models/churn_model.pkl"
)

print("\nModel saved successfully!")

print(
    "Saved file: models/churn_model.pkl"
)

print("\nModel and feature structure saved successfully!")