import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_clean_data(url: str) -> pd.DataFrame:
    """Load the dataset from a URL and perform initial data cleaning."""
    print("Loading dataset...")
    df = pd.read_csv(url)

    # Drop non-predictive identifier column
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Fix TotalCharges: convert empty whitespace strings to NaN, then to float
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].str.strip(), errors="coerce")

    # Handle missing values in TotalCharges by filling with the column median
    total_charges_median = df["TotalCharges"].median()
    df["TotalCharges"] = df["TotalCharges"].fillna(total_charges_median)

    # Convert binary target 'Churn' to numeric (1 = Yes, 0 = No)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    return df


def preprocess_features(df: pd.DataFrame):
    """Separate features and target, and apply one-hot encoding to categorical features."""
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # One-hot encode categorical features while preserving numeric columns
    X = pd.get_dummies(X, drop_first=True)

    return X, y


def main():
    # 1. Dataset URL
    dataset_url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"

    # 2. Load and Clean Data
    df = load_and_clean_data(dataset_url)

    # 3. Preprocess Features
    X, y = preprocess_features(df)

    # 4. Train/Test Split (80% Train, 20% Test)
    # Use stratify=y to ensure equal churn proportion in both train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 5. Feature Scaling
    # Fit scaler ONLY on training data to prevent data leakage
    numerical_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

    scaler = StandardScaler()
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

    # 6. Model Training
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",  # Adjusts for class imbalance automatically
    )
    model.fit(X_train, y_train)

    # 7. Model Evaluation
    print("Evaluating model performance...\n")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Metrics computation
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    # Output Results
    print("=" * 40)
    print("        MODEL EVALUATION RESULTS        ")
    print("=" * 40)
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")
    print("=" * 40)
    print("\nDetailed Classification Report:")
    print(
        classification_report(y_test, y_pred, target_names=["No Churn", "Churn"])
    )


if __name__ == "__main__":
    main()