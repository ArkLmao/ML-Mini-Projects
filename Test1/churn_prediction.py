"""Customer churn prediction on the IBM Telco Customer Churn dataset.

Install: pip install pandas scikit-learn matplotlib
Run:     python churn_prediction.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save plots to a file instead of opening a window
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
LOCAL_FILE = Path("Telco-Customer-Churn.csv")
RANDOM_STATE = 42
TEST_SIZE = 0.2


# ----------------------------------------------------------------------
# 1. Data loading and cleaning
# ----------------------------------------------------------------------
def load_data():
    """Read the CSV from a local file if present, otherwise from the URL."""
    source = LOCAL_FILE if LOCAL_FILE.exists() else DATA_URL
    try:
        return pd.read_csv(source)
    except Exception as error:
        sys.exit(f"Could not load the dataset from {source}: {error}")


def clean_data(df):
    """Drop the ID column, fix TotalCharges and encode the target."""
    required = {"customerID", "TotalCharges", "Churn"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"Dataset is missing expected columns: {missing}")

    # Remove duplicates BEFORE dropping the ID, so two different customers
    # who happen to have identical features are not mistaken for duplicates.
    df = df.drop_duplicates().drop(columns=["customerID"])

    # TotalCharges contains blank strings, so it loads as text.
    # errors="coerce" turns those blanks into NaN (imputed later).
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    if df["Churn"].isna().any():
        sys.exit("Churn column has values other than 'Yes' and 'No'.")
    return df


# ----------------------------------------------------------------------
# 2. Preprocessing and models
# ----------------------------------------------------------------------
def build_pipeline(classifier, numeric_cols, categorical_cols):
    """Preprocessing + classifier in one Pipeline.

    Because everything lives in the Pipeline, medians, scaling and
    encodings are learned from the training data only (no data leakage).
    """
    numeric_steps = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_steps, numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ])
    return Pipeline([("preprocess", preprocessor), ("model", classifier)])


# ----------------------------------------------------------------------
# 3. Evaluation
# ----------------------------------------------------------------------
def evaluate(name, model, x_test, y_test):
    """Print metrics for one trained model and return its predictions."""
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]

    print(f"\n===== {name} =====")
    print(classification_report(
        y_test, predictions, target_names=["Stayed", "Churned"]
    ))
    print(f"ROC-AUC: {roc_auc_score(y_test, probabilities):.3f}")
    return predictions


def save_confusion_matrices(results, y_test, filename="confusion_matrices.png"):
    """Plot one confusion matrix per model side by side."""
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4))
    for axis, (name, predictions) in zip(axes, results.items()):
        ConfusionMatrixDisplay.from_predictions(
            y_test, predictions,
            display_labels=["Stayed", "Churned"], ax=axis, colorbar=False,
        )
        axis.set_title(name)
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)
    print(f"\nSaved confusion matrices to {filename}")


# ----------------------------------------------------------------------
# 4. Prediction for a new customer
# ----------------------------------------------------------------------
def predict_new_customer(model, customer, feature_columns):
    """Return (label, churn probability) for one customer dictionary."""
    missing = [col for col in feature_columns if col not in customer]
    if missing:
        raise ValueError(f"Customer data is missing fields: {missing}")
    row = pd.DataFrame([customer])[feature_columns]
    probability = model.predict_proba(row)[0, 1]
    label = "Likely to churn" if probability >= 0.5 else "Likely to stay"
    return label, probability


def main():
    df = clean_data(load_data())

    print(f"Rows: {len(df)}, columns: {df.shape[1]}")
    print("Missing values per column (only those with any):")
    print(df.isna().sum()[df.isna().sum() > 0])
    print("Churn share:")
    print(df["Churn"].value_counts(normalize=True).round(3))

    x = df.drop(columns=["Churn"])
    y = df["Churn"]
    feature_columns = list(x.columns)
    numeric_cols = x.select_dtypes(include="number").columns.tolist()
    categorical_cols = x.select_dtypes(exclude="number").columns.tolist()

    # stratify keeps the same churn ratio in train and test sets
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }

    trained, results = {}, {}
    for name, classifier in models.items():
        pipeline = build_pipeline(classifier, numeric_cols, categorical_cols)
        pipeline.fit(x_train, y_train)
        trained[name] = pipeline
        results[name] = evaluate(name, pipeline, x_test, y_test)

    save_confusion_matrices(results, y_test)

    # Example: predict for a new (made-up) customer
    new_customer = {
        "gender": "Female", "SeniorCitizen": 0, "Partner": "No",
        "Dependents": "No", "tenure": 3, "PhoneService": "Yes",
        "MultipleLines": "No", "InternetService": "Fiber optic",
        "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No",
        "StreamingTV": "No", "StreamingMovies": "No",
        "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 75.0, "TotalCharges": 225.0,
    }
    label, probability = predict_new_customer(
        trained["Logistic Regression"], new_customer, feature_columns
    )
    print(f"\nNew customer: {label} (churn probability {probability:.2f})")


if __name__ == "__main__":
    main()
