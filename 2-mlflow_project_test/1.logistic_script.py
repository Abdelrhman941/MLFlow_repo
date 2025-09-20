import os, argparse, warnings, mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn_features.transformers import DataFrameSelector
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, roc_curve, auc
warnings.filterwarnings("ignore")

# ========================= [1] Data Preparation =========================
def load_and_clean_data(path: str):
    """Load dataset and drop unused features + filter outliers."""
    df = pd.read_csv(path)
    
    # Drop irrelevant columns
    df.drop(columns=["RowNumber", "CustomerId", "Surname"], inplace=True)
    
    # Remove unrealistic ages > 80
    df = df[df["Age"] <= 80]
    return df

BASE_DIR = os.path.dirname(__file__) 
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "dataset.csv")
df = load_and_clean_data(DATA_PATH)

X = df.drop(columns=["Exited"])
y = df["Exited"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y, random_state=45)

# ========================= [2] Data Processing Pipelines =========================
NUM_COLS   = ["Age", "CreditScore", "Balance", "EstimatedSalary"]
CAT_COLS   = ["Gender", "Geography"]
READY_COLS = list(set(X_train.columns) - set(NUM_COLS) - set(CAT_COLS))

# Numerical pipeline
num_pipeline = Pipeline(
    steps=[
        ("select", DataFrameSelector(NUM_COLS)),
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]
)

# Categorical pipeline
cat_pipeline = Pipeline(
    steps=[
        ("select", DataFrameSelector(CAT_COLS)),
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(drop="first", sparse_output=False)),
    ]
)

# Ready pipeline (binary/ordinal features already in good format)
ready_pipeline = Pipeline(
    steps=[
        ("select", DataFrameSelector(READY_COLS)),
        ("impute", SimpleImputer(strategy="most_frequent")),
    ]
)

# Combine all into one
full_pipeline = FeatureUnion(
    transformer_list=[
        ("numerical", num_pipeline),
        ("categorical", cat_pipeline),
        ("ready", ready_pipeline),
    ]
)

# Apply transformations
X_train_final = full_pipeline.fit_transform(X_train)
X_test_final  = full_pipeline.transform(X_test)

# ========================= [3] Handling Imbalance =========================
# Class weights approach
class_ratios  = 1 - (np.bincount(y_train) / len(y_train))
class_ratios  = class_ratios / np.sum(class_ratios)
CLASS_WEIGHTS = {i: class_ratios[i] for i in range(2)}

# SMOTE oversampling
smote = SMOTE(sampling_strategy=0.7, random_state=45)
X_train_smote, y_train_smote = smote.fit_resample(X_train_final, y_train)

# ========================= [4] Model Training & Logging =========================
def train_and_log(X_train, y_train, plot_suffix: str, C: float, penalty: str, class_weight=None):
    """Train Logistic Regression, evaluate, and log everything in MLflow."""
    mlflow.set_experiment("Churn-Detection")
    with mlflow.start_run(run_name=f"LR_{plot_suffix}"):
        mlflow.set_tag('clf', 'LogisticRegression')
        
        # ---- Train ----
        clf = LogisticRegression(C=C, penalty=penalty, random_state=45, class_weight=class_weight)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test_final)
        y_prob = clf.predict_proba(X_test_final)[:, 1]
        
        # ---- Metrics ----
        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred)
        
        # ---- Log Params & Metrics ----
        mlflow.log_params({"C": C, "penalty": penalty})
        mlflow.log_metrics({"accuracy": acc, "f1_score": f1})
        
        # ---- Log Model (with signature & example) ----
        from mlflow.models.signature import infer_signature
        signature = infer_signature(X_test_final, y_pred)
        mlflow.sklearn.log_model(sk_model=clf, artifact_path=f"LogisticRegression_{plot_suffix}", input_example=X_test_final[:5], signature=signature)
        
        # ---- Confusion Matrix ----
        plt.figure(figsize=(8, 5))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", cbar=False,)
        plt.title(f"Confusion Matrix - {plot_suffix}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        mlflow.log_figure(plt.gcf(), f"{plot_suffix}_conf_matrix.png")
        plt.close()
        
        # ---- ROC Curve ----
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.2f}")
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.title(f"ROC Curve - {plot_suffix}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        mlflow.log_figure(plt.gcf(), f"{plot_suffix}_roc_curve.png")
        plt.close()
        print(f"[Run Complete ✅] {plot_suffix:<15} | Acc={acc:.3f}, F1={f1:.3f}")

# ========================= [5] Entrypoint =========================
def main(C: float, penalty: str):
    train_and_log(X_train_final, y_train, plot_suffix="baseline", C=C, penalty=penalty)         # Run with original imbalanced data
    train_and_log(X_train_final, y_train, plot_suffix="class_weights", C=C, penalty=penalty, class_weight=CLASS_WEIGHTS)    # Run with class weights
    train_and_log(X_train_smote, y_train_smote, plot_suffix="SMOTE", C=C, penalty=penalty)      # Run with SMOTE oversampled data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Churn Detection with Logistic Regression + MLflow")
    parser.add_argument("--C", "-c", type=float, default=2.5, help="Inverse regularization strength")
    parser.add_argument("--penalty", "-p", type=str, default="l2", help="Regularization type (l1, l2, elasticnet)")
    args = parser.parse_args()
    main(C=args.C, penalty=args.penalty)