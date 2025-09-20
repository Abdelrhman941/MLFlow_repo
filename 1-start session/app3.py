import mlflow
import argparse
import pandas as pd
import warnings
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# ========================= [0] Clean Console =========================
warnings.filterwarnings("ignore")
logging.getLogger("mlflow").setLevel(logging.ERROR)

print(f"Tracking URI: {mlflow.get_tracking_uri()}")
print('-' * 50)

# ========================= [1] Load Dataset =========================
DATASET_URL = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
df = pd.read_csv(DATASET_URL, sep=";")

X = df.drop(columns=["quality"])
y = df["quality"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=45)

# ========================= [2] Evaluation Function =========================
def evaluate(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    return acc, f1

# ========================= [3] Training & Logging =========================
def main(n_estimators, max_depth):
    mlflow.set_experiment("app3_experiment")
    with mlflow.start_run(run_name=f"RF_n{n_estimators}_d{max_depth}"):
        mlflow.log_params({"n_estimators": n_estimators,"max_depth": max_depth})
        mlflow.set_tags({"model": "RandomForest","dataset": "WineQuality-Red"})
        clf = RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,random_state=45)
        clf.fit(X_train, y_train)
        
        y_pred_test = clf.predict(X_test)
        acc, f1     = evaluate(y_test, y_pred_test)
        
        mlflow.log_metrics({"Accuracy": acc, "F1_score": f1})
        
        input_example = X_test.iloc[:1]
        mlflow.sklearn.log_model(clf, artifact_path="RandomForestModel", input_example=input_example)
        
        print(f"[Run Complete ✅] Accuracy={acc:.3f} | F1_score={f1:.3f}")

# ========================= [4] CLI Entrypoint =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", "-n", type=int, default=350)
    parser.add_argument("--max_depth", "-d", type=int, default=15)
    args = parser.parse_args()
    main(args.n_estimators, args.max_depth)
