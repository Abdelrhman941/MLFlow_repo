import mlflow
from mlflow.exceptions import MlflowException

# ========================= [1] Setup Tracking URI =========================
print("Tracking URI:", mlflow.get_tracking_uri())       # to ensure it's set correctly
print('-' * 50)

# ========================= [2] Create Experiment =========================
try:
    exp_id = mlflow.create_experiment("app1_experiment")
    print("Experiment created with ID:", exp_id)
except MlflowException:
    print("Experiment already exists, skipping...")
print('-' * 50)
# ========================= [3] Set Active Experiment =========================
exp = mlflow.set_experiment("app1_experiment")
print(f"Active experiment set to: {exp.name}")
print('-' * 50)
# ========================= [4] Run #1 (Named Run) =========================
with mlflow.start_run(run_name="baseline_model") as run:
    print("Started run:", run.info.run_name)
print('-' * 50)
# ========================= [5] Run #2 (Auto-Named Run with Tag) =========================
with mlflow.start_run() as run:
    mlflow.set_tag("version", "v1.0_baseline")
    print("Started auto-named run with tag 'version=v1.0_baseline'")
print('-' * 50)
# ========================= [6] Log Params and Metrics =========================
mlflow.log_param("max_depth", 5)
mlflow.log_param("n_estimators", 100)
mlflow.log_metric("accuracy", 0.82)
print("Logged params and metrics")
print('-' * 50)
# ========================= [7] Retrieve Experiment by Name =========================
exp_name = mlflow.get_experiment_by_name(name="app1_experiment")
if exp_name:
    print("Found experiment 'app1_experiment' with ID:", exp_name.experiment_id)
else:
    print("Experiment 'app1_experiment' not found")