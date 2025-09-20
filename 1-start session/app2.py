import os
import mlflow
import argparse
import time
import warnings
import logging

# ========================= [0] Clean Console =========================
warnings.filterwarnings("ignore") 
logging.getLogger("mlflow").setLevel(logging.ERROR)
print(f"Tracking URI: {mlflow.get_tracking_uri()}")
print('-' * 50)

# ========================= [1] Simple Evaluation Function =========================
def evaluate_mean(param_1, param_2):
    """Return the mean of two parameters."""
    return (param_1 + param_2) / 2

# ========================= [2] Example Table to Log =========================
table_dict = {
    "Column_A": [10, 15, 22, 70],
    "Column_B": [15, 20, 30, 40],
}

# ========================= [3] Main Experiment Function =========================
def main(param_1, param_2):
    # --- Create / Set experiment ---
    mlflow.set_experiment(experiment_name="app2_experiment")
    
    # --- Start Run ---
    with mlflow.start_run(run_name="Run_Mode") as run:
        mlflow.set_tag("version", "1.0.0")
        
        # --- Log parameters ---
        mlflow.log_param("param_1", param_1)
        mlflow.log_param("param_2", param_2)
        
        # --- Log metric (mean of params) ---
        mlflow.log_metric("mean", evaluate_mean(param_1, param_2))
        
        # --- Create artifacts folder and log it ---
        os.makedirs("Output", exist_ok=True)
        with open("Output/artifact_info.txt", "w") as f:
            f.write(f"Artifact created at : {time.asctime()}\n")
            f.write(f"Parameters used     : param_1={param_1}, param_2={param_2}\n")
        mlflow.log_artifacts("Output")
        
        # --- Log example table as JSON artifact ---
        mlflow.log_table(table_dict, artifact_file="Output/logged_table.json")
        print(f"[Run Complete ✅] Run ID: {run.info.run_id}")

# ========================= [4] Entry Point (CLI) =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo MLflow Experiment Runner")
    parser.add_argument("--param_1", "-p1", type=int, default=10, help="First parameter")
    parser.add_argument("--param_2", "-p2", type=int, default=20, help="Second parameter")
    args = parser.parse_args()
    main(param_1=args.param_1, param_2=args.param_2)