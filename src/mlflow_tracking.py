import dagshub
import mlflow

dagshub.init(
    repo_owner="NullBitZer0",
    repo_name="real-time-fraud-detection",
    mlflow=True
)

mlflow.set_experiment(
    "fraud_detection_experiment"
)
