import dagshub
import mlflow

dagshub.init(
    repo_owner="NullBitZer0",
    repo_name="fraud-detection-system",
    mlflow=True
)

mlflow.set_experiment(
    "fraud_detection_experiment"
)