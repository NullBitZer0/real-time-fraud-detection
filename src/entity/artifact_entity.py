from dataclasses import dataclass


@dataclass
class DataIngestionArtifact:
    train_path: str
    test_path:  str


@dataclass
class ModelTrainerArtifact:
    model_path:  str
    roc_auc:     float
    pr_auc:      float


@dataclass
class ModelEvaluationArtifact:
    pr_auc:     float
    roc_auc:    float
    f1:         float
    precision:  float
    recall:     float
    threshold:  float


@dataclass
class ModelRegistryArtifact:
    model_name:    str
    model_version: str
    run_id:        str
    stage:         str


@dataclass
class ModelPusherArtifact:
    model_name:    str
    model_version: str
    run_id:        str
    stage:         str
    dst_path:      str