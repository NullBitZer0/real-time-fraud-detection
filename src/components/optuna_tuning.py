"""Optuna hyperparameter tuning for CatBoost, with MLflow tracking.

Can be run standalone or called from the training pipeline.

Standalone usage:
    python -m src.components.optuna_tuning --n-trials 50

Called from training pipeline:
    from src.components.optuna_tuning import run_optuna
    best_params = run_optuna(X_train, y_train, X_val, y_val, mlflow_tracker=tracker)
"""
import argparse
import json
import os
import time

from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score

from src.utils.logger import logging

SEARCH_SPACE = {
    "iterations":          (300, 1500),
    "depth":               (4, 10),
    "learning_rate":       (0.01, 0.15),
    "l2_leaf_reg":         (0.5, 20.0),
    "random_strength":     (0.0, 5.0),
    "bagging_temperature": (0.0, 2.0),
    "border_count":        (32, 254),
}


def _suggest(trial, param: str, bounds: tuple):
    lo, hi = bounds
    if param in ("learning_rate", "l2_leaf_reg"):
        return trial.suggest_float(param, lo, hi, log=True)
    if param in ("iterations", "depth", "border_count"):
        return trial.suggest_int(param, int(lo), int(hi))
    return trial.suggest_float(param, lo, hi)


def run_optuna(X_train, y_train, X_val, y_val,
               n_trials: int = 50,
               timeout_minutes: int = 0,
               mlflow_tracker=None,
               study_name: str = "catboost-optuna",
               seed: int = 42) -> dict:
    """Run Optuna search for CatBoost hyperparameters.

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data (used for early stopping + objective)
        n_trials: Number of Optuna trials
        timeout_minutes: Max search time (0 = no limit)
        mlflow_tracker: Optional MLflowTracker instance for logging
        study_name: Name for the Optuna study
        seed: Random seed

    Returns:
        dict of best hyperparameters (catboost kwargs)
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    timeout_seconds = timeout_minutes * 60 if timeout_minutes > 0 else None

    logging.info(
        f"Optuna search starting — {n_trials} trials, "
        f"{'no timeout' if timeout_seconds is None else f'{timeout_minutes}min timeout'}"
    )

    if mlflow_tracker is not None:
        mlflow_tracker.log_params({
            "optuna_n_trials": n_trials,
            "optuna_timeout_minutes": timeout_minutes or "none",
            "optuna_study": study_name,
        })

    def objective(trial):
        params = {
            "eval_metric": "PRAUC",
            "random_seed": seed,
            "verbose": 0,
            "early_stopping_rounds": 30,
            "task_type": "CPU",
        }
        for param, bounds in SEARCH_SPACE.items():
            params[param] = _suggest(trial, param, bounds)

        try:
            m = CatBoostClassifier(**params)
            m.fit(X_train, y_train, eval_set=(X_val, y_val))
            pr = average_precision_score(y_val, m.predict_proba(X_val)[:, 1])
        except Exception as e:
            logging.warning(f"Trial {trial.number} failed: {e}")
            return 0.0

        if mlflow_tracker is not None:
            import mlflow
            mlflow.log_metrics({f"trial_{trial.number}_pr_auc": pr}, step=trial.number)
            mlflow.log_params({f"trial_{trial.number}_{k}": v for k, v in params.items()})

        return pr

    t0 = time.time()
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=10),
        study_name=study_name,
    )

    def _progress(study, trial):
        if (trial.number + 1) % 5 == 0 or trial.number == 0:
            elapsed = time.time() - t0
            logging.info(
                f"Trial {trial.number + 1}/{n_trials} | "
                f"best={study.best_value:.4f} | current={trial.value:.4f} | "
                f"elapsed={elapsed:.0f}s"
            )

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout_seconds,
        callbacks=[_progress],
        show_progress_bar=False,
    )

    elapsed = time.time() - t0
    logging.info(f"Optuna done in {elapsed / 60:.1f} min | best PR-AUC={study.best_value:.4f}")

    best_params = {
        "iterations":           study.best_params["iterations"],
        "depth":                study.best_params["depth"],
        "learning_rate":        study.best_params["learning_rate"],
        "l2_leaf_reg":          study.best_params["l2_leaf_reg"],
        "random_strength":      study.best_params["random_strength"],
        "bagging_temperature":  study.best_params["bagging_temperature"],
        "border_count":         study.best_params["border_count"],
        "eval_metric":          "PRAUC",
        "random_seed":          seed,
        "verbose":              0,
        "early_stopping_rounds": 50,
        "task_type":            "CPU",
    }

    save_path = "models/optuna_best.json"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump({"best_params": best_params, "best_value": study.best_value}, f, indent=2)
    logging.info(f"Best params saved → {save_path}")

    if mlflow_tracker is not None:
        mlflow_tracker.log_metrics({
            "optuna_best_pr_auc": study.best_value,
            "optuna_duration_min": round(elapsed / 60, 1),
            "optuna_trials_completed": len(study.trials),
        })
        mlflow_tracker.log_params({f"best_{k}": v for k, v in best_params.items()})

    return best_params


def main():
    parser = argparse.ArgumentParser(description="Optuna tuning for CatBoost")
    parser.add_argument("--n-trials", type=int, default=50, help="Number of trials")
    parser.add_argument("--timeout-min", type=int, default=0, help="Max search time (min)")
    parser.add_argument("--mlflow", action="store_true", help="Enable MLflow tracking")
    args = parser.parse_args()

    from src.components.data_ingestion import DataIngestion, read_sparkov_split
    from src.components.feature_engineering import FeatureEngineering

    di = DataIngestion()
    train_path, val_path, test_path = di.initiate_data_ingestion()
    train_df = read_sparkov_split(train_path)
    val_df   = read_sparkov_split(val_path)

    fe = FeatureEngineering()
    train_fe = fe.fit_transform(train_df, compute_velocity=True)
    val_fe   = fe.transform(val_df, compute_velocity=True)
    features  = fe.feature_list

    X_train = train_fe[features].values.astype("float32")
    y_train = train_fe["is_fraud"].values
    X_val   = val_fe[features].values.astype("float32")
    y_val   = val_fe["is_fraud"].values

    tracker = None
    if args.mlflow:
        from src.components.mlflow_tracking import MLflowTracker
        tracker = MLflowTracker({
            "tracking_uri": os.environ.get(
                "MLFLOW_TRACKING_URI",
                os.environ.get("DAGSHUB_MLFLOW_URI", "file:./mlruns")
            ),
            "experiment_name": "sparkov-optuna",
            "run_name": f"optuna-{args.n_trials}-trials",
            "register_model": False,
        })
        tracker.start()

    best = run_optuna(X_train, y_train, X_val, y_val,
                       n_trials=args.n_trials,
                       timeout_minutes=args.timeout_min,
                       mlflow_tracker=tracker)

    if tracker:
        tracker.end()

    print(f"\nBest params: {json.dumps(best, indent=2)}")


if __name__ == "__main__":
    main()
