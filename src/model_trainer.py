import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

def get_base_models(random_state: int = 42) -> Dict[str, Any]:
    """
    Initializes high-performing classifiers with regularized & tuned hyperparams.
    """
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=0.8, penalty="l2", random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_split=5, min_samples_leaf=2, max_features="sqrt", random_state=random_state),
        "Extra Trees": ExtraTreesClassifier(n_estimators=200, max_depth=5, min_samples_split=4, min_samples_leaf=2, random_state=random_state),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.03, max_depth=3, subsample=0.85, random_state=random_state),
        "XGBoost": XGBClassifier(
            n_estimators=120,
            learning_rate=0.03,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=random_state,
            eval_metric="logloss"
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=100,
            learning_rate=0.03,
            num_leaves=12,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            verbosity=-1
        )
    }
    return models

def build_ensemble_models(base_models: Dict[str, Any], random_state: int = 42) -> Dict[str, Any]:
    """
    Constructs Meta Ensembles: Soft Voting and Stacking Meta-Learner.
    """
    estimators = [
        ("rf", base_models["Random Forest"]),
        ("gb", base_models["Gradient Boosting"]),
        ("xgb", base_models["XGBoost"]),
        ("et", base_models["Extra Trees"]),
        ("lr", base_models["Logistic Regression"])
    ]

    voting_soft = VotingClassifier(
        estimators=estimators,
        voting="soft",
        weights=[2.5, 2.0, 2.5, 2.0, 1.0]
    )

    stacking_clf = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(C=0.2, random_state=random_state),
        cv=5
    )

    return {
        "Soft Voting Ensemble": voting_soft,
        "Stacking Ensemble": stacking_clf
    }

class ModelTrainer:
    """
    Orchestrates training, Stratified 10-Fold Cross Validation, model evaluation, and artifact serialization.
    """
    def __init__(self, random_state: int = 42, models_dir: str = "models"):
        self.random_state = random_state
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        self.models = {}
        self.best_model_name = None
        self.best_model = None
        self.cv_results_df = None

    def evaluate_cv(self, X: np.ndarray, y: np.ndarray, n_splits: int = 10) -> pd.DataFrame:
        """
        Executes Stratified 10-Fold CV across all candidate algorithms.
        """
        all_models = get_base_models(self.random_state)
        ensemble_models = build_ensemble_models(all_models, self.random_state)
        all_models.update(ensemble_models)
        self.models = all_models

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        scoring = {
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
            "roc_auc": "roc_auc"
        }

        results_list = []
        print(f"[ModelTrainer] Running {n_splits}-Fold Stratified Cross-Validation on {len(all_models)} algorithms...")

        for name, model in all_models.items():
            cv_out = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
            mean_acc = cv_out["test_accuracy"].mean()
            std_acc = cv_out["test_accuracy"].std()
            mean_prec = cv_out["test_precision"].mean()
            mean_rec = cv_out["test_recall"].mean()
            mean_f1 = cv_out["test_f1"].mean()
            mean_auc = cv_out["test_roc_auc"].mean()

            results_list.append({
                "Model": name,
                "CV Accuracy": mean_acc,
                "Std Accuracy": std_acc,
                "CV Precision": mean_prec,
                "CV Recall": mean_rec,
                "CV F1-Score": mean_f1,
                "CV ROC-AUC": mean_auc
            })
            print(f"  -> {name:<32} | Acc: {mean_acc*100:.2f}% (±{std_acc*100:.2f}%) | ROC-AUC: {mean_auc:.4f} | Recall: {mean_rec*100:.2f}%")

        self.cv_results_df = pd.DataFrame(results_list).sort_values(by="CV ROC-AUC", ascending=False).reset_index(drop=True)
        return self.cv_results_df

    def train_and_select_best(self, X_train: np.ndarray, y_train: np.ndarray, metric: str = "CV ROC-AUC") -> Tuple[str, Any]:
        """
        Fits all models on the entire training set and selects the best performing model.
        """
        if self.cv_results_df is None:
            self.evaluate_cv(X_train, y_train)

        self.best_model_name = self.cv_results_df.iloc[0]["Model"]
        print(f"\n[ModelTrainer] Best model selected based on {metric}: '{self.best_model_name}'")

        # Fit all models on training data
        fitted_models = {}
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            fitted_models[name] = model

        self.models = fitted_models
        self.best_model = self.models[self.best_model_name]

        return self.best_model_name, self.best_model

    def save_artifacts(self, preprocessor: Any, feature_names: list, metadata: Dict[str, Any] = None):
        """
        Persists the pipeline, preprocessor, best model, and model benchmark suite to disk.
        """
        best_model_path = os.path.join(self.models_dir, "best_diabetes_model.joblib")
        preprocessor_path = os.path.join(self.models_dir, "preprocessor.joblib")
        ensemble_path = os.path.join(self.models_dir, "voting_ensemble.joblib")
        metadata_path = os.path.join(self.models_dir, "model_metadata.joblib")

        joblib.dump(self.best_model, best_model_path)
        joblib.dump(preprocessor, preprocessor_path)
        if "Soft Voting Ensemble" in self.models:
            joblib.dump(self.models["Soft Voting Ensemble"], ensemble_path)

        meta = {
            "best_model_name": self.best_model_name,
            "feature_names": feature_names,
            "cv_summary": self.cv_results_df.to_dict(orient="records") if self.cv_results_df is not None else {},
            "custom_metadata": metadata or {}
        }
        joblib.dump(meta, metadata_path)

        print(f"[ModelTrainer] Successfully saved best model artifact to: {best_model_path}")
        print(f"[ModelTrainer] Successfully saved preprocessor artifact to: {preprocessor_path}")
