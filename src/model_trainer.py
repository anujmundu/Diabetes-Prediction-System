import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix
)

def get_tuned_models(random_state: int = 42) -> Dict[str, Any]:
    """
    Initializes regularized and tuned classifiers including modern Gradient Boosting & Tabular Neural Nets.
    """
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=0.5, random_state=random_state),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_split=4, min_samples_leaf=2, max_features="sqrt", random_state=random_state, n_jobs=1
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=300, max_depth=6, min_samples_split=4, min_samples_leaf=2, max_features="sqrt", random_state=random_state, n_jobs=1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.03, max_depth=4, subsample=0.85, random_state=random_state
        ),
        "XGBoost": XGBClassifier(
            n_estimators=150,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.85,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.2,
            random_state=random_state,
            eval_metric="logloss",
            n_jobs=1
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=150,
            learning_rate=0.03,
            num_leaves=16,
            max_depth=4,
            subsample=0.85,
            colsample_bytree=0.8,
            random_state=random_state,
            verbose=-1,
            n_jobs=1
        ),
        "Tabular Neural Net (MLP)": MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=0.01,
            learning_rate="adaptive",
            max_iter=400,
            random_state=random_state
        )
    }

    if CATBOOST_AVAILABLE:
        models["CatBoost"] = CatBoostClassifier(
            iterations=200,
            learning_rate=0.03,
            depth=5,
            l2_leaf_reg=3.0,
            verbose=0,
            random_seed=random_state,
            thread_count=1
        )

    # 1. Calibrated Soft Voting Ensemble
    voting_estimators = [
        ("gb", models["Gradient Boosting"]),
        ("xgb", models["XGBoost"]),
        ("lgbm", models["LightGBM"]),
        ("rf", models["Random Forest"])
    ]
    if CATBOOST_AVAILABLE:
        voting_estimators.append(("cat", models["CatBoost"]))

    models["Calibrated Soft Voting"] = VotingClassifier(
        estimators=voting_estimators,
        voting="soft",
        n_jobs=1
    )

    # 2. Stacked Super Learner (StackingClassifier)
    stack_estimators = [
        ("gb", models["Gradient Boosting"]),
        ("xgb", models["XGBoost"]),
        ("lgbm", models["LightGBM"]),
        ("mlp", models["Tabular Neural Net (MLP)"])
    ]
    if CATBOOST_AVAILABLE:
        stack_estimators.append(("cat", models["CatBoost"]))

    models["Stacked Super Learner"] = StackingClassifier(
        estimators=stack_estimators,
        final_estimator=LogisticRegression(C=1.0, max_iter=500),
        passthrough=True,
        cv=5,
        n_jobs=1
    )

    return models

class ModelTrainer:
    """
    10-Fold Stratified Cross-Validation Benchmarker & Optimal Decision Threshold Optimizer.
    """
    def __init__(self, random_state: int = 42, models_dir: str = "models", n_splits: int = 10):
        self.random_state = random_state
        self.models_dir = models_dir
        self.n_splits = n_splits
        self.cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        self.models = get_tuned_models(random_state)
        self.cv_results = {}
        self.best_model_name = None
        self.best_model = None
        self.optimal_threshold = 0.50

    def evaluate_cv(self, X: np.ndarray, y: np.ndarray, n_splits: int = 10) -> pd.DataFrame:
        """
        Executes Stratified K-Fold Cross Validation across all model architectures.
        """
        if n_splits != self.n_splits:
            self.n_splits = n_splits
            self.cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

        scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        summary = []

        print(f"\n[ModelTrainer] Benchmarking {len(self.models)} Tabular & Ensemble Architectures across {self.n_splits}-Fold CV...")
        print("-" * 80)

        for name, model in self.models.items():
            scores = cross_validate(model, X, y, cv=self.cv, scoring=scoring, n_jobs=1)
            mean_acc = scores["test_accuracy"].mean()
            std_acc = scores["test_accuracy"].std()
            mean_auc = scores["test_roc_auc"].mean()
            mean_rec = scores["test_recall"].mean()
            mean_f1 = scores["test_f1"].mean()

            self.cv_results[name] = {
                "CV Accuracy": mean_acc,
                "Accuracy Std": std_acc,
                "CV ROC-AUC": mean_auc,
                "CV Recall (Sensitivity)": mean_rec,
                "CV F1-Score": mean_f1
            }

            summary.append({
                "Model": name,
                "CV Accuracy": mean_acc,
                "CV ROC-AUC": mean_auc,
                "CV Recall (Sensitivity)": mean_rec,
                "CV F1-Score": mean_f1
            })

            print(f"  -> {name:<26} | Acc: {mean_acc*100:.2f}% (±{std_acc*100:.2f}%) | ROC-AUC: {mean_auc:.4f} | Recall: {mean_rec*100:.2f}%")

        summary_df = pd.DataFrame(summary).sort_values(by=["CV ROC-AUC", "CV Recall (Sensitivity)"], ascending=False).reset_index(drop=True)
        return summary_df

    def train_and_select_best(self, X: np.ndarray, y: np.ndarray) -> Tuple[str, Any, float]:
        """
        Trains all models on training data, determines the top performer, and optimizes the decision threshold.
        """
        if not self.cv_results:
            self.evaluate_cv(X, y)

        # Sort and select best
        sorted_models = sorted(self.cv_results.items(), key=lambda x: (x[1]["CV ROC-AUC"], x[1]["CV Recall (Sensitivity)"]), reverse=True)
        self.best_model_name = sorted_models[0][0]
        self.best_model = self.models[self.best_model_name]

        print(f"\n[ModelTrainer] Identified Top Production Model: '{self.best_model_name}'")

        # Threshold calibration via Out-Of-Fold probabilities using Youden's J (J = TPR - FPR)
        oof_probs = cross_val_predict(self.best_model, X, y, cv=self.cv, method="predict_proba", n_jobs=1)[:, 1]
        fpr, tpr, thresholds = roc_curve(y, oof_probs)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        self.optimal_threshold = float(thresholds[best_idx])
        self.optimal_threshold = max(0.35, min(0.65, self.optimal_threshold))

        print(f"[ModelTrainer] Computed Optimal Clinical Decision Threshold: {self.optimal_threshold:.3f} (Maximizing Sensitivity & Specificity)")

        # Fit all models on full training data
        for name, model in self.models.items():
            model.fit(X, y)

        return self.best_model_name, self.best_model, self.optimal_threshold

    def save_artifacts(self, preprocessor: Any, feature_names: list, metadata: Optional[dict] = None):
        os.makedirs(self.models_dir, exist_ok=True)
        model_path = os.path.join(self.models_dir, "best_diabetes_model.joblib")
        prep_path = os.path.join(self.models_dir, "preprocessor.joblib")
        meta_path = os.path.join(self.models_dir, "model_metadata.joblib")

        joblib.dump(self.best_model, model_path)
        joblib.dump(preprocessor, prep_path)

        cv_summary_list = []
        for name, res in self.cv_results.items():
            cv_summary_list.append({
                "Model": name,
                "CV Accuracy": res["CV Accuracy"],
                "CV ROC-AUC": res["CV ROC-AUC"],
                "CV Recall (Sensitivity)": res["CV Recall (Sensitivity)"],
                "CV F1-Score": res["CV F1-Score"]
            })

        meta = {
            "best_model_name": self.best_model_name,
            "optimal_threshold": self.optimal_threshold,
            "feature_names": feature_names,
            "cv_summary": cv_summary_list,
            "available_models": list(self.models.keys())
        }
        if metadata:
            meta.update(metadata)

        joblib.dump(meta, meta_path)
        print(f"[ModelTrainer] Serialized production model artifact to: {model_path}")
        print(f"[ModelTrainer] Serialized preprocessor artifact to: {prep_path}")
        print(f"[ModelTrainer] Serialized metadata and threshold to: {meta_path}")
