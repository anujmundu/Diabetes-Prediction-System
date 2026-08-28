import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    brier_score_loss
)

def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> Dict[str, float]:
    """
    Computes a comprehensive dictionary of binary classification metrics.
    """
    metrics = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall (Sensitivity)": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1-Score": float(f1_score(y_true, y_pred, zero_division=0))
    }

    # Specificity calculation: TN / (TN + FP)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics["Specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    if y_prob is not None:
        metrics["ROC-AUC"] = float(roc_auc_score(y_true, y_prob))
        metrics["Brier Score"] = float(brier_score_loss(y_true, y_prob))

    return metrics

def plot_evaluation_suite(
    models_dict: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    best_model_name: str,
    optimal_threshold: float = 0.5,
    feature_names: list = None,
    output_dir: str = "reports"
) -> Dict[str, str]:
    """
    Generates high-resolution visualization plots:
    1. ROC Curves comparison for all models
    2. Confusion Matrix of the Best Model (at optimal threshold)
    3. Feature Importance rankings
    """
    os.makedirs(output_dir, exist_ok=True)
    generated_plots = {}

    sns.set_theme(style="whitegrid", font_scale=1.1)

    # 1. ROC Curves Comparison
    fig, ax = plt.subplots(figsize=(10, 7))
    for name, model in models_dict.items():
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, probs)
            auc_val = roc_auc_score(y_test, probs)
            ax.plot(fpr, tpr, lw=2.2, label=f"{name} (AUC = {auc_val:.3f})")
    
    ax.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--", label="Random Baseline (AUC = 0.50)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Recall / Sensitivity)")
    ax.set_title("Receiver Operating Characteristic (ROC) Comparison", fontsize=14, fontweight="bold", pad=15)
    ax.legend(loc="lower right", fontsize=9.5)
    plt.tight_layout()
    roc_path = os.path.join(output_dir, "roc_curves_comparison.png")
    fig.savefig(roc_path, dpi=300)
    plt.close(fig)
    generated_plots["roc_curves"] = roc_path

    # 2. Confusion Matrix of Best Model at Optimal Decision Threshold
    best_model = models_dict[best_model_name]
    probs_best = best_model.predict_proba(X_test)[:, 1]
    y_pred_best = (probs_best >= optimal_threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred_best)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Non-Diabetic (0)", "Diabetic (1)"],
        yticklabels=["Non-Diabetic (0)", "Diabetic (1)"],
        ax=ax,
        annot_kws={"size": 15, "weight": "bold"}
    )
    ax.set_xlabel("Predicted Diagnosis", fontweight="bold", labelpad=10)
    ax.set_ylabel("True Clinical Label", fontweight="bold", labelpad=10)
    ax.set_title(f"Confusion Matrix: {best_model_name}\n(Threshold = {optimal_threshold:.2f})", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "best_model_confusion_matrix.png")
    fig.savefig(cm_path, dpi=300)
    plt.close(fig)
    generated_plots["confusion_matrix"] = cm_path

    # 3. Feature Importance Plot
    if feature_names is not None:
        importances = None
        if hasattr(best_model, "feature_importances_"):
            importances = best_model.feature_importances_
        elif hasattr(best_model, "named_estimators_") and "xgb" in best_model.named_estimators_:
            importances = best_model.named_estimators_["xgb"].feature_importances_
        elif hasattr(best_model, "named_estimators_") and "rf" in best_model.named_estimators_:
            importances = best_model.named_estimators_["rf"].feature_importances_

        if importances is not None and len(importances) == len(feature_names):
            feat_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": importances
            }).sort_values(by="Importance", ascending=True)

            fig, ax = plt.subplots(figsize=(10, 7))
            sns.barplot(data=feat_df, x="Importance", y="Feature", hue="Feature", legend=False, palette="crest", ax=ax)
            ax.set_title(f"Key Diagnostic Biomarkers ({best_model_name})", fontsize=13, fontweight="bold", pad=15)
            ax.set_xlabel("Relative Feature Importance Score", fontweight="bold")
            plt.tight_layout()
            feat_path = os.path.join(output_dir, "feature_importance.png")
            fig.savefig(feat_path, dpi=300)
            plt.close(fig)
            generated_plots["feature_importance"] = feat_path

    print(f"[Evaluate] Generated visual evaluation artifacts in '{output_dir}/'")
    return generated_plots
