import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.data_loader import load_or_download_dataset
from src.preprocessor import EnhancedDataPipeline
from src.model_trainer import ModelTrainer
from src.evaluate import evaluate_predictions, plot_evaluation_suite

def run_diabetes_pipeline():
    print("=" * 75)
    print("      EARLY-STAGE DIABETES PREDICTION & RISK ASSESSMENT PIPELINE      ")
    print("=" * 75)

    # 1. Load Dataset
    print("\n[Step 1/5] Loading physiological diabetes dataset...")
    df = load_or_download_dataset(data_dir="data", filename="diabetes.csv")
    print(f"Loaded dataset: {df.shape[0]} patient records, {df.shape[1]} features.")
    print("Class Balance:\n", df["Outcome"].value_counts(normalize=True).to_dict())

    # 2. Train-Test Split (Holdout 20%)
    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\n[Step 2/5] Stratified Holdout Split: Train = {len(X_train_raw)}, Test = {len(X_test_raw)}")

    # 3. Data Preparation & Feature Engineering Pipeline
    print("\n[Step 3/5] Executing Data Preparation Pipeline...")
    print("  -> Creating missingness indicators & replacing physiological zeros with NaN")
    print("  -> Iterative MICE imputation preserving cross-feature covariance")
    print("  -> Engineering Clinical Biomarkers: HOMA-IR proxy, Metabolic index, Age-Glucose risk")
    print("  -> Applying Robust Scaling & SMOTE synthetic class resampling")

    prep_pipeline = EnhancedDataPipeline(random_state=42, use_smote=True)
    X_train_proc, y_train_res, feature_names = prep_pipeline.fit_transform(X_train_raw, y_train)
    X_test_proc = prep_pipeline.transform(X_test_raw)

    print(f"Features Engineered ({len(feature_names)}): {feature_names}")
    print(f"Resampled Training Matrix: {X_train_proc.shape}")

    # 4. Multi-Model Benchmark & 10-Fold Stratified Cross-Validation
    print("\n[Step 4/5] Multi-Model Training & 10-Fold Stratified Cross-Validation...")
    trainer = ModelTrainer(random_state=42, models_dir="models")
    cv_summary = trainer.evaluate_cv(X_train_proc, y_train_res, n_splits=10)

    print("\n" + "=" * 70)
    print("                   CROSS-VALIDATION BENCHMARK SUMMARY                 ")
    print("=" * 70)
    print(cv_summary.to_string(index=False))

    best_model_name, best_model, optimal_threshold = trainer.train_and_select_best(X_train_proc, y_train_res)

    # 5. Independent Holdout Evaluation (Standard vs Optimal Threshold)
    print("\n[Step 5/5] Evaluating on Independent Test Set (20% Holdout)...")
    test_results = []
    for name, model in trainer.models.items():
        probs = model.predict_proba(X_test_proc)[:, 1] if hasattr(model, "predict_proba") else None
        
        # Standard threshold 0.50
        preds_std = (probs >= 0.50).astype(int) if probs is not None else model.predict(X_test_proc)
        metrics_std = evaluate_predictions(y_test.values, preds_std, probs)
        
        # Optimal threshold
        thresh = optimal_threshold if name == best_model_name else 0.50
        preds_opt = (probs >= thresh).astype(int) if probs is not None else preds_std
        metrics_opt = evaluate_predictions(y_test.values, preds_opt, probs)
        
        test_results.append({
            "Model": name,
            "Accuracy": metrics_std["Accuracy"],
            "Precision": metrics_std["Precision"],
            "Recall (Sensitivity)": metrics_std["Recall (Sensitivity)"],
            "Specificity": metrics_std["Specificity"],
            "F1-Score": metrics_std["F1-Score"],
            "ROC-AUC": metrics_std["ROC-AUC"],
            "Opt Thresh Acc": metrics_opt["Accuracy"],
            "Opt Thresh Recall": metrics_opt["Recall (Sensitivity)"]
        })

    test_results_df = pd.DataFrame(test_results).sort_values(by="ROC-AUC", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 75)
    print("                    TEST SET BENCHMARK RESULTS                        ")
    print("=" * 75)
    print(test_results_df.to_string(index=False))

    # Save artifacts
    trainer.save_artifacts(
        preprocessor=prep_pipeline,
        feature_names=feature_names,
        metadata={"test_summary": test_results_df.to_dict(orient="records")}
    )

    # Generate Evaluation Charts
    print("\n[Evaluate] Generating ROC curves, confusion matrices, and importance plots...")
    plots = plot_evaluation_suite(
        models_dict=trainer.models,
        X_test=X_test_proc,
        y_test=y_test.values,
        best_model_name=best_model_name,
        optimal_threshold=optimal_threshold,
        feature_names=feature_names,
        output_dir="reports"
    )

    best_metrics = test_results_df[test_results_df['Model'] == best_model_name].iloc[0]
    print("\n" + "=" * 75)
    print("                      PIPELINE EXECUTION COMPLETE!                    ")
    print(f" [*] Top Production Model: {best_model_name}")
    print(f" [*] Optimal Clinical Decision Threshold: {optimal_threshold:.3f}")
    print(f" [*] Test ROC-AUC:            {best_metrics['ROC-AUC']:.4f}")
    print(f" [*] Test Recall/Sensitivity: {best_metrics['Opt Thresh Recall']*100:.2f}% (High Detection Rate)")
    print(f" [*] Test Accuracy:           {best_metrics['Opt Thresh Acc']*100:.2f}%")
    print(f" [*] Artifacts saved in 'models/' and visual charts in 'reports/'")
    print("=" * 75)

if __name__ == "__main__":
    run_diabetes_pipeline()
