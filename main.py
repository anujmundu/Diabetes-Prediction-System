import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.data_loader import load_or_download_dataset
from src.preprocessor import DataPreparationPipeline
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
    print("Distribution of Outcomes:\n", df["Outcome"].value_counts(normalize=True).to_dict())

    # 2. Train-Test Split (Holdout 20% for unbiased final evaluation)
    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\n[Step 2/5] Stratified Split: Train set = {len(X_train_raw)}, Test set = {len(X_test_raw)}")

    # 3. Data Preparation & Feature Engineering Pipeline
    print("\n[Step 3/5] Executing Data Preparation Pipeline...")
    print("  -> Replacing physiological 0s with NaN for Glucose, BP, Insulin, BMI, SkinThickness")
    print("  -> Imputing missing values using Iterative MICE Imputer")
    print("  -> Engineering Biomarkers: HOMA-IR proxy, Age-Glucose index, Metabolic syndrome score")
    print("  -> Applying Robust Scaling & SMOTE Synthetic Class Resampling on training set")

    prep_pipeline = DataPreparationPipeline(
        imputer_strategy="iterative",
        scaler_strategy="robust",
        handle_outliers=True,
        feature_engineering=True,
        resampler_strategy="smote",
        random_state=42
    )

    X_train_proc, y_train_res, feature_names = prep_pipeline.fit_transform(X_train_raw, y_train)
    X_test_proc = prep_pipeline.transform(X_test_raw)

    print(f"Features created ({len(feature_names)}): {feature_names}")
    print(f"Resampled Training Shape: {X_train_proc.shape}, Target distribution: {pd.Series(y_train_res).value_counts().to_dict()}")

    # 4. Multi-Model Benchmark & Cross Validation
    print("\n[Step 4/5] Multi-Model Training & 10-Fold Stratified Cross-Validation...")
    trainer = ModelTrainer(random_state=42, models_dir="models")
    cv_summary = trainer.evaluate_cv(X_train_proc, y_train_res, n_splits=10)

    print("\n" + "=" * 70)
    print("                   CROSS-VALIDATION BENCHMARK SUMMARY                 ")
    print("=" * 70)
    print(cv_summary.to_string(index=False))

    best_model_name, best_model = trainer.train_and_select_best(X_train_proc, y_train_res)

    # 5. Independent Test Evaluation & Artifact Generation
    print("\n[Step 5/5] Evaluating on Independent Test Set (20% Holdout)...")
    test_results = []
    for name, model in trainer.models.items():
        y_pred = model.predict(X_test_proc)
        y_prob = model.predict_proba(X_test_proc)[:, 1] if hasattr(model, "predict_proba") else None
        metrics = evaluate_predictions(y_test.values, y_pred, y_prob)
        metrics["Model"] = name
        test_results.append(metrics)

    test_results_df = pd.DataFrame(test_results).sort_values(by="ROC-AUC", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 70)
    print("                    TEST SET BENCHMARK RESULTS                        ")
    print("=" * 70)
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
        feature_names=feature_names,
        output_dir="reports"
    )

    print("\n" + "=" * 75)
    print("                      PIPELINE EXECUTION COMPLETE!                    ")
    print(f" Best Performing Model: {best_model_name}")
    best_test_metrics = test_results_df[test_results_df['Model'] == best_model_name].iloc[0]
    print(f" Test Accuracy: {best_test_metrics['Accuracy']*100:.2f}%")
    print(f" Test ROC-AUC:  {best_test_metrics['ROC-AUC']:.4f}")
    print(f" Test Recall (Sensitivity): {best_test_metrics['Recall (Sensitivity)']*100:.2f}%")
    print(f" Test Specificity: {best_test_metrics['Specificity']*100:.2f}%")
    print(f" Artifacts saved in 'models/' and visual charts in 'reports/'")
    print("=" * 75)

if __name__ == "__main__":
    run_diabetes_pipeline()
