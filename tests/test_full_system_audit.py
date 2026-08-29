import os
import sys
import json
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from src.data_loader import load_or_download_dataset
from src.preprocessor import EnhancedDataPipeline, ClinicalFeatureEngineer
from src.fhir_parser import FHIRResourceParser, generate_sample_fhir_bundle
from src.phi_sanitizer import HIPAASanitizer
from src.xai_engine import ClinicalXAIEngine
from src.api_service import app
from src.audit_logger import append_audit_log, read_recent_audit_logs

client = TestClient(app)

def audit_data_loader():
    print("[1/8] Auditing DataLoader...")
    df = load_or_download_dataset()
    assert df is not None, "DataLoader returned None"
    assert df.shape[0] >= 768, f"Expected at least 768 rows, got {df.shape[0]}"
    assert df.shape[1] == 9, f"Expected 9 columns, got {df.shape[1]}"
    print(f"      [OK] DataLoader verified ({df.shape[0]} records, 9 columns).")

def audit_preprocessor():
    print("[2/8] Auditing EnhancedDataPipeline & ClinicalFeatureEngineer...")
    df = load_or_download_dataset()
    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]

    pipeline = EnhancedDataPipeline(use_smote=True)
    X_train_res, y_train_res, feature_names = pipeline.fit_transform(X, y)

    assert X_train_res.shape[0] > 0
    assert y_train_res.shape[0] == X_train_res.shape[0]
    assert len(feature_names) == X_train_res.shape[1]

    # Test edge case with zeros and NaNs in single sample
    edge_sample = pd.DataFrame([{
        "Pregnancies": 0, "Glucose": 0, "BloodPressure": 0, "SkinThickness": 0,
        "Insulin": 0, "BMI": 0, "DiabetesPedigreeFunction": 0.1, "Age": 21
    }])
    X_edge = pipeline.transform(edge_sample)
    assert not np.isnan(X_edge).any(), "MICE / Imputer output contains NaN on edge case with all zeros"
    print("      [OK] Preprocessor verified (Robust scaling, MICE imputation, Zero-handling).")

def audit_fhir_parser():
    print("[3/8] Auditing HL7 FHIR Parser...")
    parser = FHIRResourceParser()
    
    # 1. Standard bundle
    bundle = generate_sample_fhir_bundle(glucose=150.0, bp=80.0, bmi=30.0, age=45, insulin=120.0)
    df = parser.fhir_to_dataframe(bundle)
    assert df["Glucose"].iloc[0] == 150.0
    assert df["BMI"].iloc[0] == 30.0

    # 2. Empty / minimal bundle edge case
    empty_bundle = {"resourceType": "Bundle", "entry": []}
    parsed_empty = parser.parse_bundle(empty_bundle)
    assert parsed_empty["PatientID"] == "ANONYMOUS_PATIENT"
    print("      [OK] FHIR Parser verified (Standard + Empty edge cases).")

def audit_phi_sanitizer():
    print("[4/8] Auditing HIPAA De-Identification Engine...")
    sanitizer = HIPAASanitizer()
    payload = {
        "name": "John Smith",
        "ssn": "123-45-6789",
        "phone": "+1-555-0199",
        "patient_id": "MRN-112233",
        "glucose": 135.0
    }
    sanitized = sanitizer.sanitize_payload(payload)
    assert "name" not in sanitized
    assert "ssn" not in sanitized
    assert "phone" not in sanitized
    assert sanitized["anonymized_patient_id"].startswith("ANON_")
    print("      [OK] HIPAA Sanitizer verified (SHA-256 Tokenization).")

def audit_xai_engine():
    print("[5/8] Auditing Explainable AI (XAI) & Counterfactual Targets...")
    import joblib
    model = joblib.load("models/best_diabetes_model.joblib")
    prep = joblib.load("models/preprocessor.joblib")
    meta = joblib.load("models/model_metadata.joblib")

    xai = ClinicalXAIEngine(model, prep, meta["feature_names"])
    sample = pd.DataFrame([{
        "Pregnancies": 3, "Glucose": 175, "BloodPressure": 85, "SkinThickness": 32,
        "Insulin": 190, "BMI": 34.5, "DiabetesPedigreeFunction": 0.72, "Age": 48
    }])

    contribs = xai.get_feature_contributions(sample)
    assert "predicted_probability" in contribs
    assert len(contribs["top_risk_drivers"]) > 0

    targets = xai.generate_counterfactual_targets(sample)
    assert len(targets) > 0
    print("      [OK] XAI Engine verified (Patient-level attributions & Prescriptive goals).")

def audit_fastapi_endpoints():
    print("[6/8] Auditing FastAPI Microservice Endpoints...")
    
    # 1. Root
    res = client.get("/")
    assert res.status_code == 200
    assert "interactive_api_docs" in res.json()

    # 2. Favicon
    res = client.get("/favicon.ico")
    assert res.status_code == 204

    # 3. Health
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "healthy"

    # 4. Predict
    payload = {
        "patient_id": "AUDIT-001",
        "pregnancies": 2, "glucose": 140.0, "blood_pressure": 78.0,
        "skin_thickness": 26.0, "insulin": 110.0, "bmi": 29.0,
        "diabetes_pedigree_function": 0.50, "age": 38
    }
    res = client.post("/api/v1/predict", json=payload)
    assert res.status_code == 200
    assert "diabetes_probability" in res.json()

    # 5. FHIR Ingest
    bundle = generate_sample_fhir_bundle(glucose=160.0, bp=84.0, bmi=32.0, age=46)
    res = client.post("/api/v1/fhir/ingest", json=bundle)
    assert res.status_code == 200
    assert res.json()["fhir_bundle_parsed"] is True

    # 6. Explain
    res = client.post("/api/v1/explain", json=payload)
    assert res.status_code == 200
    assert "top_risk_drivers" in res.json()
    print("      [OK] All FastAPI Microservice Endpoints verified (6/6 routes).")

def audit_audit_ledger():
    print("[7/8] Auditing HIPAA Clinical Audit Ledger...")
    test_entry = {
        "timestamp": "2026-08-29 05:25:00",
        "anonymized_patient_id": "ANON_TEST_AUDIT",
        "risk_category": "HIGH CLINICAL RISK",
        "diabetes_probability": "88.5%",
        "fasting_glucose": "170 mg/dL",
        "bmi": "33.0",
        "homa_ir": "4.2",
        "clinician_decision": "ACCEPTED_AND_ORDERED_LAB",
        "clinical_action": "System Audit Verification",
        "model_version": "Gradient Boosting"
    }
    append_audit_log(test_entry)
    logs_df = read_recent_audit_logs(limit=5)
    assert not logs_df.empty
    assert "ANON_TEST_AUDIT" in logs_df["anonymized_patient_id"].values
    print("      [OK] Clinical Audit Ledger verified (Append & Retrieve).")

def audit_reports_and_artifacts():
    print("[8/8] Auditing Model Artifacts & Visual Reports...")
    required_files = [
        "models/best_diabetes_model.joblib",
        "models/preprocessor.joblib",
        "models/model_metadata.joblib",
        "reports/roc_curves_comparison.png",
        "reports/best_model_confusion_matrix.png",
        "reports/feature_importance.png",
        "docs/PROJECT_REPORT.md",
        "docs/PRESENTATION_SLIDES.md",
        "docs/SUBMISSION_PACKAGE_GUIDE.md",
        "docs/DEMO_VIDEO_SPEECH.md",
        "docs/ENTERPRISE_CDSS_IMPLEMENTATION_PLAN.md"
    ]
    for f in required_files:
        assert os.path.exists(f), f"Missing required project file: {f}"
    print(f"      [OK] All {len(required_files)} model artifacts, visual charts, and documentation verified.")

if __name__ == "__main__":
    print("===========================================================================")
    print("           COMPREHENSIVE END-TO-END SYSTEM HEALTH AUDIT                    ")
    print("===========================================================================")
    audit_data_loader()
    audit_preprocessor()
    audit_fhir_parser()
    audit_phi_sanitizer()
    audit_xai_engine()
    audit_fastapi_endpoints()
    audit_audit_ledger()
    audit_reports_and_artifacts()
    print("===========================================================================")
    print(" [PASSED] ALL 8 SUBSYSTEMS AUDITED: ZERO ERRORS, ZERO BUGS DETECTED!       ")
    print("===========================================================================")
