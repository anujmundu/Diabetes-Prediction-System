import os
import sys
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from src.fhir_parser import FHIRResourceParser, generate_sample_fhir_bundle
from src.phi_sanitizer import HIPAASanitizer
from src.api_service import app

client = TestClient(app)

def test_fhir_bundle_parser():
    parser = FHIRResourceParser()
    sample_bundle = generate_sample_fhir_bundle(
        patient_id="PATIENT-9921",
        glucose=175.0,
        bp=88.0,
        bmi=33.5,
        age=52,
        insulin=210.0
    )
    df = parser.fhir_to_dataframe(sample_bundle)
    assert not df.empty
    assert df["Glucose"].iloc[0] == 175.0
    assert df["BMI"].iloc[0] == 33.5
    assert df["Age"].iloc[0] == 52
    print("[Test PASSED] FHIR Bundle Parser extracted LOINC observations successfully.")

def test_hipaa_anonymizer():
    sanitizer = HIPAASanitizer()
    payload = {
        "name": "Jane Doe",
        "ssn": "000-12-3456",
        "email": "jane.doe@hospital.org",
        "patient_id": "MRN-583921",
        "glucose": 140.0
    }
    cleaned = sanitizer.sanitize_payload(payload)
    assert "name" not in cleaned
    assert "ssn" not in cleaned
    assert "email" not in cleaned
    assert cleaned["anonymized_patient_id"].startswith("ANON_")
    print("[Test PASSED] HIPAA Safe-Harbor De-Identification verified.")

def test_root_and_favicon():
    res_root = client.get("/")
    assert res_root.status_code == 200
    assert "interactive_api_docs" in res_root.json()

    res_fav = client.get("/favicon.ico")
    assert res_fav.status_code == 204
    print("[Test PASSED] Root portal (GET /) and Favicon handler verified.")

def test_api_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "active_model" in data
    print("[Test PASSED] API Health Endpoint is active and model registry loaded.")

def test_api_predict():
    payload = {
        "patient_id": "TEST-PT-101",
        "pregnancies": 2,
        "glucose": 165.0,
        "blood_pressure": 82.0,
        "skin_thickness": 30.0,
        "insulin": 180.0,
        "bmi": 32.5,
        "diabetes_pedigree_function": 0.65,
        "age": 45
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "diabetes_probability" in data
    assert "risk_tier" in data
    assert data["risk_tier"] in ["LOW", "MODERATE", "HIGH"]
    assert "HOMA_IR_Proxy" in data["derived_biomarkers"]
    print(f"[Test PASSED] Real-time inference: {data['risk_category']} (Prob: {data['diabetes_probability']*100:.1f}%)")

def test_api_fhir_ingest():
    sample_bundle = generate_sample_fhir_bundle(glucose=180.0, bmi=35.0, age=50)
    response = client.post("/api/v1/fhir/ingest", json=sample_bundle)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "diabetes_probability" in data
    print(f"[Test PASSED] FHIR Bundle Ingestion API: Risk Tier = {data['risk_tier']}")

def test_api_explain():
    payload = {
        "patient_id": "TEST-PT-102",
        "pregnancies": 3,
        "glucose": 170.0,
        "blood_pressure": 85.0,
        "skin_thickness": 32.0,
        "insulin": 190.0,
        "bmi": 34.0,
        "diabetes_pedigree_function": 0.75,
        "age": 49
    }
    response = client.post("/api/v1/explain", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "top_risk_drivers" in data
    assert "prescriptive_counterfactual_targets" in data
    print(f"[Test PASSED] XAI Engine: Generated {len(data['top_risk_drivers'])} risk drivers and {len(data['prescriptive_counterfactual_targets'])} counterfactual targets.")

if __name__ == "__main__":
    test_root_and_favicon()
    test_fhir_bundle_parser()
    test_hipaa_anonymizer()
    test_api_health()
    test_api_predict()
    test_api_fhir_ingest()
    test_api_explain()
    print("\n[SUCCESS] ALL ENTERPRISE CDSS MODULE TESTS PASSED (7/7)!")
