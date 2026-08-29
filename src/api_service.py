import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Request, Depends, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from src.fhir_parser import FHIRResourceParser, generate_sample_fhir_bundle
from src.phi_sanitizer import HIPAASanitizer
from src.xai_engine import ClinicalXAIEngine

# Initialize FastAPI App with Swagger / OpenAPI Docs
app = FastAPI(
    title="Clinical CDSS - Diabetes Early Detection & Risk API",
    description="Enterprise-Grade FHIR-Native Clinical Decision Support System for Diabetes Risk Stratification and Explainable AI Triaging.",
    version="2.0.0",
    contact={
        "name": "Anuj Mundu",
        "url": "https://github.com/anujmundu/Diabetes-Prediction-System"
    }
)

# Global artifacts state
MODEL = None
PREPROCESSOR = None
METADATA = None
XAI_ENGINE = None
FHIR_PARSER = FHIRResourceParser()
HIPAA_ENGINE = HIPAASanitizer()

def load_system_artifacts():
    global MODEL, PREPROCESSOR, METADATA, XAI_ENGINE
    models_dir = "models"
    model_path = os.path.join(models_dir, "best_diabetes_model.joblib")
    prep_path = os.path.join(models_dir, "preprocessor.joblib")
    meta_path = os.path.join(models_dir, "model_metadata.joblib")

    if os.path.exists(model_path) and os.path.exists(prep_path):
        MODEL = joblib.load(model_path)
        PREPROCESSOR = joblib.load(prep_path)
        METADATA = joblib.load(meta_path) if os.path.exists(meta_path) else {}
        feature_names = METADATA.get("feature_names", [])
        XAI_ENGINE = ClinicalXAIEngine(MODEL, PREPROCESSOR, feature_names)
        print("[API] Successfully loaded models, preprocessor, and XAI engine.")
    else:
        print("[API] Model artifacts not found. Please train models first using main.py.")

# Load immediately on module import
load_system_artifacts()

@app.on_event("startup")
def startup_event():
    load_system_artifacts()

# =========================================================================
# Pydantic Schemas with Rich Examples & Typing
# =========================================================================

class RootPortalResponse(BaseModel):
    system: str = Field(..., example="Enterprise Clinical Decision Support System (CDSS)")
    status: str = Field(..., example="Online & Operational")
    interactive_api_docs: str = Field(..., example="/docs")
    redoc_documentation: str = Field(..., example="/redoc")
    openapi_schema: str = Field(..., example="/openapi.json")
    health_check: str = Field(..., example="/health")
    endpoints: Dict[str, str] = Field(..., example={
        "predict": "/api/v1/predict (POST)",
        "fhir_ingest": "/api/v1/fhir/ingest (POST)",
        "explain_xai": "/api/v1/explain (POST)"
    })
    description: str = Field(..., example="Navigate to /docs in your browser to interact with the API endpoints via Swagger UI.")

class HealthCheckResponse(BaseModel):
    status: str = Field(..., example="healthy")
    service: str = Field(..., example="Clinical Diabetes CDSS API")
    active_model: str = Field(..., example="Gradient Boosting")
    optimal_threshold: float = Field(..., example=0.502)
    features_count: int = Field(..., example=19)
    hipaa_compliance: str = Field(..., example="Safe-Harbor SHA-256 Tokenization Enabled")

class PatientClinicalPayload(BaseModel):
    patient_id: Optional[str] = Field("PATIENT-001", description="Patient MRN or identifier (will be de-identified)")
    pregnancies: int = Field(1, ge=0, le=25, description="Number of pregnancies")
    glucose: float = Field(120.0, ge=30.0, le=500.0, description="Plasma Glucose (mg/dL)")
    blood_pressure: float = Field(75.0, ge=30.0, le=250.0, description="Diastolic Blood Pressure (mm Hg)")
    skin_thickness: float = Field(25.0, ge=0.0, le=120.0, description="Triceps skinfold thickness (mm)")
    insulin: float = Field(85.0, ge=0.0, le=1000.0, description="2-Hour serum insulin (uIU/mL)")
    bmi: float = Field(28.5, ge=10.0, le=80.0, description="Body Mass Index (kg/m2)")
    diabetes_pedigree_function: float = Field(0.45, ge=0.01, le=3.0, description="Genetic diabetes pedigree score")
    age: int = Field(35, ge=18, le=120, description="Patient age in years")

    model_config = {
        "json_schema_extra": {
            "example": {
                "patient_id": "MRN-849201",
                "pregnancies": 2,
                "glucose": 145.0,
                "blood_pressure": 82.0,
                "skin_thickness": 28.0,
                "insulin": 160.0,
                "bmi": 31.4,
                "diabetes_pedigree_function": 0.58,
                "age": 42
            }
        }
    }

class DerivedBiomarkers(BaseModel):
    HOMA_IR_Proxy: float = Field(..., example=5.73, description="Insulin resistance surrogate calculation")
    Metabolic_Risk_Score: str = Field(..., example="2/3", description="Criteria met for Metabolic Syndrome")
    BMI_Class: str = Field(..., example="Obese", description="WHO BMI classification category")

class PredictionResponse(BaseModel):
    anonymized_patient_id: str = Field(..., example="ANON_7f9c2d1b8e3a4f50")
    risk_tier: str = Field(..., example="HIGH")
    risk_category: str = Field(..., example="High Clinical Risk (Diabetic Physiology)")
    diabetes_probability: float = Field(..., example=0.8742)
    decision_threshold: float = Field(..., example=0.502)
    confidence_score: float = Field(..., example=0.744)
    derived_biomarkers: DerivedBiomarkers
    clinical_triaging_action: str = Field(..., example="Urgent physician follow-up and confirmatory Glycated Hemoglobin (HbA1c) profiling advised.")

class FHIRIngestResponse(BaseModel):
    status: str = Field(..., example="success")
    fhir_bundle_parsed: bool = Field(..., example=True)
    anonymized_patient_id: str = Field(..., example="ANON_3a8f1b9c2d4e5f60")
    extracted_observations: Dict[str, float] = Field(..., example={
        "Glucose": 168.0,
        "BloodPressure": 86.0,
        "BMI": 34.2,
        "Insulin": 195.0,
        "Age": 48.0
    })
    diabetes_probability: float = Field(..., example=0.9124)
    risk_tier: str = Field(..., example="HIGH")
    clinical_decision_support: str = Field(..., example="Urgent HbA1c Lab Required")

class FeatureContributionItem(BaseModel):
    feature: str = Field(..., example="Glucose")
    normalized_value: float = Field(..., example=1.42)
    impact_percent: float = Field(..., example=34.5)
    direction: str = Field(..., example="Risk Driver (+)")

class CounterfactualTargetItem(BaseModel):
    parameter: str = Field(..., example="Fasting Plasma Glucose")
    current_value: str = Field(..., example="145 mg/dL")
    target_value: str = Field(..., example="105 mg/dL")
    expected_risk_reduction: str = Field(..., example="-32.4%")
    clinical_rationale: str = Field(..., example="Adopting low-glycemic dietary interventions and carbohydrate restriction.")

class XAIExplanationResponse(BaseModel):
    anonymized_patient_id: str = Field(..., example="ANON_7f9c2d1b8e3a4f50")
    predicted_probability: float = Field(..., example=0.8742)
    top_risk_drivers: List[FeatureContributionItem]
    top_protective_factors: List[FeatureContributionItem]
    prescriptive_counterfactual_targets: List[CounterfactualTargetItem]

# =========================================================================
# API Endpoints
# =========================================================================

@app.get("/", response_model=RootPortalResponse, tags=["Root"])
def root_portal():
    """
    Root portal providing system documentation links and active status.
    """
    return RootPortalResponse(
        system="Enterprise Clinical Decision Support System (CDSS)",
        status="Online & Operational",
        interactive_api_docs="/docs",
        redoc_documentation="/redoc",
        openapi_schema="/openapi.json",
        health_check="/health",
        endpoints={
            "predict": "/api/v1/predict (POST)",
            "fhir_ingest": "/api/v1/fhir/ingest (POST)",
            "explain_xai": "/api/v1/explain (POST)"
        },
        description="Navigate to /docs in your browser to interact with the API endpoints via Swagger UI."
    )

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    """
    Suppresses 404 logs for browser favicon requests.
    """
    return Response(status_code=204)

@app.get("/health", response_model=HealthCheckResponse, tags=["System Health"])
def health_check():
    """
    Health check and active model registry metadata.
    """
    if MODEL is None or PREPROCESSOR is None:
        raise HTTPException(status_code=503, detail="Models not initialized. Train models first.")
    
    return HealthCheckResponse(
        status="healthy",
        service="Clinical Diabetes CDSS API",
        active_model=str(METADATA.get("best_model_name", "Gradient Boosting")),
        optimal_threshold=float(METADATA.get("optimal_threshold", 0.502)),
        features_count=len(METADATA.get("feature_names", [])),
        hipaa_compliance="Safe-Harbor SHA-256 Tokenization Enabled"
    )

@app.post("/api/v1/predict", response_model=PredictionResponse, tags=["Clinical Inference"])
def predict_patient_risk(payload: PatientClinicalPayload):
    """
    Real-time clinical inference and risk stratification for individual patient.
    """
    if MODEL is None or PREPROCESSOR is None:
        raise HTTPException(status_code=503, detail="Model pipeline unavailable.")

    # 1. HIPAA De-identification
    sanitized = HIPAA_ENGINE.sanitize_payload(payload.model_dump())
    anon_id = sanitized["anonymized_patient_id"]

    # 2. Format to DataFrame
    input_df = pd.DataFrame([{
        "Pregnancies": payload.pregnancies,
        "Glucose": payload.glucose,
        "BloodPressure": payload.blood_pressure,
        "SkinThickness": payload.skin_thickness,
        "Insulin": payload.insulin,
        "BMI": payload.bmi,
        "DiabetesPedigreeFunction": payload.diabetes_pedigree_function,
        "Age": payload.age
    }])

    # 3. Predict Probability
    X_proc = PREPROCESSOR.transform(input_df)
    prob = float(MODEL.predict_proba(X_proc)[0, 1])
    thresh = float(METADATA.get("optimal_threshold", 0.502))

    # 4. Risk Stratification
    if prob < 0.35:
        tier = "LOW"
        category = "Low Clinical Risk"
        action = "Maintain routine wellness and annual checkups."
    elif prob < thresh:
        tier = "MODERATE"
        category = "Borderline / Pre-Diabetic Risk"
        action = "Recommend lifestyle modification and non-urgent HbA1c screening."
    else:
        tier = "HIGH"
        category = "High Clinical Risk (Diabetic Physiology)"
        action = "Urgent physician follow-up and confirmatory Glycated Hemoglobin (HbA1c) profiling advised."

    # 5. Derived Biomarkers
    homa_ir = (payload.glucose * (payload.insulin if payload.insulin > 0 else 30.0)) / 405.0
    metabolic_score = int(payload.glucose >= 100) + int(payload.blood_pressure >= 80) + int(payload.bmi >= 30)

    return PredictionResponse(
        anonymized_patient_id=anon_id,
        risk_tier=tier,
        risk_category=category,
        diabetes_probability=round(prob, 4),
        decision_threshold=round(thresh, 3),
        confidence_score=round(abs(prob - thresh) * 2, 3),
        derived_biomarkers=DerivedBiomarkers(
            HOMA_IR_Proxy=round(homa_ir, 2),
            Metabolic_Risk_Score=f"{metabolic_score}/3",
            BMI_Class="Obese" if payload.bmi >= 30 else ("Overweight" if payload.bmi >= 25 else "Normal")
        ),
        clinical_triaging_action=action
    )

@app.post("/api/v1/fhir/ingest", response_model=FHIRIngestResponse, tags=["Interoperability (HL7 FHIR)"])
def ingest_fhir_bundle(bundle: Dict[str, Any]):
    """
    Ingests and parses HL7 FHIR JSON Bundle from hospital EHR (Epic / Cerner) and executes CDSS triaging.
    """
    if MODEL is None or PREPROCESSOR is None:
        raise HTTPException(status_code=503, detail="Model pipeline unavailable.")

    try:
        input_df = FHIR_PARSER.fhir_to_dataframe(bundle)
        parsed_data = FHIR_PARSER.parse_bundle(bundle)
        anon_id = HIPAA_ENGINE.anonymize_patient_id(str(parsed_data.get("PatientID", "UNKNOWN")))

        X_proc = PREPROCESSOR.transform(input_df)
        prob = float(MODEL.predict_proba(X_proc)[0, 1])
        thresh = float(METADATA.get("optimal_threshold", 0.502))

        extracted_obs = {}
        for k, v in parsed_data.items():
            if isinstance(v, (int, float)) and not np.isnan(v):
                extracted_obs[k] = float(v)

        return FHIRIngestResponse(
            status="success",
            fhir_bundle_parsed=True,
            anonymized_patient_id=anon_id,
            extracted_observations=extracted_obs,
            diabetes_probability=round(prob, 4),
            risk_tier="HIGH" if prob >= thresh else ("MODERATE" if prob >= 0.35 else "LOW"),
            clinical_decision_support="Urgent HbA1c Lab Required" if prob >= thresh else "Routine Care"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse FHIR bundle: {str(e)}")

@app.post("/api/v1/explain", response_model=XAIExplanationResponse, tags=["Explainable AI (XAI)"])
def explain_prediction(payload: PatientClinicalPayload):
    """
    Computes patient-level feature attributions (SHAP style) and counterfactual prescriptive targets.
    """
    if XAI_ENGINE is None:
        raise HTTPException(status_code=503, detail="XAI Engine unavailable.")

    input_df = pd.DataFrame([{
        "Pregnancies": payload.pregnancies,
        "Glucose": payload.glucose,
        "BloodPressure": payload.blood_pressure,
        "SkinThickness": payload.skin_thickness,
        "Insulin": payload.insulin,
        "BMI": payload.bmi,
        "DiabetesPedigreeFunction": payload.diabetes_pedigree_function,
        "Age": payload.age
    }])

    contributions = XAI_ENGINE.get_feature_contributions(input_df)
    counterfactuals = XAI_ENGINE.generate_counterfactual_targets(input_df)

    return XAIExplanationResponse(
        anonymized_patient_id=HIPAA_ENGINE.anonymize_patient_id(payload.patient_id or "PATIENT-001"),
        predicted_probability=round(contributions["predicted_probability"], 4),
        top_risk_drivers=[FeatureContributionItem(**item) for item in contributions["top_risk_drivers"]],
        top_protective_factors=[FeatureContributionItem(**item) for item in contributions["top_protective_factors"]],
        prescriptive_counterfactual_targets=[CounterfactualTargetItem(**rec) for rec in counterfactuals]
    )

if __name__ == "__main__":
    uvicorn.run("src.api_service:app", host="0.0.0.0", port=8000, reload=True)
