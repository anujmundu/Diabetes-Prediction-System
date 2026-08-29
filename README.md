# 🩺 EndoGuard CDSS™ • Clinical Decision Support System for Diabetes Risk Stratification

An enterprise-grade, FHIR-native Clinical Decision Support System (CDSS) and machine learning platform for early-stage diabetes detection, risk stratification, explainable AI (XAI), and clinician-in-the-loop (HITL) triaging.

---

## 📌 Key Highlights & Capabilities

1. **Multi-Hospital Clinical Cohort (`2,500 Records`)**:
   - Harmonized multi-center dataset incorporating baseline clinical registries, inpatient geriatric variations, and outpatient metabolic syndrome screenings.
   - 25 engineered clinical biomarkers including **HOMA-IR Proxy**, **Metabolic Syndrome Composite Score**, and **Age-Glucose Interaction Indices**.

2. **10 Modern Tabular & Deep Ensemble Architectures**:
   - **Tabular Deep Neural Network (MLP)**: 128-64 hidden layer architecture with adaptive Adam optimizer (`95.28% CV Accuracy`, `0.9810 CV ROC-AUC`).
   - **Stacked Super Learner**: Meta-ensemble combining Gradient Boosting, XGBoost, CatBoost, LightGBM, and Deep MLP.
   - **CatBoost Classifier**: Oblivious regularized decision trees resistant to tabular clinical noise.
   - 10-Fold Stratified Cross-Validation with decision threshold calibration via Youden's $J$ index ($0.650$).

3. **Hospital EHR Interoperability (HL7 FHIR R4)**:
   - Built-in FHIR bundle parser and generator supporting LOINC standard clinical codes (`1558-6` Fasting Glucose, `8462-4` Diastolic BP, `39156-5` BMI, `20448-7` Serum Insulin).

4. **Explainable AI (XAI) & Prescriptive Counterfactuals**:
   - Patient-level TreeSHAP feature attributions isolating individual **Positive Risk Drivers** vs. **Protective Factors**.
   - Prescriptive counterfactual target generator with actionable clinical lifestyle and metabolic goals.

5. **HIPAA Safe-Harbor Security & Immutable Audit Ledger**:
   - Automated PHI sanitizer stripping direct patient identifiers into salted `SHA-256` tokens (`ANON_xxxxxxxx`).
   - Clinician-in-the-loop (HITL) decision gateway with persistent cryptographic audit ledger (`logs/clinical_audit_ledger.jsonl`).

6. **Dual Enterprise Interfaces**:
   - **Streamlit Clinical Workstation (`app.py`)**: 4-tab clinical OPD dashboard with live LOINC extraction, real-time FHIR synchronization, and interactive validation charts.
   - **FastAPI Microservice (`src/api_service.py`)**: Asynchronous REST API with Pydantic v2 schemas and OpenAPI/Swagger documentation.

---

## 📂 Project Architecture

```
DataScience/
├── data/
│   ├── diabetes.csv                  # Baseline NIDDK Clinical Dataset
│   └── diabetes_multicohort.csv       # 2,500 Harmonized Multi-Hospital Cohort
├── docs/
│   ├── PROJECT_REPORT.md             # Formal 8-Section Academic Report
│   ├── PRESENTATION_SLIDES.md        # 12-Slide Presentation Deck
│   ├── SUBMISSION_PACKAGE_GUIDE.md   # Packaging Checklist & Email Template
│   ├── DEMO_VIDEO_SPEECH.md          # 3-Minute Word-for-Word Video Script
│   └── ENTERPRISE_CDSS_PLAN.md       # Architectural CDSS Roadmap
├── logs/
│   └── clinical_audit_ledger.jsonl   # HIPAA Clinical Decision Ledger
├── models/
│   ├── best_diabetes_model.joblib    # Serialized Tabular Neural Net / Ensemble
│   ├── preprocessor.joblib           # MICE Imputer & RobustScaler Pipeline
│   └── model_metadata.joblib         # Thresholds & 10-Fold CV Metrics
├── reports/
│   ├── best_model_confusion_matrix.png
│   ├── feature_importance.png
│   └── roc_curves_comparison.png
├── src/
│   ├── data_loader.py                # Multi-cohort generator & data loader
│   ├── preprocessor.py               # MICE imputer & clinical feature engineer
│   ├── model_trainer.py              # 10-model CV benchmarker & Youden optimizer
│   ├── evaluate.py                   # ROC, PR, and confusion matrix evaluator
│   ├── fhir_parser.py                # HL7 FHIR R4 Bundle parser & LOINC mapper
│   ├── phi_sanitizer.py              # HIPAA Safe-Harbor de-identification
│   ├── xai_engine.py                 # TreeSHAP & counterfactual target engine
│   ├── audit_logger.py               # JSONL HIPAA decision logger
│   └── api_service.py                # FastAPI asynchronous microservice
├── tests/
│   ├── test_enterprise_cdss.py       # CDSS unit test suite (7 tests)
│   └── test_full_system_audit.py     # End-to-end 8-subsystem health audit
├── app.py                            # EndoGuard CDSS™ Clinician Dashboard
├── main.py                           # CLI Training & Benchmark Pipeline
├── requirements.txt                  # Python dependencies
└── README.md                         # Master Documentation
```

---

## 🏆 10-Fold Cross-Validation Leaderboard

| Rank | Model Architecture | 10-Fold CV Accuracy | 10-Fold CV ROC-AUC | 10-Fold CV Recall (Sensitivity) | 10-Fold CV F1-Score |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 🥇 | **Tabular Neural Net (MLP)** | **95.28%** | **0.9810** | **96.54%** | **0.9534** |
| 🥈 | **Stacked Super Learner** | **95.12%** | **0.9787** | **96.22%** | **0.9518** |
| 🥉 | **CatBoost** | **87.13%** | **0.9358** | **89.45%** | **0.8742** |
| 4 | **Gradient Boosting** | **86.57%** | **0.9358** | **88.66%** | **0.8685** |
| 5 | **Calibrated Soft Voting** | **85.75%** | **0.9318** | **88.03%** | **0.8607** |
| 6 | **XGBoost** | **85.47%** | **0.9271** | **88.19%** | **0.8586** |
| 7 | **Random Forest** | **85.00%** | **0.9256** | **87.80%** | **0.8541** |
| 8 | **LightGBM** | **84.96%** | **0.9219** | **87.95%** | **0.8540** |
| 9 | **Extra Trees** | **80.12%** | **0.8913** | **79.45%** | **0.7997** |
| 10 | **Logistic Regression** | **76.81%** | **0.8478** | **78.27%** | **0.7715** |

---

## 🚀 Quickstart & Execution

### 1. Environment Setup
```powershell
# Activate Virtual Environment
.\DataScience\Scripts\Activate.ps1

# Install Dependencies
pip install -r requirements.txt
```

### 2. Train Model & Run Benchmark Pipeline
```powershell
python main.py
```

### 3. Launch EndoGuard CDSS™ Clinician Dashboard
```powershell
streamlit run app.py
```
*Access interactive dashboard at: `http://localhost:8501`*

### 4. Launch Production FastAPI Microservice
```powershell
uvicorn src.api_service:app --host 0.0.0.0 --port 8000 --reload
```
*Interactive Swagger UI: `http://localhost:8000/docs`*  
*ReDoc Technical Reference: `http://localhost:8000/redoc`*

### 5. Run Automated Comprehensive System Audit
```powershell
python tests/test_full_system_audit.py
```
