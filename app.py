import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="Early Diabetes Detection & Clinical Decision System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for clean clinical decision dashboard
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F3F4F6;
        border-radius: 10px;
        padding: 1.2rem;
        border-left: 5px solid #3B82F6;
        margin-bottom: 1rem;
    }
    .risk-high {
        background: #FEE2E2;
        border-left: 6px solid #EF4444;
        padding: 1.2rem;
        border-radius: 8px;
    }
    .risk-low {
        background: #ECFDF5;
        border-left: 6px solid #10B981;
        padding: 1.2rem;
        border-radius: 8px;
    }
    .risk-moderate {
        background: #FEF3C7;
        border-left: 6px solid #F59E0B;
        padding: 1.2rem;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_artifacts():
    models_dir = "models"
    model_path = os.path.join(models_dir, "best_diabetes_model.joblib")
    prep_path = os.path.join(models_dir, "preprocessor.joblib")
    meta_path = os.path.join(models_dir, "model_metadata.joblib")

    if not (os.path.exists(model_path) and os.path.exists(prep_path)):
        return None, None, None

    model = joblib.load(model_path)
    preprocessor = joblib.load(prep_path)
    metadata = joblib.load(meta_path) if os.path.exists(meta_path) else {}
    return model, preprocessor, metadata

def main():
    st.markdown('<div class="main-header">🩺 Early Diabetes Detection & Clinical Decision System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Machine Learning-Powered Risk Stratification & Clinical Guidance</div>', unsafe_allow_html=True)

    model, preprocessor, metadata = load_model_artifacts()

    if model is None or preprocessor is None:
        st.warning("⚠️ Model artifacts not found. Please run the training pipeline first by executing `python main.py` in your terminal.")
        st.info("Click the button below to trigger training.")
        if st.button("🚀 Train Models Now"):
            with st.spinner("Training models across 10-fold cross validation..."):
                from main import run_diabetes_pipeline
                run_diabetes_pipeline()
                st.success("Model trained successfully! Please refresh the page.")
        return

    best_model_name = metadata.get("best_model_name", "Trained Ensemble Classifier")
    optimal_threshold = metadata.get("optimal_threshold", 0.45)

    st.sidebar.header("📋 Patient Clinical Inputs")
    st.sidebar.caption("Provide the patient's physiological parameters below:")

    # Input Fields
    pregnancies = st.sidebar.number_input("Pregnancies (Count)", min_value=0, max_value=20, value=1, step=1, help="Number of times pregnant")
    glucose = st.sidebar.slider("Plasma Glucose (mg/dL)", min_value=40, max_value=240, value=115, help="2-hour Oral Glucose Tolerance Test concentration")
    blood_pressure = st.sidebar.slider("Diastolic Blood Pressure (mm Hg)", min_value=30, max_value=140, value=72, help="Resting diastolic blood pressure")
    skin_thickness = st.sidebar.slider("Triceps Skinfold Thickness (mm)", min_value=5, max_value=100, value=25, help="Measure of body fat composition")
    insulin = st.sidebar.slider("2-Hour Serum Insulin (μU/mL)", min_value=5, max_value=600, value=85, help="Fasting/2-hour serum insulin level")
    bmi = st.sidebar.number_input("Body Mass Index (BMI kg/m²)", min_value=10.0, max_value=65.0, value=28.5, step=0.1, help="Weight (kg) / (Height (m))^2")
    dpf = st.sidebar.number_input("Diabetes Pedigree Function", min_value=0.05, max_value=2.50, value=0.45, step=0.01, help="Genetic family history risk coefficient")
    age = st.sidebar.slider("Patient Age (Years)", min_value=18, max_value=100, value=33, help="Age in years")

    # Quick Case Presets
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚡ Quick Case Scenarios")
    col_pre1, col_pre2 = st.sidebar.columns(2)
    if col_pre1.button("Healthy Profile"):
        st.session_state.update({"glucose": 92, "bmi": 21.8, "age": 25, "insulin": 60, "blood_pressure": 70, "dpf": 0.25, "pregnancies": 0, "skin_thickness": 18})
        st.rerun()
    if col_pre2.button("High-Risk Profile"):
        st.session_state.update({"glucose": 178, "bmi": 36.4, "age": 52, "insulin": 210, "blood_pressure": 88, "dpf": 0.85, "pregnancies": 4, "skin_thickness": 38})
        st.rerun()

    # Construct input dataframe
    input_data = pd.DataFrame([{
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age
    }])

    # Transform through pipeline
    input_processed = preprocessor.transform(input_data)
    
    # Predict
    prob = float(model.predict_proba(input_processed)[0, 1]) if hasattr(model, "predict_proba") else 0.5
    prediction = int(prob >= optimal_threshold)

    # Main dashboard layout
    col_left, col_right = st.columns([1.1, 1])

    with col_left:
        st.subheader("🎯 Clinical Risk Stratification")
        
        if prob < 0.35:
            risk_category = "LOW CLINICAL RISK"
            box_class = "risk-low"
            emoji = "🟢"
            guidance = "The patient demonstrates healthy metabolic indicators. Encourage maintaining standard physical activity, balanced whole-food diet, and periodic annual checkups."
        elif prob < optimal_threshold:
            risk_category = "MODERATE / BORDERLINE RISK"
            box_class = "risk-moderate"
            emoji = "🟡"
            guidance = "Borderline elevated metabolic markers detected. Recommend screening for Glycated Hemoglobin (HbA1c), nutritional counseling, and 150 mins/week moderate exercise."
        else:
            risk_category = "HIGH CLINICAL RISK (DIABETIC PHYSIOLOGY)"
            box_class = "risk-high"
            emoji = "🔴"
            guidance = "Significant likelihood of diabetic physiology. Immediate follow-up with complete glycated hemoglobin (HbA1c) profiling, physician consultation, and glucose monitoring is strongly advised."

        st.markdown(f"""
        <div class="{box_class}">
            <h3 style="margin-top:0; margin-bottom: 0.5rem;">{emoji} Diagnosis Assessment: <b>{risk_category}</b></h3>
            <p style="font-size:1.15rem; margin-bottom:0.5rem;">Calculated Probability of Diabetes: <b>{prob * 100:.1f}%</b></p>
            <p style="margin-bottom:0; color:#374151;">{guidance}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.progress(prob, text=f"Diabetes Likelihood Score: {prob*100:.1f}% (Decision Threshold: {optimal_threshold*100:.1f}%)")

        # Computed Physiological Biomarkers
        st.subheader("🔬 Derived Metabolic Biomarkers")
        homa_ir = (glucose * insulin) / 405.0
        metabolic_score = int(glucose >= 100) + int(blood_pressure >= 80) + int(bmi >= 30)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("HOMA-IR (Insulin Resistance)", f"{homa_ir:.2f}", help="Normal: < 2.0. Insulin resistance: > 2.5")
        c2.metric("Metabolic Score", f"{metabolic_score}/3", help="Syndrome criteria met (Glucose, BP, BMI)")
        c3.metric("BMI Class", "Obese" if bmi >= 30 else ("Overweight" if bmi >= 25 else "Normal"))

    with col_right:
        st.subheader("📊 Model Diagnostic Benchmarks")
        st.info(f"🏆 Active Model: **{best_model_name}** | Decision Threshold: **{optimal_threshold:.2f}**")

        cv_summary = metadata.get("cv_summary", [])
        if cv_summary:
            cv_df = pd.DataFrame(cv_summary)
            if "CV Recall" in cv_df.columns and "CV Recall (Sensitivity)" not in cv_df.columns:
                cv_df = cv_df.rename(columns={"CV Recall": "CV Recall (Sensitivity)"})
            display_cols = [c for c in ["Model", "CV Accuracy", "CV ROC-AUC", "CV Recall (Sensitivity)"] if c in cv_df.columns]
            cv_display = cv_df[display_cols].copy()
            if "CV Accuracy" in cv_display.columns:
                cv_display["CV Accuracy"] = cv_display["CV Accuracy"].map(lambda x: f"{x*100:.1f}%" if isinstance(x, (int, float)) else str(x))
            if "CV ROC-AUC" in cv_display.columns:
                cv_display["CV ROC-AUC"] = cv_display["CV ROC-AUC"].map(lambda x: f"{x:.3f}" if isinstance(x, (int, float)) else str(x))
            if "CV Recall (Sensitivity)" in cv_display.columns:
                cv_display["CV Recall (Sensitivity)"] = cv_display["CV Recall (Sensitivity)"].map(lambda x: f"{x*100:.1f}%" if isinstance(x, (int, float)) else str(x))
            st.dataframe(cv_display.head(6), use_container_width=True, hide_index=True)

        if os.path.exists("reports/roc_curves_comparison.png"):
            with st.expander("📈 View Multi-Model ROC Curves", expanded=False):
                st.image("reports/roc_curves_comparison.png", use_container_width=True)

        if os.path.exists("reports/best_model_confusion_matrix.png"):
            with st.expander("🧩 View Test Confusion Matrix", expanded=False):
                st.image("reports/best_model_confusion_matrix.png", use_container_width=True)

        if os.path.exists("reports/feature_importance.png"):
            with st.expander("🔍 Key Diagnostic Feature Drivers", expanded=True):
                st.image("reports/feature_importance.png", use_container_width=True)

    # Clinical Guidance Section
    st.markdown("---")
    st.subheader("💡 Evidence-Based Clinical Guidance")
    rec_col1, rec_col2, rec_col3 = st.columns(3)
    with rec_col1:
        st.markdown("#### 🥗 Dietary Interventions")
        st.markdown("""
        - Prioritize low-glycemic index carbohydrates (whole grains, legumes).
        - Increase dietary fiber (> 30g/day) to slow glucose absorption.
        - Eliminate refined sugars and sweetened beverages.
        """)
    with rec_col2:
        st.markdown("#### 🏃 Physical Activity")
        st.markdown("""
        - Minimum 150 mins/week of moderate aerobic activity.
        - Incorporate resistance/strength training 2-3x/week to improve insulin sensitivity.
        - Avoid prolonged sitting (> 60 minutes uninterrupted).
        """)
    with rec_col3:
        st.markdown("#### 🩺 Clinical Diagnostics")
        st.markdown("""
        - Measure Fasting Blood Glucose (FBG) and Glycated Hemoglobin (HbA1c).
        - Lipid panel screening (Triglycerides, HDL, LDL).
        - Continuous blood pressure and weight monitoring.
        """)

if __name__ == "__main__":
    main()
