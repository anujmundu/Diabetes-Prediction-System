# 🩺 Early-Stage Diabetes Prediction & Clinical Risk Stratification System

An end-to-end Machine Learning and Data Science project designed to detect diabetes in its early stages using physiological biomarkers and patient health profiles.

---

## 📌 Project Overview
Diabetes is a chronic metabolic disease with increasing prevalence across all age cohorts. Predicting diabetes at an early or pre-diabetic stage enables proactive lifestyle interventions, targeted dietary management, and preventive clinical care.

This project implements:
1. **Multi-Strategy Data Preparation**:
   - Treatment of physiologically invalid zero values (Glucose, Blood Pressure, Skin Thickness, Insulin, BMI).
   - Advanced missing data imputation (Comparative MICE / Iterative Imputer & KNN).
   - Robust outlier clipping and scaling (RobustScaler).
   - Synthetic Minority Over-sampling (SMOTE) to prevent majority-class bias.
2. **Domain-Specific Feature Engineering**:
   - `HOMA_IR_Proxy`: Surrogate index of Insulin Resistance `(Glucose * Insulin) / 405`.
   - `Metabolic_Risk_Score`: Composite score combining hyperglycemia, hypertension, and obesity.
   - `Age_Glucose_Interaction`: Interaction term capturing age-correlated glycemic risk.
   - `BMI_Category`: Stratified body composition tiers (Underweight, Normal, Overweight, Obese).
3. **Multi-Model Benchmark & Ensembles**:
   - Logistic Regression, Random Forest, Extra Trees, Gradient Boosting, XGBoost, LightGBM.
   - Meta-Ensembles: Soft Voting Classifier and Stacking Classifier.
   - 10-Fold Stratified Cross-Validation.
4. **Clinical Decision Dashboard**:
   - Interactive Streamlit application (`app.py`) for real-time risk assessment, probability gauge, derived biomarker calculations, and actionable lifestyle recommendations.

---

## 📂 Project Structure

```
DataScience/
├── DataScience/               # Python Virtual Environment
├── data/
│   └── diabetes.csv           # Pima Indians Diabetes Dataset
├── models/
│   ├── best_diabetes_model.joblib # Serialized Top Classifier
│   ├── preprocessor.joblib    # Serialized Data Preparation Pipeline
│   └── model_metadata.joblib  # Feature names & CV scores
├── notebooks/
│   └── diabetes_analysis.ipynb # Interactive EDA & Modeling Jupyter Notebook
├── reports/
│   ├── best_model_confusion_matrix.png
│   ├── feature_importance.png
│   └── roc_curves_comparison.png
├── src/
│   ├── data_loader.py         # Automated dataset loader & mirror fetcher
│   ├── preprocessor.py        # Imputation, Feature Engineering & Scaling
│   ├── model_trainer.py       # 10-Fold CV, Benchmark Suite & Ensembles
│   └── evaluate.py            # Diagnostic metrics & plot generator
├── app.py                     # Streamlit Web Application
├── main.py                    # CLI Execution Pipeline
├── requirements.txt           # Environment dependencies
└── README.md                  # Project Documentation
```

---

## 📥 Dataset Information

### Automated Download
The project automatically checks for `data/diabetes.csv`. If not present, `src/data_loader.py` downloads it automatically from public mirrors.

### Manual Download (Optional)
If you prefer to download the dataset manually:
1. Download the **Pima Indians Diabetes Database** (or CSV file) from [Kaggle / UCI Machine Learning Repository](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database).
2. Save the file as `diabetes.csv` in the `data/` directory:
   ```
   DataScience/data/diabetes.csv
   ```
3. Columns expected:
   `Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome`

---

## 🚀 Getting Started

### 1. Activate the Virtual Environment
```powershell
# Windows PowerShell
.\DataScience\Scripts\Activate.ps1
```

### 2. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 3. Run the Training Pipeline
To run the full data preparation, 10-fold cross-validation, and generate diagnostic plots:
```powershell
python main.py
```

### 4. Launch the Interactive Web Application
```powershell
streamlit run app.py
```

---

## 📊 Model Evaluation Summary

| Model | 10-Fold CV Accuracy | 10-Fold CV ROC-AUC | Test Recall (Sensitivity) |
| :--- | :---: | :---: | :---: |
| **LightGBM / XGBoost** | **79.8% - 80.5%** | **0.880** | **79.6% - 85.5%** |
| **Random Forest** | **78.4%** | **0.876** | **84.3%** |
| **Stacking / Soft Voting** | **79.1%** | **0.877** | **84.3%** |
*All visual evaluation charts are available in `reports/`.*

---

# 👨💻 Author

## Anuj Mundu

**Master of Computer Applications (MCA)**  
Maulana Azad National Institute of Technology (MANIT), Bhopal

### Areas of Interest
- Artificial Intelligence
- Agentic AI
- Retrieval-Augmented Generation
- Large Language Models
- Machine Learning
- Full-Stack AI Engineering
- AI Systems Design

---

**GitHub:**  
[https://github.com/anujmundu](https://github.com/anujmundu)

---

**LinkedIn:**  
[https://www.linkedin.com/in/anujmundu](https://www.linkedin.com/in/anujmundu)

---
