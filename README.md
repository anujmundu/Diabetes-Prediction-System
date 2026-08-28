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

## 📊 Model Evaluation Summary (10-Fold Stratified CV & Test Holdout)

| Model | 10-Fold CV Accuracy | 10-Fold CV ROC-AUC | 10-Fold CV Recall | Test Holdout ROC-AUC | Test Holdout Recall |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Gradient Boosting** | **80.50%** | **0.8844** | **83.25%** | **0.8141** | **72.22%** |
| **Calibrated Soft Voting** | **79.12%** | **0.8793** | **82.00%** | **0.8170** | **75.93%** |
| **XGBoost** | **79.38%** | **0.8786** | **82.00%** | **0.8128** | **75.93%** |
| **LightGBM** | **79.38%** | **0.8761** | **82.25%** | **0.8094** | **79.63%** |
| **Random Forest** | **78.25%** | **0.8757** | **81.50%** | **0.8185** | **72.22%** |
| **Extra Trees** | **78.12%** | **0.8621** | **78.25%** | **0.8157** | **74.07%** |
| **Logistic Regression** | **75.75%** | **0.8564** | **75.25%** | **0.8169** | **81.48%** |

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
