import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Any
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from imblearn.over_sampling import SMOTE

PHYSIOLOGICAL_ZERO_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

class ClinicalFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Scikit-Learn compatible transformer engineering non-linear clinical risk biomarkers:
    - Insulin Resistance proxy (HOMA-IR)
    - Age-Glucose and Pedigree-Glucose interactions
    - Metabolic Syndrome composite score
    - Glycemic & BMI WHO clinical tiers
    - Pregnancy-Age parity risk
    """
    def __init__(self, include_interactions: bool = True):
        self.include_interactions = include_interactions

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, np.ndarray):
            cols = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
            df = pd.DataFrame(X, columns=cols[:X.shape[1]])
        else:
            df = X.copy()

        # Ensure non-negative values for mathematical stability
        for col in PHYSIOLOGICAL_ZERO_COLS:
            if col in df.columns:
                df[col] = np.maximum(0.1, df[col])

        # 1. Insulin Resistance Biomarkers
        if "Glucose" in df.columns and "Insulin" in df.columns:
            df["HOMA_IR_Proxy"] = (df["Glucose"] * df["Insulin"]) / 405.0
            df["Glucose_Insulin_Ratio"] = df["Glucose"] / (df["Insulin"] + 1.0)
            df["Log_Insulin"] = np.log1p(df["Insulin"])
            df["Log_Glucose"] = np.log1p(df["Glucose"])

        # 2. Risk Factor Interactions
        if self.include_interactions:
            if "Age" in df.columns and "Glucose" in df.columns:
                df["Age_Glucose_Risk"] = (df["Age"] * df["Glucose"]) / 100.0
            if "DiabetesPedigreeFunction" in df.columns and "Glucose" in df.columns:
                df["Pedigree_Glucose_Risk"] = df["DiabetesPedigreeFunction"] * df["Glucose"]
            if "BMI" in df.columns and "Glucose" in df.columns:
                df["BMI_Glucose_Product"] = (df["BMI"] * df["Glucose"]) / 100.0
            if "Pregnancies" in df.columns and "Age" in df.columns:
                df["Pregnancy_Age_Ratio"] = df["Pregnancies"] / (df["Age"] + 1.0)

        # 3. Clinical Composite Metabolic Score
        high_glucose = (df["Glucose"] >= 100).astype(float) if "Glucose" in df.columns else 0.0
        high_bp = (df["BloodPressure"] >= 80).astype(float) if "BloodPressure" in df.columns else 0.0
        high_bmi = (df["BMI"] >= 30).astype(float) if "BMI" in df.columns else 0.0
        df["Metabolic_Risk_Score"] = high_glucose + high_bp + high_bmi

        # 4. Clinical Tiers
        if "Glucose" in df.columns:
            df["Diabetic_Range_Glucose"] = (df["Glucose"] >= 126).astype(float)
            df["Impaired_Fasting_Glucose"] = ((df["Glucose"] >= 100) & (df["Glucose"] < 126)).astype(float)

        if "BMI" in df.columns:
            df["Obese_BMI_Class"] = (df["BMI"] >= 30).astype(float)

        return df

class EnhancedDataPipeline:
    """
    Industrial-strength Data Preparation Pipeline:
    - Missingness indicator generation
    - Iterative MICE physiological imputation
    - Clinical biomarker engineering
    - Robust scaling
    - SMOTE synthetic resampling
    """
    def __init__(self, random_state: int = 42, use_smote: bool = True):
        self.random_state = random_state
        self.use_smote = use_smote
        self.imputer = None
        self.feature_engineer = ClinicalFeatureEngineer()
        self.scaler = RobustScaler()
        self.feature_names = None

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Tuple[np.ndarray, Optional[np.ndarray], List[str]]:
        df = X.copy()
        
        # 1. Missingness indicators & physiological zero masking
        for col in PHYSIOLOGICAL_ZERO_COLS:
            if col in df.columns:
                df[f"{col}_Missing"] = (df[col] == 0).astype(float)
                df[col] = df[col].replace(0, np.nan)

        # 2. Iterative MICE Imputation
        self.imputer = IterativeImputer(max_iter=30, random_state=self.random_state)
        df_imputed = pd.DataFrame(
            self.imputer.fit_transform(df),
            columns=df.columns,
            index=df.index
        )

        # 3. Clinical Biomarker Engineering
        df_feat = self.feature_engineer.transform(df_imputed)
        self.feature_names = list(df_feat.columns)

        # 4. Robust Scaling
        X_scaled = self.scaler.fit_transform(df_feat)

        # 5. Class Resampling via SMOTE (Training only)
        y_out = y.copy() if y is not None else None
        if y is not None and self.use_smote:
            sampler = SMOTE(random_state=self.random_state)
            X_scaled, y_out = sampler.fit_resample(X_scaled, y)

        return X_scaled, (y_out.values if y_out is not None else None), self.feature_names

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        df = X.copy()
        
        for col in PHYSIOLOGICAL_ZERO_COLS:
            if col in df.columns:
                df[f"{col}_Missing"] = (df[col] == 0).astype(float)
                df[col] = df[col].replace(0, np.nan)

        # Impute
        df_imputed = pd.DataFrame(
            self.imputer.transform(df),
            columns=df.columns,
            index=df.index
        )

        # Feature Engineering
        df_feat = self.feature_engineer.transform(df_imputed)

        # Scale
        X_scaled = self.scaler.transform(df_feat)
        return X_scaled

# Backwards compatibility alias
DataPreparationPipeline = EnhancedDataPipeline
