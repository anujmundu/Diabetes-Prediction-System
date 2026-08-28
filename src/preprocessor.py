import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from imblearn.over_sampling import SMOTE, ADASYN

PHYSIOLOGICAL_ZERO_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

class DiabetesFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom Scikit-Learn transformer to engineer clinically meaningful risk indicators:
    - Insulin Resistance proxy (HOMA-IR approximation)
    - BMI Category indices
    - Age-Glucose Risk Index
    - Metabolic Syndrome composite score
    - Pregnancy-Age Ratio
    """
    def __init__(self, include_homa_ir: bool = True, include_metabolic: bool = True):
        self.include_homa_ir = include_homa_ir
        self.include_metabolic = include_metabolic

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, np.ndarray):
            # If passed as numpy array, convert to DataFrame with standard column names
            cols = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
            df = pd.DataFrame(X, columns=cols[:X.shape[1]])
        else:
            df = X.copy()

        # 1. HOMA-IR proxy: (Glucose (mg/dL) * Insulin (uIU/mL)) / 405
        if self.include_homa_ir and "Glucose" in df.columns and "Insulin" in df.columns:
            df["HOMA_IR_Proxy"] = (df["Glucose"] * df["Insulin"]) / 405.0
            # Safe ratio to avoid division by zero
            df["Glucose_Insulin_Ratio"] = df["Glucose"] / (df["Insulin"] + 1e-5)

        # 2. Age-Glucose Interaction Risk Index
        if "Age" in df.columns and "Glucose" in df.columns:
            df["Age_Glucose_Interaction"] = (df["Age"] * df["Glucose"]) / 100.0

        # 3. Pregnancy to Age Ratio
        if "Pregnancies" in df.columns and "Age" in df.columns:
            df["Pregnancy_Age_Ratio"] = df["Pregnancies"] / (df["Age"] + 1.0)

        # 4. Metabolic Syndrome indicators
        if self.include_metabolic:
            high_glucose = (df["Glucose"] >= 100).astype(int) if "Glucose" in df.columns else 0
            high_bp = (df["BloodPressure"] >= 80).astype(int) if "BloodPressure" in df.columns else 0
            high_bmi = (df["BMI"] >= 30).astype(int) if "BMI" in df.columns else 0
            df["Metabolic_Risk_Score"] = high_glucose + high_bp + high_bmi

        # 5. BMI Categories (0: Underweight, 1: Normal, 2: Overweight, 3: Obese)
        if "BMI" in df.columns:
            df["BMI_Category"] = pd.cut(
                df["BMI"],
                bins=[-np.inf, 18.5, 24.9, 29.9, np.inf],
                labels=[0, 1, 2, 3]
            ).astype(float)

        return df

def clean_physiological_zeros(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replaces 0 with NaN in columns where 0 is physiologically impossible or invalid.
    """
    df_clean = df.copy()
    for col in PHYSIOLOGICAL_ZERO_COLS:
        if col in df_clean.columns:
            zeros_count = (df_clean[col] == 0).sum()
            if zeros_count > 0:
                df_clean[col] = df_clean[col].replace(0, np.nan)
    return df_clean

def cap_outliers_iqr(df: pd.DataFrame, factor: float = 1.5) -> pd.DataFrame:
    """
    Performs IQR-based Winsorization/Capping on numerical columns.
    """
    df_capped = df.copy()
    num_cols = df_capped.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if col == "Outcome":
            continue
        q25 = df_capped[col].quantile(0.25)
        q75 = df_capped[col].quantile(0.75)
        iqr = q75 - q25
        lower_bound = q25 - (factor * iqr)
        upper_bound = q75 + (factor * iqr)
        # Only cap positive physiological metrics at lower bound of 0
        lower_bound = max(0, lower_bound)
        df_capped[col] = df_capped[col].clip(lower=lower_bound, upper=upper_bound)
    return df_capped

class DataPreparationPipeline:
    """
    Comprehensive Data Preparation Pipeline that encapsulates:
    - Zero replacement
    - Outlier handling
    - Imputation (Median / KNN / Iterative MICE)
    - Feature Engineering
    - Feature Scaling (Standard / Robust / MinMax)
    - Class Resampling (SMOTE / ADASYN)
    """
    def __init__(
        self,
        imputer_strategy: str = "iterative", # 'median', 'knn', 'iterative'
        scaler_strategy: str = "robust",     # 'standard', 'robust', 'minmax'
        handle_outliers: bool = True,
        feature_engineering: bool = True,
        resampler_strategy: Optional[str] = "smote", # 'smote', 'adasyn', None
        random_state: int = 42
    ):
        self.imputer_strategy = imputer_strategy
        self.scaler_strategy = scaler_strategy
        self.handle_outliers = handle_outliers
        self.feature_engineering = feature_engineering
        self.resampler_strategy = resampler_strategy
        self.random_state = random_state

        self.imputer = None
        self.scaler = None
        self.feature_engineer = DiabetesFeatureEngineer() if feature_engineering else None
        self.feature_names = None

    def _get_imputer(self):
        if self.imputer_strategy == "knn":
            return KNNImputer(n_neighbors=5, weights="distance")
        elif self.imputer_strategy == "iterative":
            return IterativeImputer(max_iter=20, random_state=self.random_state)
        elif self.imputer_strategy == "median":
            from sklearn.impute import SimpleImputer
            return SimpleImputer(strategy="median")
        else:
            raise ValueError(f"Unknown imputer strategy: {self.imputer_strategy}")

    def _get_scaler(self):
        if self.scaler_strategy == "robust":
            return RobustScaler()
        elif self.scaler_strategy == "standard":
            return StandardScaler()
        elif self.scaler_strategy == "minmax":
            return MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaler strategy: {self.scaler_strategy}")

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Tuple[np.ndarray, Optional[np.ndarray], list]:
        """
        Fit the preparation pipeline on training data and transform.
        """
        X_df = X.copy()
        
        # 1. Zero Cleaning
        X_df = clean_physiological_zeros(X_df)

        # 2. Outliers
        if self.handle_outliers:
            X_df = cap_outliers_iqr(X_df)

        # 3. Imputation
        self.imputer = self._get_imputer()
        X_imputed = pd.DataFrame(
            self.imputer.fit_transform(X_df),
            columns=X_df.columns,
            index=X_df.index
        )

        # 4. Feature Engineering
        if self.feature_engineering:
            X_feat = self.feature_engineer.transform(X_imputed)
        else:
            X_feat = X_imputed

        self.feature_names = list(X_feat.columns)

        # 5. Scaling
        self.scaler = self._get_scaler()
        X_scaled = self.scaler.fit_transform(X_feat)

        # 6. Resampling (Training only, if y provided)
        y_res = y.copy() if y is not None else None
        if y is not None and self.resampler_strategy is not None:
            if self.resampler_strategy == "smote":
                sampler = SMOTE(random_state=self.random_state)
                X_scaled, y_res = sampler.fit_resample(X_scaled, y)
            elif self.resampler_strategy == "adasyn":
                sampler = ADASYN(random_state=self.random_state)
                X_scaled, y_res = sampler.fit_resample(X_scaled, y)

        return X_scaled, (y_res.values if y_res is not None else None), self.feature_names

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform unseen test/production data using fitted state.
        """
        X_df = X.copy()
        X_df = clean_physiological_zeros(X_df)
        
        # Impute
        X_imputed = pd.DataFrame(
            self.imputer.transform(X_df),
            columns=X_df.columns,
            index=X_df.index
        )

        # Feature Engineering
        if self.feature_engineering:
            X_feat = self.feature_engineer.transform(X_imputed)
        else:
            X_feat = X_imputed

        # Scaling
        X_scaled = self.scaler.transform(X_feat)
        return X_scaled
