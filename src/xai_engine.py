import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple

class ClinicalXAIEngine:
    """
    Explainable AI (XAI) & Clinical Prescriptive Decision Engine:
    - Patient-level Feature Attributions & Contribution Percentages
    - Counterfactual Goal Recommendations (Prescriptive Targets)
    """
    def __init__(self, model: Any, preprocessor: Any, feature_names: List[str]):
        self.model = model
        self.preprocessor = preprocessor
        self.feature_names = feature_names

    def get_feature_contributions(self, input_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculates patient-level feature attributions based on model weights/trees.
        """
        X_proc = self.preprocessor.transform(input_df)
        base_prob = float(self.model.predict_proba(X_proc)[0, 1])

        # Feature importances from base estimators or gradient tree
        importances = None
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
        elif hasattr(self.model, "named_estimators_") and "rf" in self.model.named_estimators_:
            importances = self.model.named_estimators_["rf"].feature_importances_
        else:
            importances = np.ones(len(self.feature_names)) / len(self.feature_names)

        # Approximate local attribution (Importance * Scaled Deviation)
        raw_vals = X_proc[0]
        local_impacts = raw_vals * importances

        # Normalize to percentage impacts
        total_impact = np.sum(np.abs(local_impacts)) + 1e-6
        pct_impacts = (local_impacts / total_impact) * 100.0

        contrib_list = []
        for feat, val, pct in zip(self.feature_names, raw_vals, pct_impacts):
            contrib_list.append({
                "feature": feat,
                "normalized_value": float(val),
                "impact_percent": float(pct),
                "direction": "Risk Driver (+)" if pct > 0 else "Protective Factor (-)"
            })

        contrib_df = pd.DataFrame(contrib_list)
        # Separate top drivers and protective factors
        top_risk_drivers = contrib_df[contrib_df["impact_percent"] > 0].sort_values(by="impact_percent", ascending=False).head(4).to_dict(orient="records")
        top_protective = contrib_df[contrib_df["impact_percent"] < 0].sort_values(by="impact_percent", ascending=True).head(4).to_dict(orient="records")

        return {
            "predicted_probability": base_prob,
            "top_risk_drivers": top_risk_drivers,
            "top_protective_factors": top_protective,
            "all_contributions": contrib_list
        }

    def generate_counterfactual_targets(self, input_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Generates actionable counterfactual patient goals to bring risk down.
        """
        current_data = input_df.copy()
        current_prob = float(self.model.predict_proba(self.preprocessor.transform(current_data))[0, 1])

        if current_prob < 0.35:
            return [{
                "parameter": "Overall Lifestyle",
                "current_value": "Optimal",
                "target_value": "Maintain",
                "expected_risk_reduction": "0%",
                "clinical_rationale": "Patient is already in the low-risk metabolic zone."
            }]

        recommendations = []

        # 1. Glucose Target Simulation
        curr_glucose = float(current_data["Glucose"].iloc[0])
        if curr_glucose > 100:
            target_glucose = max(95.0, curr_glucose * 0.80)
            sim_data = current_data.copy()
            sim_data["Glucose"] = target_glucose
            sim_prob = float(self.model.predict_proba(self.preprocessor.transform(sim_data))[0, 1])
            reduction = (current_prob - sim_prob) * 100.0
            if reduction > 2.0:
                recommendations.append({
                    "parameter": "Fasting Plasma Glucose",
                    "current_value": f"{curr_glucose:.0f} mg/dL",
                    "target_value": f"{target_glucose:.0f} mg/dL",
                    "expected_risk_reduction": f"-{reduction:.1f}%",
                    "clinical_rationale": "Adopting low-glycemic dietary interventions and carbohydrate restriction."
                })

        # 2. BMI Target Simulation
        curr_bmi = float(current_data["BMI"].iloc[0])
        if curr_bmi > 25.0:
            target_bmi = max(23.5, curr_bmi - 3.5)
            sim_data = current_data.copy()
            sim_data["BMI"] = target_bmi
            sim_prob = float(self.model.predict_proba(self.preprocessor.transform(sim_data))[0, 1])
            reduction = (current_prob - sim_prob) * 100.0
            if reduction > 2.0:
                recommendations.append({
                    "parameter": "Body Mass Index (BMI)",
                    "current_value": f"{curr_bmi:.1f} kg/m²",
                    "target_value": f"{target_bmi:.1f} kg/m²",
                    "expected_risk_reduction": f"-{reduction:.1f}%",
                    "clinical_rationale": "5-7% intentional body weight loss via 150 mins/week moderate exercise."
                })

        # 3. Blood Pressure Target Simulation
        curr_bp = float(current_data["BloodPressure"].iloc[0])
        if curr_bp > 80.0:
            target_bp = 75.0
            sim_data = current_data.copy()
            sim_data["BloodPressure"] = target_bp
            sim_prob = float(self.model.predict_proba(self.preprocessor.transform(sim_data))[0, 1])
            reduction = (current_prob - sim_prob) * 100.0
            if reduction > 1.0:
                recommendations.append({
                    "parameter": "Diastolic Blood Pressure",
                    "current_value": f"{curr_bp:.0f} mm Hg",
                    "target_value": f"{target_bp:.0f} mm Hg",
                    "expected_risk_reduction": f"-{reduction:.1f}%",
                    "clinical_rationale": "DASH diet (sodium reduction) and aerobic conditioning."
                })

        if not recommendations:
            recommendations.append({
                "parameter": "Comprehensive Lifestyle Protocol",
                "current_value": "Borderline Elevation",
                "target_value": "Clinical Guidance",
                "expected_risk_reduction": "Variable",
                "clinical_rationale": "Consult with a clinical dietitian for balanced macronutrient distribution and regular aerobic activity."
            })

        return recommendations
