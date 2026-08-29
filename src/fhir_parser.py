import json
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

# Standard LOINC (Logical Observation Identifiers Names and Codes) for Clinical Labs
LOINC_MAP = {
    "1558-6": "Glucose",            # Fasting Glucose (mg/dL)
    "2339-0": "Glucose",            # Glucose [Mass/volume] in Blood
    "8480-6": "SystolicBP",         # Systolic Blood Pressure (mm Hg)
    "8462-4": "BloodPressure",      # Diastolic Blood Pressure (mm Hg)
    "39156-5": "BMI",               # Body Mass Index (kg/m^2)
    "4548-4": "HbA1c",              # Hemoglobin A1c (% of total Hgb)
    "2085-9": "HDL",                # HDL Cholesterol (mg/dL)
    "2093-3": "TotalCholesterol",   # Total Cholesterol (mg/dL)
    "2571-8": "Triglycerides",      # Triglycerides (mg/dL)
    "20570-8": "SkinThickness",     # Triceps skin fold thickness (mm)
    "20448-7": "Insulin"            # Insulin [Units/volume] in Serum or Plasma
}

class FHIRResourceParser:
    """
    HL7 FHIR (Fast Healthcare Interoperability Resources) Parser.
    Extracts clinical observations and patient demographics from standard FHIR JSON bundles.
    """
    def __init__(self):
        self.loinc_map = LOINC_MAP

    def parse_bundle(self, bundle: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parses a standard FHIR Bundle JSON object.
        """
        extracted = {
            "Pregnancies": 0,
            "Glucose": np.nan,
            "BloodPressure": np.nan,
            "SkinThickness": np.nan,
            "Insulin": np.nan,
            "BMI": np.nan,
            "DiabetesPedigreeFunction": 0.45, # Default population baseline if absent
            "Age": 35,
            "PatientID": "ANONYMOUS_PATIENT",
            "Gender": "female"
        }

        entries = bundle.get("entry", [])
        for item in entries:
            resource = item.get("resource", {})
            res_type = resource.get("resourceType")

            # 1. Parse Patient Demographics
            if res_type == "Patient":
                extracted["PatientID"] = resource.get("id", "UNKNOWN_ID")
                extracted["Gender"] = resource.get("gender", "unknown")
                # Calculate age from birthDate
                birth_date = resource.get("birthDate")
                if birth_date:
                    try:
                        birth_year = int(birth_date.split("-")[0])
                        current_year = 2026
                        extracted["Age"] = max(18, current_year - birth_year)
                    except Exception:
                        pass

            # 2. Parse Observations (Labs & Vitals)
            elif res_type == "Observation":
                coding = resource.get("code", {}).get("coding", [])
                for c in coding:
                    code = c.get("code")
                    if code in self.loinc_map:
                        param_name = self.loinc_map[code]
                        val = resource.get("valueQuantity", {}).get("value")
                        if val is not None:
                            extracted[param_name] = float(val)

                # Check for Blood Pressure Component (Systolic & Diastolic combined)
                components = resource.get("component", [])
                for comp in components:
                    comp_coding = comp.get("code", {}).get("coding", [])
                    for cc in comp_coding:
                        code = cc.get("code")
                        if code in self.loinc_map:
                            param_name = self.loinc_map[code]
                            val = comp.get("valueQuantity", {}).get("value")
                            if val is not None:
                                extracted[param_name] = float(val)

        return extracted

    def fhir_to_dataframe(self, fhir_json_or_dict: Any) -> pd.DataFrame:
        """
        Converts FHIR JSON string or dictionary into a normalized DataFrame ready for preprocessing.
        """
        if isinstance(fhir_json_or_dict, str):
            bundle = json.loads(fhir_json_or_dict)
        else:
            bundle = fhir_json_or_dict

        data_dict = self.parse_bundle(bundle)
        cols = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
        df = pd.DataFrame([{c: data_dict.get(c, np.nan) for c in cols}])
        return df

def generate_sample_fhir_bundle(
    patient_id: str = "PATIENT-84920",
    glucose: float = 168.0,
    bp: float = 86.0,
    bmi: float = 34.2,
    age: int = 48,
    insulin: float = 195.0
) -> Dict[str, Any]:
    """
    Utility generator to simulate standard hospital EHR FHIR JSON bundles.
    """
    return {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [
            {
                "resource": {
                    "resourceType": "Patient",
                    "id": patient_id,
                    "gender": "female",
                    "birthDate": f"{2026 - age}-05-14"
                }
            },
            {
                "resource": {
                    "resourceType": "Observation",
                    "status": "final",
                    "code": {"coding": [{"system": "http://loinc.org", "code": "1558-6", "display": "Fasting Glucose"}]},
                    "valueQuantity": {"value": glucose, "unit": "mg/dL"}
                }
            },
            {
                "resource": {
                    "resourceType": "Observation",
                    "status": "final",
                    "code": {"coding": [{"system": "http://loinc.org", "code": "8462-4", "display": "Diastolic Blood Pressure"}]},
                    "valueQuantity": {"value": bp, "unit": "mm[Hg]"}
                }
            },
            {
                "resource": {
                    "resourceType": "Observation",
                    "status": "final",
                    "code": {"coding": [{"system": "http://loinc.org", "code": "39156-5", "display": "Body Mass Index"}]},
                    "valueQuantity": {"value": bmi, "unit": "kg/m2"}
                }
            },
            {
                "resource": {
                    "resourceType": "Observation",
                    "status": "final",
                    "code": {"coding": [{"system": "http://loinc.org", "code": "20448-7", "display": "Serum Insulin"}]},
                    "valueQuantity": {"value": insulin, "unit": "uIU/mL"}
                }
            }
        ]
    }
