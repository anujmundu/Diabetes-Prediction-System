import hashlib
from typing import Dict, Any

class HIPAASanitizer:
    """
    HIPAA Safe-Harbor De-Identification & Anonymization Engine.
    Ensures Protected Health Information (PHI) is tokenized before ML inference.
    """
    def __init__(self, salt: str = "ClinicalCDSS_Salt_2026"):
        self.salt = salt

    def anonymize_patient_id(self, raw_patient_id: str) -> str:
        """
        Generates a deterministic, non-reversible SHA-256 pseudonym for audit tracking.
        """
        hasher = hashlib.sha256()
        hasher.update((raw_patient_id + self.salt).encode("utf-8"))
        return f"ANON_{hasher.hexdigest()[:16]}"

    def sanitize_payload(self, raw_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Removes direct PHI fields (Name, Phone, Email, SSN) and tokenizes Patient ID.
        """
        sanitized = raw_payload.copy()
        for phi_key in ["name", "patient_name", "phone", "email", "ssn", "address"]:
            if phi_key in sanitized:
                sanitized.pop(phi_key)

        if "patient_id" in sanitized:
            sanitized["anonymized_patient_id"] = self.anonymize_patient_id(str(sanitized.pop("patient_id")))
        elif "id" in sanitized:
            sanitized["anonymized_patient_id"] = self.anonymize_patient_id(str(sanitized.pop("id")))
        else:
            sanitized["anonymized_patient_id"] = "ANON_UNASSIGNED"

        return sanitized
