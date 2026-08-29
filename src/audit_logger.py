import os
import json
import pandas as pd
from typing import Dict, Any, Optional

LOGS_DIR = "logs"
AUDIT_LOG_FILE = os.path.join(LOGS_DIR, "clinical_audit_ledger.jsonl")

def append_audit_log(entry: Dict[str, Any], log_file: str = AUDIT_LOG_FILE):
    """
    Appends an immutable clinical decision record to the persistent JSON Lines audit ledger.
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

def read_recent_audit_logs(limit: int = 25, log_file: str = AUDIT_LOG_FILE) -> pd.DataFrame:
    """
    Reads the most recent audit records from the ledger in reverse chronological order.
    """
    if not os.path.exists(log_file):
        return pd.DataFrame()
    records = []
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line.strip()))
                except Exception:
                    pass
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    return df.tail(limit).iloc[::-1]
