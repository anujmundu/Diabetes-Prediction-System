import os
import pandas as pd
import requests

DATASET_URLS = [
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",
    "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv",
    "https://raw.githubusercontent.com/saurabh0910/Pima-Indians-Diabetes-Dataset/master/diabetes.csv"
]

COLUMN_NAMES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome"
]

def load_or_download_dataset(data_dir: str = "data", filename: str = "diabetes.csv") -> pd.DataFrame:
    """
    Loads the diabetes dataset from local data directory.
    If not found, attempts to download it automatically from standard public mirrors.
    """
    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, filename)

    if os.path.exists(file_path):
        print(f"[DataLoader] Found existing dataset at: {file_path}")
        df = pd.read_csv(file_path)
        # Normalize column names if headerless or different
        if len(df.columns) == 9 and list(df.columns) != COLUMN_NAMES:
            if all(isinstance(col, (int, float)) or col.isdigit() for col in df.columns):
                df = pd.read_csv(file_path, names=COLUMN_NAMES)
            else:
                # Rename standard lowercase / formatted column variations
                rename_map = {c: COLUMN_NAMES[i] for i, c in enumerate(df.columns)}
                df = df.rename(columns=rename_map)
        return df

    print(f"[DataLoader] Dataset not found locally at {file_path}. Fetching from repository mirror...")
    for url in DATASET_URLS:
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                with open(file_path, "wb") as f:
                    f.write(response.content)
                print(f"[DataLoader] Successfully downloaded dataset from {url} to {file_path}")
                
                # Check if file has header
                df_test = pd.read_csv(file_path)
                if df_test.columns[0] != "Pregnancies" and df_test.columns[0] != "pregnancies":
                    # Likely headerless UCI format
                    df = pd.read_csv(file_path, names=COLUMN_NAMES)
                    df.to_csv(file_path, index=False)
                else:
                    df = df_test
                    df.columns = COLUMN_NAMES
                    df.to_csv(file_path, index=False)
                return df
        except Exception as e:
            print(f"[DataLoader] Failed to download from {url}: {e}")

    raise FileNotFoundError(
        f"Unable to locate or download diabetes dataset. Please place 'diabetes.csv' in '{data_dir}/' directory."
    )

if __name__ == "__main__":
    df = load_or_download_dataset()
    print("Dataset Shape:", df.shape)
    print(df.head())
