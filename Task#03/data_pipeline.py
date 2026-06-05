from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split


def load_and_clean_data(file_path: str) -> pd.DataFrame:
    """Loads the Heart Disease dataset and handles missing values."""
    print(f"📦 Loading dataset from: {file_path}")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        # Fallback if running with the standard online repository link
        url = "https://storage.googleapis.com/download.tensorflow.org/data/heart.csv"
        print(f"⚠️ Local file not found. Ingesting from fallback URL: {url}")
        df = pd.read_csv(url)

    # In standard UCI datasets, the target column might be named 'target' or 'target '
    df.columns = df.columns.str.strip()

    # Handle missing values if any exist
    if df.isnull().sum().sum() > 0:
        print("🔧 Found missing values. Applying median/mode imputation...")
        for col in df.columns:
            if df[col].dtype == "object":
                df[col].fillna(df[col].mode()[0], inplace=True)
            else:
                df[col].fillna(df[col].median(), inplace=True)
    else:
        print("✅ Data integrity check passed: No missing values found.")

    return df


def prepare_datasets(
    df: pd.DataFrame, target_col: str = "target", test_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Separates the features from the target label and splits into train/test sets."""
    if target_col not in df.columns:
        # Handle variance in different versions of the UCI dataset (e.g., 'chol' vs 'target')
        possible_targets = ["target", "num", "target_col"]
        for t in possible_targets:
            if t in df.columns:
                target_col = t
                break

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # For clinical classifications, a stratified split ensures balanced class distribution
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )