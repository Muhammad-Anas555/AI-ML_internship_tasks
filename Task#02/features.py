from typing import List, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split


def create_next_day_target(
    df: pd.DataFrame, feature_cols: List[str]
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Shifts the close price to create a predictive target variable for the next day."""
    working_df = df.copy()

    # Align today's metrics with tomorrow's close price
    working_df["Next_Close"] = working_df["Close"].shift(-1)

    # Remove the last row since its tomorrow target is unknown
    working_df.dropna(inplace=True)

    X = working_df[feature_cols]
    y = working_df["Next_Close"]

    return X, y, working_df


def split_time_series_data(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Splits features and targets sequentially without shuffling to preserve time sequence."""
    # shuffle=False is critical to avoid chronological data leakage
    return train_test_split(X, y, test_size=test_size, shuffle=False)