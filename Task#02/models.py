import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


def train_linear_regression(X_train, y_train, X_test) -> np.ndarray:
    """Trains a Linear Regression model and predicts on test features."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model.predict(X_test)


def train_random_forest(X_train, y_train, X_test) -> np.ndarray:
    """Trains a Random Forest Regressor and predicts on test features."""
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model.predict(X_test)


def evaluate_predictions(model_name: str, y_true, y_pred) -> None:
    """Calculates and logs standard regression evaluation metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"📊 {model_name} Results:")
    print(f"   -> Mean Absolute Error (MAE): ${mae:.2f}")
    print(f"   -> R-squared (R²) Score:     {r2:.4f}\n")