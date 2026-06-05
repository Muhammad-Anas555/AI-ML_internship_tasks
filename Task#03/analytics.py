# analytics.py
import numpy as np
import pandas as pd

def extract_feature_importance(model, feature_names: list) -> None:
    """Extracts and displays feature weights depending on the model architecture type."""
    print(f"\n=====================================")
    print("      FEATURE IMPORTANCE ANALYSIS      ")
    print("=====================================")

    # Check if the model uses coefficients (Logistic Regression) or tree feature importances
    if hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
        metric_name = "Absolute Coefficient"
    elif hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        metric_name = "Gini Importance Weight"
    else:
        print("⚠️ Model architecture type not supported for static extraction.")
        return

    # Create a structured frame
    importance_df = pd.DataFrame(
        {"Feature": feature_names, metric_name: importances}
    ).sort_values(by=metric_name, ascending=False)

    print(importance_df.to_string(index=False))