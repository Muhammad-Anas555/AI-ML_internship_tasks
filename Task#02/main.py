import matplotlib.pyplot as plt
import seaborn as sns

# Import our custom modules
from data_fetcher import fetch_stock_data
from features import create_next_day_target, split_time_series_data
from models import (
    evaluate_predictions,
    train_linear_regression,
    train_random_forest,
)

def main():
    # Configuration profiles
    TICKER = "AAPL"
    START_DATE = "2021-01-01"
    END_DATE = "2026-01-01"
    FEATURES = ["Open", "High", "Low", "Volume"]

    # 1. Fetch
    raw_data = fetch_stock_data(TICKER, START_DATE, END_DATE)

    # 2. Feature Engineering
    X, y, processed_df = create_next_day_target(raw_data, FEATURES)

    # 3. Train-Test Split
    X_train, X_test, y_train, y_test = split_time_series_data(X, y, test_size=0.2)

    # 4. Modeling & Prediction
    lr_preds = train_linear_regression(X_train, y_train, X_test)
    rf_preds = train_random_forest(X_train, y_train, X_test)

    # 5. Evaluation Metrics Logging
    print("\n" + "=" * 40)
    print("      EVALUATING MODEL METRICS      ")
    print("=" * 40)
    evaluate_predictions("Linear Regression", y_test, lr_preds)
    evaluate_predictions("Random Forest Regressor", y_test, rf_preds)

    # 6. Visualization
    sns.set_theme(style="whitegrid")
    test_dates = processed_df.index[len(processed_df) - len(y_test) :]

    plt.figure(figsize=(14, 7))
    plt.plot(
        test_dates, y_test.values, label="Actual Close", color="black", linewidth=2
    )
    plt.plot(
        test_dates,
        lr_preds,
        label="Linear Regression",
        color="blue",
        linestyle="--",
    )
    plt.plot(
        test_dates, rf_preds, label="Random Forest", color="orange", linestyle=":"
    )

    plt.title(
        f"{TICKER} Next-Day Closing Price Prediction Pipeline",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("Timeline Date")
    plt.ylabel("Stock Valuation (USD)")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()


    #  How to Run Your Modular Application          ----         "python main.py" in terminal.