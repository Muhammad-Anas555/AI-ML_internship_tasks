# Data Science Portfolio: Machine Learning & Engineering Pipelines

This repository contains a portfolio of engineering-focused data tasks, demonstrating structured pipeline design, data fetching, time-series feature engineering, and predictive regression modeling.

## 📈 Task 2: Short-Term Stock Price Prediction Pipeline

### 🎯 Task Objective
The objective of this task is to design and implement a production-grade, modular machine learning pipeline capable of predicting a stock's next-day closing valuation using daily market indicators. This covers establishing connection frameworks with data APIs, configuring time-series dependencies sequentially to eliminate data leakage, applying regression engines, and evaluating predictive error metrics.

### 📊 Dataset Used
* **Source:** Yahoo Finance API (via the `yfinance` library wrapper)
* **Asset Tracked:** Apple Inc. (`AAPL`)
* **Data Range:** 5 Years of daily historical market transaction data
* **Input Features ($X$):**
  * `Open`: Initial price at market open.
  * `High`: Peak price reached during the trading session.
  * `Low`: Minimum price reached during the trading session.
  * `Volume`: Total quantity of shares transacted throughout the day.
* **Target Label ($y$):** * `Next_Close`: The closing valuation shifted chronologically by $-1$ day (`Close.shift(-1)`), forcing the models to use today's observable information to project tomorrow's closing price.

### Project Architecture & Modules
To maintain clean separation of concerns, the predictive codebase is separated into independent, reusable modules:
* `data_fetcher.py`: Manages network requests, API data downloading, and multi-index cleanups.
* `features.py`: Configures the time-shifted target variable and handles sequential train/test dataset splits.
* `models.py`: Abstracts model initializations, model training routines, and statistical metric calculations.
* `main.py`: The root orchestrator script that manages runtime execution and displays comparison visualizations.

### 🤖 Models Applied
Two different classes of regression architectures were used to contrast linear vs. non-linear patterns:
1. **Linear Regression (`LinearRegression`):** Assumes a straight-line scaling function between input day metrics and the target price.
2. **Random Forest Regressor (`RandomForestRegressor`):** An ensemble machine learning model leveraging 100 parallel decision trees to identify non-linear changes without overfitting.

### 🔑 Key Results and Findings
1. **The Necessity of Temporal Continuity:** Because chronological sequencing matters in time-series datasets, data shuffling was disabled (`shuffle=False`). Shuffling breaks chronological integrity and introduces severe data leakage, inflating performance metrics artificially.
2. **The Lag Phenomenon:** Performance reviews reveal an expected "lag effect" common in short-term regression engines. The models frequently lean on the previous day's momentum as a baseline, trailing slightly when hit with unexpected macroeconomic market shocks.
3. **Performance Metrics:** Mean Absolute Error (MAE) was tracked to identify the average absolute dollar deviation per predicted trade, paired with R-squared ($R^2$) to evaluate what percentage of overall target price variation the models explained.