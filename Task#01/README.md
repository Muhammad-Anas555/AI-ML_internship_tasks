# Data Exploration and Visualization: Iris Dataset

## 🎯 Task Objective
The primary objective of this task is to establish a foundational understanding of Exploratory Data Analysis (EDA) workflows. This involves loading raw data, inspecting its structure, checking data integrity, generating descriptive statistical summaries, and utilizing visual representations to identify underlying trends, patterns, data distributions, and anomalies (outliers).

## 📊 Dataset Used
* **Dataset Name:** Classic Iris Dataset
* **Dimensions:** 150 rows, 5 columns (Shape: `(150, 5)`)
* **Attributes / Features:**
  1. `sepal_length` (Continuous Numerical - cm)
  2. `sepal_width` (Continuous Numerical - cm)
  3. `petal_length` (Continuous Numerical - cm)
  4. `petal_width` (Continuous Numerical - cm)
  5. `species` (Categorical Target - *Setosa*, *Versicolor*, *Virginica*)

## 🤖 Models Applied
As this assignment is strictly centered on **Exploratory Data Analysis (EDA)** and data visualization, **no formal Machine Learning predictive models (e.g., classifiers) were applied**. 

Instead, descriptive statistical models and data visualization structures were used to understand the separability of the classes:
* **Descriptive Metrics:** Summary metrics via pandas `.describe()` (Mean, Standard Deviation, Quartiles).
* **Data Visualization Frameworks:** * **Relationship Analysis:** Scatter Plots via `seaborn.scatterplot`
  * **Distribution Analysis:** Histograms with Kernel Density Estimations (KDE) via `seaborn.histplot`
  * **Anomaly / Dispersion Analysis:** Box Plots via `seaborn.boxplot`

## 🔑 Key Results and Findings
1. **Clear Class Separability:** The scatter and distribution plots show that the **Setosa** species is completely distinct and linearly separable from *Versicolor* and *Virginica*, particularly when looking at `petal_length` and `petal_width`. 
2. **Feature Importance for Identification:** Petal dimensions (`petal_length` and `petal_width`) show much tighter clustering and distinct boundaries per species than sepal dimensions, making them the most critical features for any future machine learning classification models.
3. **Outlier Detection:** The box plots successfully flagged data anomalies. Specifically, **Sepal Width** contains a small number of clear statistical outliers (points beyond the whiskers) for both the *Setosa* and *Virginica* species, which would require preprocessing before feeding into a machine learning model.
4. **Normal Distribution Trends:** The `sepal_width` feature across the entire dataset exhibits a clean bell-shaped curve (normal distribution), whereas petal features show clear multi-modal distributions due to distinct species differences.