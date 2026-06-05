import matplotlib.pyplot as plt                     # import libraries & load dataset
import seaborn as sns
sns.set_theme(style="whitegrid")
# Load built-in Iris dataset
df = sns.load_dataset("iris")

# Display dataset shape (Rows, Columns)            # inspect dataset
print("Dataset Shape:", df.shape)
print("-" * 50)
print("Column Names:", df.columns.tolist())
print("-" * 50)
# View first 5 rows
print("First 5 Rows:")
print(df.head())

# Technical summary of dataframe                  # Summary
print("\n--- Dataframe Info ---")
df.info()
# Statistical summary of columns
print("\n--- Statistical Summary ---")
print(df.describe())


# A). Scatter Plot (Feature Relationships)             # Data Visualization
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df, x="sepal_length", y="sepal_width", hue="species", palette="deep"
)
plt.title("Sepal Length vs Sepal Width")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.legend(title="Species")
plt.show()

# B). Histograms (Value Distributions)
features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()  # Flatten 2D array to 1D for easy looping
for i, col in enumerate(features):
    sns.histplot(
        data=df, x=col, hue="species", kde=True, multiple="stack", ax=axes[i]
    )
    axes[i].set_title(f"Distribution of {col.replace('_', ' ').title()}")
    axes[i].set_xlabel("")
plt.tight_layout()
plt.show()

# C). Box Plots (Identifying Outliers)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()
for i, col in enumerate(features):
    sns.boxplot(data=df, x="species", y=col, ax=axes[i], palette="pastel")
    axes[i].set_title(f"{col.replace('_', ' ').title()} by Species")
    axes[i].set_ylabel("")
    axes[i].set_xlabel("")
plt.tight_layout()
plt.show()
