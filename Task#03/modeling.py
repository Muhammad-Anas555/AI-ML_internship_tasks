import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.tree import DecisionTreeClassifier

def train_logistic_regression(X_train, y_train) -> LogisticRegression:
    """Trains a Logistic Regression classifier with scaled iteration limits."""
    # max_iter is increased to ensure model convergence on medical attributes
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_decision_tree(X_train, y_train) -> DecisionTreeClassifier:
    """Trains a Decision Tree classifier optimized to prevent overfitting."""
    model = DecisionTreeClassifier(max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_classifier(model, X_test, y_test, model_name: str) -> None:
    """Computes and displays structural metric matrix validations."""
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)
    auc_score = roc_auc_score(y_test, probabilities)

    print(f"\n=====================================")
    print(f"   {model_name.upper()} PERFORMANCE METRICS   ")
    print(f"=====================================")
    print(f"▶️ Accuracy Score:          {accuracy:.4f}")
    print(f"▶️ ROC-AUC Score:          {auc_score:.4f}")
    print(f"▶️ Confusion Matrix:\n{cm}")

    # Generate plots sequentially
    _plot_roc_curve(y_test, probabilities, model_name)

def _plot_roc_curve(y_test, probabilities, model_name: str) -> None:
    """Generates the Receiver Operating Characteristic (ROC) curve."""
    fpr, tpr, _ = roc_curve(y_test, probabilities)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc_score(y_test, probabilities):.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--")  # Baseline reference line
    plt.title(f"ROC Curve: {model_name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()