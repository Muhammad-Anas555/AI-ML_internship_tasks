from analytics import extract_feature_importance
from data_pipeline import load_and_clean_data, prepare_datasets
from modeling import (
    evaluate_classifier,
    train_decision_tree,
    train_logistic_regression,
)

def main():
    # 1. Pipeline Settings
    DATA_PATH = "heart.csv"  

    # 2. Ingest and Clean
    df = load_and_clean_data(DATA_PATH)

    # 3. Structural Processing & Splitting
    X_train, X_test, y_train, y_test = prepare_datasets(df, target_col="target")

    # 4. Train & Evaluate Model A: Logistic Regression
    lr_model = train_logistic_regression(X_train, y_train)
    evaluate_classifier(lr_model, X_test, y_test, "Logistic Regression")
    extract_feature_importance(lr_model, X_train.columns.tolist())

    # 5. Train & Evaluate Model B: Decision Tree
    dt_model = train_decision_tree(X_train, y_train)
    evaluate_classifier(dt_model, X_test, y_test, "Decision Tree")
    extract_feature_importance(dt_model, X_train.columns.tolist())

if __name__ == "__main__":
    main()