import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


DATA_PATH = "data/processed/clean_data.csv"
MODEL_PATH = "models/career_model.pkl"
TARGET_COLUMN = "Career_Category"

def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError("Clean data not found. Run preprocessing first.")
    return pd.read_csv(DATA_PATH)


def train_models(X_train, y_train):
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            random_state=42,
            n_jobs=-1
        )
    }
    return models


def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    cv_score = cross_val_score(model, X_train, y_train, cv=5).mean()

    return accuracy, cv_score, classification_report(y_test, y_pred)


def main():
    df = load_data()

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = train_models(X_train, y_train)

    best_model = None
    best_score = 0

    for name, model in models.items():
        accuracy, cv_score, report = evaluate_model(
            model, X_train, y_train, X_test, y_test
        )

        print(f"\n📌 Model: {name}")
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"CV Accuracy:   {cv_score:.4f}")

        if cv_score > best_score:
            best_score = cv_score
            best_model = model

    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)

    print("\n✅ Best model saved successfully")
    print(f"📁 Model path: {MODEL_PATH}")


if __name__ == "__main__":
    main()
