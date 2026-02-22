import pandas as pd
import numpy as np
import os
import joblib

from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


RAW_DATA_PATH = "data/raw/career_fe.csv"
PROCESSED_DATA_PATH = "data/processed/clean_data.csv"
PREPROCESSOR_PATH = "models/preprocessor.pkl"
TARGET_COLUMN = "Career_Category"

def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")
    return pd.read_csv(path)


def validate_data(df):
    if TARGET_COLUMN not in df.columns:
        raise ValueError("Target column 'Career' not found")
    if df.isnull().mean().max() > 0.3:
        raise ValueError("Too many missing values")
    return True


def build_preprocessor(X):
    """Build preprocessor based on the X dataframe that will be used for training.
    X should already have target and non-feature columns removed."""
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols)
        ]
    )

    return preprocessor, numeric_cols, categorical_cols


def encode_target(y):
    le = LabelEncoder()
    return le.fit_transform(y), le


def main():
    df = load_data(RAW_DATA_PATH)
    validate_data(df)

    # Drop target and non-feature columns that aren't available during prediction
    cols_to_drop = [TARGET_COLUMN, "Field", "Career"]
    cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    X = df.drop(columns=cols_to_drop)
    y = df[TARGET_COLUMN]

    # Build preprocessor on the clean X
    preprocessor, numeric_cols, categorical_cols = build_preprocessor(X)
    
    X_processed = preprocessor.fit_transform(X)
    y_encoded, label_encoder = encode_target(y)

    # 🔥 PRESERVE FEATURE NAMES (MOST IMPORTANT PART)
    feature_names = numeric_cols.copy()
    
    if categorical_cols:  # Only get onehot features if there are categorical columns
        onehot_features = preprocessor.named_transformers_["cat"] \
            .named_steps["onehot"] \
            .get_feature_names_out(categorical_cols)
        feature_names = numeric_cols + list(onehot_features)

    processed_df = pd.DataFrame(X_processed, columns=feature_names)
    processed_df[TARGET_COLUMN] = y_encoded

    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    processed_df.to_csv(PROCESSED_DATA_PATH, index=False)

    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    joblib.dump(label_encoder, "models/label_encoder.pkl")

    print("✅ Data preprocessing completed successfully")
    print(f"📁 Clean data saved to: {PROCESSED_DATA_PATH}")
    print(f"📁 Preprocessor saved to: {PREPROCESSOR_PATH}")


if __name__ == "__main__":
    main()
