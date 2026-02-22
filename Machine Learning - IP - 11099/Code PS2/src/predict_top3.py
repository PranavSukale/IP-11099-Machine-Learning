import joblib
import numpy as np
from .feature_engineering_utils import add_engineered_features

MODEL_PATH = "models/career_model.pkl"
PREPROCESSOR_PATH = "models/preprocessor.pkl"
LABEL_ENCODER_PATH = "models/label_encoder.pkl"


def load_artifacts():
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    return model, preprocessor, label_encoder


def predict_top_3(input_df):
    model, preprocessor, label_encoder = load_artifacts()

    # ✅ APPLY SAME FEATURE ENGINEERING AS TRAINING
    input_df = add_engineered_features(input_df)

    # ✅ REMOVE ANY COLUMNS THAT AREN'T FEATURES (like 'Field', 'Career', etc)
    # Keep only numeric and object columns that should be in the input
    cols_to_drop = [col for col in input_df.columns if col in ["Field", "Career", "Career_Category"]]
    if cols_to_drop:
        input_df = input_df.drop(columns=cols_to_drop)

    # ✅ PREPROCESS
    X_processed = preprocessor.transform(input_df)

    # ✅ PROBABILITIES
    probabilities = model.predict_proba(X_processed)[0]

    # ✅ TOP-3
    top_indices = np.argsort(probabilities)[::-1][:3]

    results = []
    for idx in top_indices:
        category = label_encoder.inverse_transform([idx])[0]
        confidence = round(probabilities[idx] * 100, 2)
        results.append((category, confidence))

    return results
