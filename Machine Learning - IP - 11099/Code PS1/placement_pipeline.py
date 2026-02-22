import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score


# ===================================
# 1️⃣ LOAD DATA
# ===================================

DATA_PATH = "dataset/01 Train Data.xlsx"

df = pd.read_excel(DATA_PATH)

df = df[df["Placement Status"].notna()].copy()

df["Placement Status"] = df["Placement Status"].map({
    "Placed": 1,
    "Not placed": 0
})

print("\nTotal Rows:", len(df))
print("\nClass Distribution:\n", df["Placement Status"].value_counts(normalize=True))


# ===================================
# 2️⃣ BASELINE RATE
# ===================================

base_rate = df["Placement Status"].mean()
print("\nBase Placement Rate:", round(base_rate, 3))


# ===================================
# 3️⃣ FEATURES (No modification)
# ===================================

features = ["CGPA", "Speaking Skills", "ML Knowledge"]

X = df[features]
y = df["Placement Status"]


# ===================================
# 4️⃣ TRAIN / TEST SPLIT
# ===================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ===================================
# 5️⃣ LOGISTIC REGRESSION PIPELINE
# ===================================

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(max_iter=1000))
])

# Calibrated model
model = CalibratedClassifierCV(pipeline, method="sigmoid", cv=5)

model.fit(X_train, y_train)


# ===================================
# 6️⃣ VALIDATION
# ===================================

probs = model.predict_proba(X_test)[:, 1]

# Decision rule: probability > base rate
preds = (probs > base_rate).astype(int)

accuracy = accuracy_score(y_test, preds)
roc = roc_auc_score(y_test, probs)
conf_matrix = confusion_matrix(y_test, preds)

print("\nValidation Accuracy:", round(accuracy, 3))
print("Validation ROC-AUC:", round(roc, 3))
print("Confusion Matrix:\n", conf_matrix)


# ===================================
# 7️⃣ REPEATED CV
# ===================================

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=cv)

print("Repeated CV Mean Accuracy:", round(cv_scores.mean(), 3))


# ===================================
# 8️⃣ FULL DATA RISK SCORES
# ===================================

all_probs = model.predict_proba(X)[:, 1]
all_preds = (all_probs > base_rate).astype(int)

df["Placement_Probability"] = all_probs
df["Placement_Prediction"] = all_preds


# ===================================
# 9️⃣ SAVE OUTPUT
# ===================================

os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

OUTPUT_PATH = "outputs/placement_risk_predictions.xlsx"

df.to_excel(OUTPUT_PATH, index=False)
joblib.dump(model, "models/placement_risk_model.pkl")

print("\nPredictions saved to:", OUTPUT_PATH)
