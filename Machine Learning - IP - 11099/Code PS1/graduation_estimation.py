import pandas as pd
import os
from datetime import datetime


# ======================================
# 1️⃣ LOAD DATA
# ======================================

DATA_PATH = "dataset/Final Lead Data.xlsx"

df = pd.read_excel(DATA_PATH)

print("\nTotal Rows:", len(df))


# ======================================
# 2️⃣ PREPARE YEAR LEVEL
# ======================================

# Convert Academic Year to numeric safely
df["Academic Year"] = pd.to_numeric(df["Academic Year"], errors="coerce")


# Extract year level from text column (fallback)
def extract_year_from_text(text):
    if pd.isna(text):
        return None

    text = str(text).lower()

    if "1" in text:
        return 1
    elif "2" in text:
        return 2
    elif "3" in text:
        return 3
    elif "4" in text or "final" in text:
        return 4
    else:
        return None


df["Text_Year_Level"] = df["What is your current academic year?"].apply(
    extract_year_from_text
)

# Use Academic Year first, else fallback to text
df["Final_Year_Level"] = df["Academic Year"].combine_first(
    df["Text_Year_Level"]
)

print("\nMissing Final Year Level Count:",
      df["Final_Year_Level"].isna().sum())


# ======================================
# 3️⃣ ASSUME COURSE DURATION
# ======================================

COURSE_DURATION = 4
current_year = datetime.now().year


# ======================================
# 4️⃣ CALCULATE ESTIMATED GRADUATION YEAR
# ======================================

df["Estimated_Graduation_Year"] = df["Final_Year_Level"].apply(
    lambda x: current_year + (COURSE_DURATION - x)
    if pd.notna(x)
    else None
)


# ======================================
# 5️⃣ CREATE 0/1 GRADUATION FLAG
# ======================================

df["Graduation_Flag"] = df["Estimated_Graduation_Year"].apply(
    lambda x: 1 if pd.notna(x) and x <= current_year else 0
)


# ======================================
# 6️⃣ SAVE OUTPUT
# ======================================

os.makedirs("outputs", exist_ok=True)

OUTPUT_PATH = "outputs/graduation_estimation.xlsx"

df.to_excel(OUTPUT_PATH, index=False)

print("\nGraduation estimation saved to:", OUTPUT_PATH)
