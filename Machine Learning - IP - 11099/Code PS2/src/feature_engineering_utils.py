import pandas as pd
import os

INPUT_PATH = "data/raw/career_with_category.csv"
OUTPUT_PATH = "data/raw/career_fe.csv"

def add_engineered_features(df):
    """Apply feature engineering transformations to a dataframe."""
    df = df.copy()
    
    # --- Feature Engineering ---
    df["technical_strength"] = (
        df["Coding_Skills"] +
        df["Analytical_Skills"] +
        df["Problem_Solving_Skills"]
    ) / 3

    df["soft_skill_strength"] = (
        df["Communication_Skills"] +
        df["Presentation_Skills"] +
        df["Teamwork_Skills"]
    ) / 3

    df["leadership_index"] = (
        df["Leadership_Positions"] +
        df["Networking_Skills"]
    )

    df["experience_index"] = (
        df["Internships"] +
        df["Projects"] +
        df["Research_Experience"] +
        df["Industry_Certifications"]
    )

    df["academic_index"] = (
        df["GPA"] +
        df["Field_Specific_Courses"]
    )
    
    return df

def main():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError("Input dataset not found")

    df = pd.read_csv(INPUT_PATH)

    # Apply feature engineering
    df = add_engineered_features(df)

    df.to_csv(OUTPUT_PATH, index=False)

    print("✅ Feature engineering completed")
    print("📁 Saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
