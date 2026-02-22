import pandas as pd
import os

RAW_DATA_PATH = "data/raw/career_path_in_all_field.csv"
OUTPUT_PATH = "data/raw/career_with_category.csv"

def map_category(field):
    field = field.lower()

    if field in ["computer science", "information technology", "software engineering"]:
        return "Technology"
    if field in ["electronics", "mechanical", "civil", "electrical"]:
        return "Core Engineering"
    if field in ["management", "business", "commerce"]:
        return "Management"
    if field in ["medicine", "pharmacy", "biotechnology", "health sciences"]:
        return "Healthcare"
    if field in ["data science", "statistics", "mathematics"]:
        return "Data & Analytics"
    if field in ["design", "arts", "media"]:
        return "Design & Creative"
    if field in ["law", "public policy", "political science"]:
        return "Public & Legal"
    if field in ["education", "teaching", "training"]:
        return "Education"
    if field in ["science", "research", "physics", "chemistry", "biology"]:
        return "Research & Science"

    return "Other"


def main():
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError("Raw dataset not found")

    df = pd.read_csv(RAW_DATA_PATH)

    df["Career_Category"] = df["Field"].apply(map_category)

    df.to_csv(OUTPUT_PATH, index=False)

    print("✅ Career_Category created successfully")
    print("📁 Saved to:", OUTPUT_PATH)
    print("📊 Category distribution:")
    print(df["Career_Category"].value_counts())


if __name__ == "__main__":
    main()
