import pandas as pd

DATA_PATH = "data/raw/career_with_category.csv"


def recommend_fields_and_careers(input_df, predicted_category, top_n_fields=2, top_n_careers=3):
    """
    Returns top fields and careers based on predicted category
    """

    df = pd.read_csv(DATA_PATH)

    # Filter by predicted career category
    df_cat = df[df["Career_Category"] == predicted_category]

    if df_cat.empty:
        return [], []

    # -------------------------
    # FIELD RECOMMENDATION
    # -------------------------
    top_fields = (
        df_cat["Field"]
        .value_counts()
        .head(top_n_fields)
        .index
        .tolist()
    )

    # -------------------------
    # CAREER RECOMMENDATION
    # -------------------------
    df_field = df_cat[df_cat["Field"].isin(top_fields)]

    top_careers = (
        df_field["Career"]
        .value_counts()
        .head(top_n_careers)
        .index
        .tolist()
    )

    return top_fields, top_careers
