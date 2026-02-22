def interpret_other(input_df):
    tech_score = (
        input_df["Coding_Skills"].iloc[0] +
        input_df["Analytical_Skills"].iloc[0] +
        input_df["Problem_Solving_Skills"].iloc[0]
    )

    soft_score = (
        input_df["Communication_Skills"].iloc[0] +
        input_df["Presentation_Skills"].iloc[0] +
        input_df["Teamwork_Skills"].iloc[0] +
        input_df["Leadership_Positions"].iloc[0]
    )

    research_score = (
        input_df["Research_Experience"].iloc[0] +
        input_df["Projects"].iloc[0]
    )

    if tech_score >= soft_score and tech_score >= research_score:
        return "Interdisciplinary (Tech-Oriented Profile)"
    elif research_score >= tech_score and research_score >= soft_score:
        return "Interdisciplinary (Research-Oriented Profile)"
    else:
        return "Interdisciplinary (Management / Leadership Profile)"
