import streamlit as st
import pandas as pd

from src.predict_top3 import predict_top_3
from src.other_interpreter import interpret_other
from src.recommend_field_and_career import recommend_fields_and_careers

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Career Path Recommendation System",
    page_icon="🎓",
    layout="centered"
)

# -------------------------------------------------
# SESSION STATE INIT
# -------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "🧾 Student Profile"

# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.markdown(
    "<h1 style='text-align:center;'>🎓 Career Path Recommendation System</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;'>AI-powered, data-driven career guidance</p>",
    unsafe_allow_html=True
)

st.divider()

# -------------------------------------------------
# RADIO NAVIGATION (TAB REPLACEMENT)
# -------------------------------------------------
page = st.radio(
    "",
    ["🧾 Student Profile", "📊 Recommendations"],
    index=0 if st.session_state.page == "🧾 Student Profile" else 1,
    horizontal=True
)

# -------------------------------------------------
# PAGE 1 — STUDENT PROFILE
# -------------------------------------------------
if page == "🧾 Student Profile":

    st.subheader("📋 Enter Student Details")

    col1, col2 = st.columns(2)

    with col1:
        gpa = st.slider("GPA", 0.0, 10.0, 7.5)
        field_courses = st.slider("Field Specific Courses", 0, 10, 5)
        internships = st.slider("Internships", 0, 5, 1)
        projects = st.slider("Projects", 0, 10, 3)
        research = st.slider("Research Experience", 0, 5, 1)
        certifications = st.slider("Industry Certifications", 0, 5, 1)
        presentation = st.slider("Presentation Skills", 0, 5, 3)
        networking = st.slider("Networking Skills", 0, 5, 2)
    with col2:
        extracurricular = st.slider("Extracurricular Activities", 0, 10, 3)
        leadership = st.slider("Leadership Positions", 0, 5, 1)
        coding = st.slider("Coding Skills", 0, 5, 3)
        communication = st.slider("Communication Skills", 0, 5, 3)
        problem_solving = st.slider("Problem Solving Skills", 0, 5, 3)
        analytical = st.slider("Analytical Skills", 0, 5, 3)
        teamwork = st.slider("Teamwork Skills", 0, 5, 3)
        

    # Buttons
    col_btn1, col_btn2 = st.columns(2)

    with col_btn1:
        if st.button("🚀 Get Career Recommendation"):
            st.session_state.input_df = pd.DataFrame([{
                "GPA": gpa,
                "Field_Specific_Courses": field_courses,
                "Internships": internships,
                "Projects": projects,
                "Research_Experience": research,
                "Industry_Certifications": certifications,
                "Extracurricular_Activities": extracurricular,
                "Leadership_Positions": leadership,
                "Coding_Skills": coding,
                "Communication_Skills": communication,
                "Problem_Solving_Skills": problem_solving,
                "Analytical_Skills": analytical,
                "Teamwork_Skills": teamwork,
                "Presentation_Skills": presentation,
                "Networking_Skills": networking
            }])

            st.session_state.page = "📊 Recommendations"
            st.rerun()

    with col_btn2:
        if st.button("🔄 Reset"):
            st.session_state.clear()
            st.rerun()

# -------------------------------------------------
# PAGE 2 — RECOMMENDATIONS
# -------------------------------------------------
if page == "📊 Recommendations":

    if "input_df" not in st.session_state:
        st.warning("⚠️ Please fill the student profile first.")
    else:
        input_df = st.session_state.input_df

        results = predict_top_3(input_df)

        st.subheader("📊 Top Career Recommendations")

        for rank, (category, confidence) in enumerate(results, start=1):

            display_category = category
            if category == "Other":
                display_category = interpret_other(input_df)

            st.markdown(f"### {rank}. {display_category}")
            st.progress(int(confidence))
            st.write(f"**Confidence:** {confidence}%")

            # 🔥 FIELD + CAREER FOR EACH CATEGORY
            fields, careers = recommend_fields_and_careers(
                input_df, category
            )

            if fields:
                st.write("**Recommended Fields:**")
                for f in fields:
                    st.write(f"- {f}")

            if careers:
                st.write("**Suggested Careers:**")
                for c in careers:
                    st.write(f"- {c}")

            st.divider()

        if st.button("⬅️ Back to Profile"):
            st.session_state.page = "🧾 Student Profile"
            st.rerun()
