
import streamlit as st

from src.ingestion.loader import load_dataset
from src.profiling.profile import profile_dataset

st.set_page_config(
    page_title="DataPrep | Data Quality Assistant",
    page_icon="🔍",
    layout="wide",
)

st.title("DataPrep")
st.subheader("Explainable Automated Preprocessing Assistant")

st.write(
    "Upload a dataset to explore its structure and "
    "prepare it for data analysis."
)

uploaded_file = st.file_uploader(
    "Upload your dataset",
    type=["csv", "xlsx", "parquet"],
)

if uploaded_file is not None:
    try:
        df = load_dataset(uploaded_file)
        profile = profile_dataset(df)

        st.success("Dataset uploaded successfully!")

        col1, col2, col3 = st.columns(3)

        col1.metric("Rows", f"{df.shape[0]:,}")
        col2.metric("Columns", f"{df.shape[1]:,}")
        col3.metric("Missing Values", f"{int(df.isna().sum().sum()):,}")

        st.subheader("Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)

        st.subheader("Column Information")

        column_info = df.dtypes.astype(str).rename("Data Type").to_frame()
        column_info["Missing Values"] = df.isna().sum()
        column_info["Unique Values"] = df.nunique()


        st.subheader("Data Quality Report")

        quality_issues = profile["quality_issues"]

        if not quality_issues:
            st.success("No quality issues were detected.")
        else:
            st.write(f"Detected {len(quality_issues)} quality issues.")

            for issue in quality_issues:
                severity = issue.get("severity", "low").upper()
                column = issue.get("column") or "Entire dataset"

                with st.expander(
                    f"{issue['issue_type'].replace('_', ' ').title()} "
                    f"— {column} [{severity}]"
                ):
                    st.write(issue["description"])

                    if "count" in issue:
                        st.write(f"Affected count: {issue['count']}")

                    if "percentage" in issue:
                        st.write(
                            f"Affected percentage: "
                            f"{issue['percentage']}%"
                        )

        st.subheader("Recommended Actions")

        recommendations = profile["recommendations"]

        if not recommendations:
            st.info("No recommendations are available.")
        else:
            for recommendation in recommendations:
                with st.expander(
                    f"{recommendation['title']} — "
                    f"{recommendation.get('column') or 'Entire dataset'}"
                ):
                    st.write(recommendation["explanation"])

                    for action in recommendation["actions"]:
                        st.markdown(f"- {action}")

        st.dataframe(column_info, use_container_width=True)

    except (ValueError, OSError, ImportError) as exc:
        st.error(f"Could not load the dataset: {exc}")
