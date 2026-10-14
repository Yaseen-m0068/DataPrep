
import streamlit as st
import pandas as pd
import hashlib
from io import BytesIO
from src.ingestion.loader import load_dataset
from src.profiling.profile import profile_dataset
from src.transformations.missing_values import handle_missing_values
from src.transformations.duplicates import remove_duplicate_rows
from src.transformations.categories import standardize_categories

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
        # Initialize or reset state when the uploaded file changes.
        uploaded_bytes = uploaded_file.getvalue()
        upload_fingerprint = hashlib.sha256(uploaded_bytes).hexdigest()

        if (
            st.session_state.get("upload_fingerprint")
            != upload_fingerprint
        ):
            original_df = load_dataset(uploaded_file)

            st.session_state["upload_fingerprint"] = upload_fingerprint
            st.session_state["original_data"] = original_df.copy(deep=True)
            st.session_state["cleaned_data"] = original_df.copy(deep=True)
            st.session_state.pop("preprocessing_preview", None)

        original_df = st.session_state["original_data"]
        df = st.session_state["cleaned_data"]

        # Profile the current working dataset.
        profile = profile_dataset(df)

        st.success("Dataset uploaded successfully!")

        # Dataset summary.
        col1, col2, col3 = st.columns(3)

        col1.metric("Rows", f"{len(df):,}")
        col2.metric("Columns", f"{df.shape[1]:,}")
        col3.metric(
            "Missing Values",
            f"{int(df.isna().sum().sum()):,}",
        )

        st.subheader("Current Working Dataset")
        st.dataframe(df.head(10), use_container_width=True)

        # Column information.
        st.subheader("Column Information")

        column_info = df.dtypes.astype(str).rename("Data Type").to_frame()
        column_info["Missing Values"] = df.isna().sum()
        column_info["Unique Values"] = df.nunique()

        st.dataframe(column_info, use_container_width=True)

        # Data quality report.
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

        # Recommended actions.
        st.subheader("Recommended Actions")

        recommendations = profile["recommendations"]

        if not recommendations:
            st.info("No recommendations are available.")
        else:
            for recommendation in recommendations:
                column = (
                    recommendation.get("column") or "Entire dataset"
                )

                with st.expander(
                    f"{recommendation['title']} — {column}"
                ):
                    st.write(recommendation["explanation"])

                    for action in recommendation["actions"]:
                        st.markdown(f"- {action}")

        # Missing-value preprocessing.
        st.subheader("Missing-Value Preprocessing")

        missing_columns = [
            column
            for column in df.columns
            if df[column].isna().any()
        ]

        if not missing_columns:
            st.success("No missing values are available to clean.")
        else:
            selected_column = st.selectbox(
                "Choose a column",
                missing_columns,
                key="missing_column",
            )

            if pd.api.types.is_numeric_dtype(df[selected_column].dtype):
                strategies = [
                    "drop_rows",
                    "fill_mean",
                    "fill_median",
                    "fill_mode",
                ]
            else:
                strategies = ["drop_rows", "fill_mode"]

            strategy_labels = {
                "drop_rows": "Remove rows with missing values",
                "fill_mean": "Fill with mean",
                "fill_median": "Fill with median",
                "fill_mode": "Fill with mode",
            }

            selected_strategy = st.selectbox(
                "Choose a strategy",
                strategies,
                format_func=lambda value: strategy_labels[value],
                key="missing_strategy",
            )

            explanations = {
                "fill_mean": (
                    "Mean imputation can be sensitive to extreme values."
                ),
                "fill_median": (
                    "Median imputation is generally less sensitive "
                    "to extreme values."
                ),
                "fill_mode": (
                    "Mode imputation fills missing values with "
                    "the most frequent observed value."
                ),
                "drop_rows": (
                    "This removes rows where the selected column "
                    "has missing values."
                ),
            }

            st.caption(explanations[selected_strategy])

            if st.button("Preview transformation"):
                try:
                    preview_df = handle_missing_values(
                        df,
                        selected_column,
                        selected_strategy,
                    )

                    st.session_state["preprocessing_preview"] = {
                        "data": preview_df,
                        "column": selected_column,
                        "strategy": selected_strategy,
                    }
                except (TypeError, ValueError) as exc:
                    st.error(str(exc))

            preview = st.session_state.get("preprocessing_preview")

            if (
                preview is not None
                and preview["column"] == selected_column
                and preview["strategy"] == selected_strategy
            ):
                preview_df = preview["data"]

                st.write("Preview of transformed data")
                st.dataframe(
                    preview_df.head(10),
                    use_container_width=True,
                )

                col1, col2, col3 = st.columns(3)

                col1.metric("Current rows", len(df))
                col2.metric("Preview rows", len(preview_df))
                col3.metric(
                    "Remaining missing values",
                    int(preview_df[selected_column].isna().sum()),
                )

                if st.button("Apply transformation"):
                    st.session_state["cleaned_data"] = (
                        preview_df.copy(deep=True)
                    )
                    st.session_state.pop("preprocessing_preview", None)
                    st.rerun()

        # Duplicate-row preprocessing.
        st.subheader("Duplicate-Row Preprocessing")

        duplicate_count = int(df.duplicated().sum())

        if duplicate_count == 0:
            st.success("No exact duplicate rows found.")
            st.session_state.pop("duplicate_preview", None)
        else:
            st.info(
                f"Found {duplicate_count} exact duplicate row(s). "
                "The first occurrence of each row will be preserved."
            )

            if st.button("Preview duplicate removal"):
                preview_df = remove_duplicate_rows(df)

                st.session_state["duplicate_preview"] = {
                    "data": preview_df,
                    "source_rows": len(df),
                }

            duplicate_preview = st.session_state.get("duplicate_preview")

            if (
                duplicate_preview is not None
                and duplicate_preview["source_rows"] == len(df)
            ):
                preview_df = duplicate_preview["data"]

                st.write("Preview after removing duplicate rows")
                st.dataframe(
                    preview_df.head(10),
                    use_container_width=True,
                )

                preview_col1, preview_col2 = st.columns(2)
                preview_col1.metric("Current Rows", len(df))
                preview_col2.metric(
                    "Rows After Removal",
                    len(preview_df),
                    delta=len(preview_df) - len(df),
                )

                if st.button("Apply duplicate removal"):
                    st.session_state["cleaned_data"] = (
                        preview_df.copy(deep=True)
                    )
                    st.session_state.pop("duplicate_preview", None)
                    st.session_state.pop("preprocessing_preview", None)
                    st.rerun()


        # Inconsistent-category preprocessing.
        st.subheader("Category Standardization")

        text_columns = [
            column
            for column in df.columns
            if (
                pd.api.types.is_object_dtype(df[column].dtype)
                or pd.api.types.is_string_dtype(df[column].dtype)
                or isinstance(df[column].dtype, pd.CategoricalDtype)
            )
        ]

        if not text_columns:
            st.info("No text or categorical columns are available.")
            st.session_state.pop("category_preview", None)
        else:
            category_column = st.selectbox(
                "Choose a column to standardize",
                text_columns,
                key="category_column",
            )

            strip_whitespace = st.checkbox(
                "Remove leading and trailing whitespace",
                value=True,
                key="category_strip_whitespace",
            )

            case_strategy = st.selectbox(
                "Choose capitalization",
                options=["preserve", "lower", "upper", "title"],
                format_func=lambda value: {
                    "preserve": "Preserve existing capitalization",
                    "lower": "Convert to lowercase",
                    "upper": "Convert to uppercase",
                    "title": "Convert to title case",
                }[value],
                key="category_case_strategy",
            )

            if st.button("Preview category standardization"):
                try:
                    preview_df = standardize_categories(
                        df,
                        category_column,
                        strip_whitespace=strip_whitespace,
                        case_strategy=case_strategy,
                    )

                    st.session_state["category_preview"] = {
                        "data": preview_df,
                        "column": category_column,
                        "strip_whitespace": strip_whitespace,
                        "case_strategy": case_strategy,
                    }
                except (TypeError, ValueError) as exc:
                    st.error(str(exc))

            category_preview = st.session_state.get("category_preview")

            if (
                category_preview is not None
                and category_preview["column"] == category_column
                and category_preview["strip_whitespace"]
                == strip_whitespace
                and category_preview["case_strategy"] == case_strategy
            ):
                preview_df = category_preview["data"]

                st.write("Preview of standardized categories")
                st.dataframe(
                    preview_df[[category_column]].head(10),
                    use_container_width=True,
                )

                before_values = int(df[category_column].nunique(dropna=True))
                after_values = int(
                    preview_df[category_column].nunique(dropna=True)
                )

                metric1, metric2 = st.columns(2)
                metric1.metric("Distinct values before", before_values)
                metric2.metric("Distinct values after", after_values)

                if st.button("Apply category standardization"):
                    st.session_state["cleaned_data"] = (
                        preview_df.copy(deep=True)
                    )
                    st.session_state.pop("category_preview", None)
                    st.session_state.pop("preprocessing_preview", None)
                    st.session_state.pop("duplicate_preview", None)
                    st.rerun()

        # Original versus current working dataset.
        st.subheader("Original vs Cleaned Dataset")

        original_rows = len(original_df)
        cleaned_rows = len(df)

        original_missing = int(original_df.isna().sum().sum())
        cleaned_missing = int(df.isna().sum().sum())

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Original Rows", original_rows)
        col2.metric(
            "Current Rows",
            cleaned_rows,
            delta=cleaned_rows - original_rows,
        )
        col3.metric("Original Missing Values", original_missing)
        col4.metric(
            "Current Missing Values",
            cleaned_missing,
            delta=cleaned_missing - original_missing,
            delta_color="inverse",
        )

        st.write("Current working dataset")
        st.dataframe(df.head(10), use_container_width=True)

        # Export the current cleaned dataset.
        st.subheader("Export Cleaned Dataset")

        if df.empty:
            st.warning(
                "The current dataset has no rows to export. "
                "Reset the dataset or review your transformations."
            )
        else:
            csv_data = df.to_csv(index=False).encode("utf-8")

            excel_buffer = BytesIO()

            with pd.ExcelWriter(
                excel_buffer,
                engine="openpyxl",
            ) as writer:
                df.to_excel(
                    writer,
                    index=False,
                    sheet_name="Cleaned Data",
                )

            excel_data = excel_buffer.getvalue()

            export_col1, export_col2 = st.columns(2)

            with export_col1:
                st.download_button(
                    label="Download as CSV",
                    data=csv_data,
                    file_name="cleaned_dataset.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            with export_col2:
                st.download_button(
                    label="Download as Excel",
                    data=excel_data,
                    file_name="cleaned_dataset.xlsx",
                    mime=(
                        "application/vnd.openxmlformats-officedocument."
                        "spreadsheetml.sheet"
                    ),
                    use_container_width=True,
                )

        # Reset the working dataset.
        if st.button("Reset to Original Dataset"):
            st.session_state["cleaned_data"] = original_df.copy(deep=True)
            st.session_state.pop("preprocessing_preview", None)
            st.rerun()

    except (ValueError, OSError, ImportError) as exc:
        st.error(f"Could not process the dataset: {exc}")
