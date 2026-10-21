
import streamlit as st
import pandas as pd
import hashlib
from io import BytesIO
from src.ingestion.loader import load_dataset
from src.profiling.profile import profile_dataset
from src.transformations.missing_values import handle_missing_values
from src.transformations.duplicates import remove_duplicate_rows
from src.transformations.categories import standardize_categories
from src.transformations.outliers import handle_numeric_outliers
from src.transformations.scaling import scale_numeric_columns
from src.detection.column_types import infer_semantic_type
from src.pipeline.history import record_transformation
from src.pipeline.undo_redo import (
    undo_last_transformation,
    redo_last_transformation,
)

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
            st.session_state["transformation_history"] = []
            st.session_state["transformation_snapshots"] = []
            st.session_state["redo_history"] = []
            st.session_state["redo_snapshots"] = []
            for preview_key in (
                "preprocessing_preview",
                "duplicate_preview",
                "category_preview",
                "outlier_preview",
                "scaling_preview",
            ):
                st.session_state.pop(preview_key, None)

        st.session_state.setdefault("transformation_history", [])
        st.session_state.setdefault("transformation_snapshots", [])
        st.session_state.setdefault("redo_history", [])
        st.session_state.setdefault("redo_snapshots", [])
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
        column_info["Unique Values"] = df.nunique(dropna=True)

        st.dataframe(column_info, use_container_width=True)

        # Data quality report: group repeated issues by category.
        st.subheader("Data Quality Report")

        quality_issues = profile["quality_issues"]
        missing_issues = [
            issue for issue in quality_issues
            if issue["issue_type"] == "missing_values"
        ]
        outlier_issues = [
            issue for issue in quality_issues
            if issue["issue_type"] == "numeric_outliers"
        ]
        grouped_issue_types = {"missing_values", "numeric_outliers"}
        other_issues = [
            issue for issue in quality_issues
            if issue["issue_type"] not in grouped_issue_types
        ]

        if not quality_issues:
            st.success("No quality issues were detected.")
        else:
            st.write(f"Detected {len(quality_issues)} quality issue(s).")

            if missing_issues:
                st.markdown("#### Missing Values")
                missing_table = pd.DataFrame([
                    {
                        "Column": issue["column"],
                        "Missing Values": issue["count"],
                        "Missing (%)": issue["percentage"],
                        "Severity": issue.get("severity", "low").title(),
                    }
                    for issue in missing_issues
                ])
                st.dataframe(missing_table, use_container_width=True)
                st.caption(
                    f"{len(missing_issues)} column(s) affected · "
                    f"{sum(issue['count'] for issue in missing_issues):,} "
                    "missing cell(s) in total."
                )

            if outlier_issues:
                st.markdown("#### Numeric Outliers")
                outlier_table = pd.DataFrame([
                    {
                        "Column": issue["column"],
                        "Outlier Count": issue["count"],
                        "Outliers (%)": issue["percentage"],
                        "Lower Bound": issue.get("lower_bound"),
                        "Upper Bound": issue.get("upper_bound"),
                        "Example Values": ", ".join(
                            repr(value)
                            for value in issue.get("outlier_values", [])[:5]
                        ),
                        "Method": issue.get("method", "IQR"),
                    }
                    for issue in outlier_issues
                ])
                st.dataframe(outlier_table, use_container_width=True)
                st.caption(
                    f"{len(outlier_issues)} numeric column(s) contain "
                    "potential outliers. Example Values shows up to five "
                    "flagged values per column."
                )

            # Keep other existing issue categories visible.
            if other_issues:
                st.markdown("#### Other Issues")
                for issue in other_issues:
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
                        if "variants" in issue:
                            st.write("Category variants:")
                            st.json(issue["variants"])
                        if "constant_value" in issue:
                            st.write(
                                f"Constant value: {issue['constant_value']}"
                            )

        # Show one recommendation per relevant issue category.
        # This section is hidden when neither missing values nor outliers exist.
        relevant_issue_types = {"missing_values", "numeric_outliers"}
        relevant_issues = [
            issue for issue in quality_issues
            if issue["issue_type"] in relevant_issue_types
        ]

        if relevant_issues:
            st.subheader("Recommended Actions")
            recommendations = profile["recommendations"]

            for issue_type, heading in (
                ("missing_values", "Handle Missing Values"),
                ("numeric_outliers", "Review Numeric Outliers"),
            ):
                category_issues = [
                    issue for issue in relevant_issues
                    if issue["issue_type"] == issue_type
                ]
                if not category_issues:
                    continue

                category_recommendations = [
                    recommendation for recommendation in recommendations
                    if recommendation["issue_type"] == issue_type
                ]
                if not category_recommendations:
                    continue

                recommendation = category_recommendations[0]
                affected_columns = [
                    issue["column"] for issue in category_issues
                    if issue.get("column") is not None
                ]

                with st.expander(heading, expanded=True):
                    st.write(recommendation["explanation"])
                    st.write(
                        "**Affected columns:** "
                        + ", ".join(affected_columns)
                    )
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
                    st.session_state["transformation_snapshots"].append(
                        st.session_state["cleaned_data"].copy(deep=True)
                    )
                    st.session_state["redo_history"] = []
                    st.session_state["redo_snapshots"] = []
                    st.session_state["transformation_history"] = (
                        record_transformation(
                            st.session_state["transformation_history"],
                            "handle_missing_values",
                            {
                                "column": selected_column,
                                "strategy": selected_strategy,
                            },
                            rows_before=len(df),
                            rows_after=len(preview_df),
                        )
                    )
                    st.session_state["cleaned_data"] = (
                        preview_df.copy(deep=True)
                    )
                    for preview_key in (
                        "duplicate_preview",
                        "category_preview",
                        "outlier_preview",
                        "scaling_preview",
                    ):
                        st.session_state.pop(preview_key, None)
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
                    st.session_state["transformation_snapshots"].append(
                        st.session_state["cleaned_data"].copy(deep=True)
                    )
                    st.session_state["redo_history"] = []
                    st.session_state["redo_snapshots"] = []
                    st.session_state["transformation_history"] = (
                        record_transformation(
                            st.session_state["transformation_history"],
                            "remove_duplicate_rows",
                            {},
                            rows_before=len(df),
                            rows_after=len(preview_df),
                        )
                    )
                    st.session_state["cleaned_data"] = (
                        preview_df.copy(deep=True)
                    )
                    for preview_key in (
                        "preprocessing_preview",
                        "category_preview",
                        "outlier_preview",
                        "scaling_preview",
                    ):
                        st.session_state.pop(preview_key, None)
                    st.session_state.pop("duplicate_preview", None)
                    st.rerun()


        # Inconsistent-category preprocessing.
        st.subheader("Category Standardization")

        # Offer only categorical text columns, excluding date-like values,
        # identifiers, free text, and columns already standardized in the
        # current transformation history. Undo/redo updates this list because
        # history changes together with the dataset snapshots.
        categorical_candidates = []
        for column in df.columns:
            is_text_or_category = (
                pd.api.types.is_object_dtype(df[column].dtype)
                or pd.api.types.is_string_dtype(df[column].dtype)
                or isinstance(df[column].dtype, pd.CategoricalDtype)
            )
            semantic_type = profile["columns"][str(column)]["semantic_type"]
            if is_text_or_category and semantic_type in {
                "categorical", "boolean_candidate"
            }:
                categorical_candidates.append(column)

        standardized_columns = {
            record.get("parameters", {}).get("column")
            for record in st.session_state["transformation_history"]
            if record.get("operation") == "standardize_categories"
        }
        text_columns = [
            column for column in categorical_candidates
            if column not in standardized_columns
        ]

        # Clear stale widget state when an applied transformation removes
        # its column from the options. Undo restores it; redo removes it again.
        if st.session_state.get("category_column") not in text_columns:
            st.session_state.pop("category_column", None)

        if not text_columns:
            st.session_state.pop("category_preview", None)
            if categorical_candidates:
                st.success("All categorical columns are standardized.")
            else:
                st.info("No categorical columns are available for standardization.")
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
                    st.session_state["transformation_snapshots"].append(
                        st.session_state["cleaned_data"].copy(deep=True)
                    )
                    st.session_state["redo_history"] = []
                    st.session_state["redo_snapshots"] = []
                    st.session_state["transformation_history"] = (
                        record_transformation(
                            st.session_state["transformation_history"],
                            "standardize_categories",
                            {
                                "column": category_column,
                                "strip_whitespace": strip_whitespace,
                                "case_strategy": case_strategy,
                            },
                            rows_before=len(df),
                            rows_after=len(preview_df),
                        )
                    )
                    st.session_state["cleaned_data"] = (
                        preview_df.copy(deep=True)
                    )
                    for preview_key in (
                        "preprocessing_preview",
                        "duplicate_preview",
                        "outlier_preview",
                        "scaling_preview",
                    ):
                        st.session_state.pop(preview_key, None)
                    st.session_state.pop("category_preview", None)
                    st.rerun()

        # Numeric outlier preprocessing.
        st.subheader("Numeric Outlier Handling")

        # Only show numeric columns currently flagged by the IQR outlier
        # detector. Re-profiling on each rerun removes a column from the list
        # after its outliers have been capped or its rows removed.
        detected_outlier_names = {
            issue["column"]
            for issue in quality_issues
            if issue["issue_type"] == "numeric_outliers"
        }
        numeric_columns = [
            column
            for column in df.select_dtypes(include="number").columns
            if (not pd.api.types.is_bool_dtype(df[column].dtype)
                and df[column].notna().any()
                and df[column].nunique(dropna=True) > 4
                and str(column) in detected_outlier_names
            )
        ]

        if st.session_state.get("outlier_column") not in numeric_columns:
            st.session_state.pop("outlier_column", None)

        if not numeric_columns:
            st.session_state.pop("outlier_preview", None)
            st.success("No outliers found.")
        else:
            outlier_column = st.selectbox(
                "Select numerical column with detected outliers",
                options=numeric_columns,
                key="outlier_column",
            )

            outlier_strategy = st.selectbox(
                "Choose outlier-handling strategy",
                options=["cap_iqr", "drop_rows"],
                format_func=lambda strategy: {
                    "cap_iqr": "Cap outliers using IQR boundaries",
                    "drop_rows": "Remove rows containing outliers",
                }[strategy],
                key="outlier_strategy",
            )

            if st.button("Preview outlier handling"):
                try:
                    preview_df = handle_numeric_outliers(
                        df,
                        outlier_column,
                        outlier_strategy,
                    )
                    st.session_state["outlier_preview"] = {
                        "data": preview_df,
                        "column": outlier_column,
                        "strategy": outlier_strategy,
                        "source_rows": len(df),
                    }
                except (TypeError, ValueError) as exc:
                    st.error(str(exc))
                    st.session_state.pop("outlier_preview", None)

            outlier_preview = st.session_state.get("outlier_preview")

            if (
                outlier_preview is not None
                and outlier_preview["column"] == outlier_column
                and outlier_preview["strategy"] == outlier_strategy
                and outlier_preview["source_rows"] == len(df)
            ):
                preview_df = outlier_preview["data"]

                st.write("Preview of the selected column")
                st.dataframe(
                    preview_df[[outlier_column]].head(10),
                    use_container_width=True,
                )

                preview_col1, preview_col2 = st.columns(2)
                preview_col1.metric("Rows before", len(df))
                preview_col2.metric("Rows after", len(preview_df))

                if outlier_strategy == "cap_iqr":
                    changed_values = int(
                        (
                            df[outlier_column].notna()
                            & df[outlier_column].ne(preview_df[outlier_column])
                        ).sum()
                    )
                    st.write(f"Values capped: {changed_values}")
                else:
                    removed_rows = len(df) - len(preview_df)
                    st.write(f"Rows to remove: {removed_rows}")

                if st.button("Apply outlier handling"):
                    st.session_state["transformation_snapshots"].append(
                        st.session_state["cleaned_data"].copy(deep=True)
                    )
                    st.session_state["redo_history"] = []
                    st.session_state["redo_snapshots"] = []
                    st.session_state["transformation_history"] = (
                        record_transformation(
                            st.session_state["transformation_history"],
                            "handle_numeric_outliers",
                            {
                                "column": outlier_column,
                                "strategy": outlier_strategy,
                            },
                            rows_before=len(df),
                            rows_after=len(preview_df),
                        )
                    )
                    st.session_state["cleaned_data"] = preview_df.copy(deep=True)
                    for preview_key in (
                        "preprocessing_preview",
                        "duplicate_preview",
                        "category_preview",
                        "scaling_preview",
                    ):
                        st.session_state.pop(preview_key, None)
                    st.session_state.pop("outlier_preview", None)
                    st.rerun()

        # Numeric scaling workflow.
        st.subheader("Numerical Feature Scaling")
        st.caption(
            "Scale selected numerical measurements before machine learning. "
            "Scaling changes values and units; it does not remove outliers."
        )

        scaling_columns = []

        for column in df.select_dtypes(include="number").columns:
            series = df[column]

            # Exclude Boolean columns.
            if pd.api.types.is_bool_dtype(series.dtype):
                continue

            # Exclude columns containing only missing values.
            if not series.notna().any():
                continue

            # Exclude binary numeric columns.
            if series.nunique(dropna=True) <= 2:
                continue

            # Exclude columns identified as identifiers by semantic inference.
            semantic = infer_semantic_type(series, str(column))
            if semantic["semantic_type"] == "identifier":
                continue

            scaling_columns.append(column)

        if not scaling_columns:
            st.info("No numerical columns with observed values are available to scale.")
            st.session_state.pop("scaling_preview", None)
        else:
            selected_scaling_columns = st.multiselect(
                "Select numerical columns to scale",
                options=scaling_columns,
                key="scaling_columns",
                help="Choose only measurement/features you want to scale. Avoid IDs and target labels unless scaling them is intentional.",
            )
            scaling_strategy = st.selectbox(
                "Choose a scaling method",
                options=["standard", "minmax"],
                format_func=lambda value: {
                    "standard": "Standardization (mean 0, standard deviation 1)",
                    "minmax": "Min-Max Scaling (range 0 to 1)",
                }[value],
                key="scaling_strategy",
            )

            if st.button("Preview numerical scaling"):
                try:
                    preview_df = scale_numeric_columns(
                        df,
                        selected_scaling_columns,
                        strategy=scaling_strategy,
                    )
                    st.session_state["scaling_preview"] = {
                        "data": preview_df,
                        "columns": list(selected_scaling_columns),
                        "strategy": scaling_strategy,
                        "source_rows": len(df),
                    }
                except (TypeError, ValueError) as exc:
                    st.error(str(exc))
                    st.session_state.pop("scaling_preview", None)

            scaling_preview = st.session_state.get("scaling_preview")
            if (
                scaling_preview is not None
                and scaling_preview["columns"] == list(selected_scaling_columns)
                and scaling_preview["strategy"] == scaling_strategy
                and scaling_preview["source_rows"] == len(df)
            ):
                preview_df = scaling_preview["data"]
                st.write("Preview of scaled numerical columns")
                st.dataframe(
                    preview_df[selected_scaling_columns].head(10),
                    use_container_width=True,
                )
                st.caption(
                    "Missing values are preserved. Constant columns become 0. "
                    "Only the selected columns are changed."
                )

                if st.button("Apply numerical scaling"):
                    st.session_state["transformation_snapshots"].append(
                        st.session_state["cleaned_data"].copy(deep=True)
                    )
                    st.session_state["redo_history"] = []
                    st.session_state["redo_snapshots"] = []
                    st.session_state["transformation_history"] = record_transformation(
                        st.session_state["transformation_history"],
                        "scale_numeric_columns",
                        {
                            "columns": list(selected_scaling_columns),
                            "strategy": scaling_strategy,
                        },
                        rows_before=len(df),
                        rows_after=len(preview_df),
                    )
                    st.session_state["cleaned_data"] = preview_df.copy(deep=True)
                    for preview_key in (
                        "preprocessing_preview",
                        "duplicate_preview",
                        "category_preview",
                        "outlier_preview",
                        "scaling_preview",
                    ):
                        st.session_state.pop(preview_key, None)
                    st.session_state.pop("scaling_preview", None)
                    st.rerun()

        # Show applied transformations and provide undo/redo controls.
        st.subheader("Transformation History")
        transformation_history = st.session_state["transformation_history"]

        if transformation_history:
            history_df = pd.DataFrame(transformation_history)
            st.dataframe(history_df, use_container_width=True)
        else:
            st.info("No transformations applied yet.")

        st.caption(
            "Undo restores the previous dataset state. Redo reapplies the "
            "most recently undone state. Applying a new transformation "
            "after undo clears the redo history."
        )

        undo_col, redo_col = st.columns(2)
        with undo_col:
            if st.button(
                "Undo Last Transformation",
                disabled=not st.session_state["transformation_history"],
            ):
                try:
                    result = undo_last_transformation(
                        st.session_state["cleaned_data"],
                        st.session_state["transformation_history"],
                        st.session_state["transformation_snapshots"],
                        st.session_state["redo_history"],
                        st.session_state["redo_snapshots"],
                    )
                    (
                        restored_df,
                        history,
                        snapshots,
                        redo_history,
                        redo_snapshots,
                    ) = result
                    st.session_state["cleaned_data"] = restored_df
                    st.session_state["transformation_history"] = history
                    st.session_state["transformation_snapshots"] = snapshots
                    st.session_state["redo_history"] = redo_history
                    st.session_state["redo_snapshots"] = redo_snapshots
                    for preview_key in (
                        "preprocessing_preview",
                        "duplicate_preview",
                        "category_preview",
                        "outlier_preview",
                        "scaling_preview",
                    ):
                        st.session_state.pop(preview_key, None)
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))

        with redo_col:
            if st.button(
                "Redo Last Transformation",
                disabled=not st.session_state["redo_history"],
            ):
                try:
                    result = redo_last_transformation(
                        st.session_state["cleaned_data"],
                        st.session_state["transformation_history"],
                        st.session_state["transformation_snapshots"],
                        st.session_state["redo_history"],
                        st.session_state["redo_snapshots"],
                    )
                    (
                        restored_df,
                        history,
                        snapshots,
                        redo_history,
                        redo_snapshots,
                    ) = result
                    st.session_state["cleaned_data"] = restored_df
                    st.session_state["transformation_history"] = history
                    st.session_state["transformation_snapshots"] = snapshots
                    st.session_state["redo_history"] = redo_history
                    st.session_state["redo_snapshots"] = redo_snapshots
                    for preview_key in (
                        "preprocessing_preview",
                        "duplicate_preview",
                        "category_preview",
                        "outlier_preview",
                        "scaling_preview",
                    ):
                        st.session_state.pop(preview_key, None)
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))

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
            st.session_state["transformation_history"] = []
            st.session_state["transformation_snapshots"] = []
            st.session_state["redo_history"] = []
            st.session_state["redo_snapshots"] = []
            for preview_key in (
                "preprocessing_preview",
                "duplicate_preview",
                "category_preview",
                "outlier_preview",
                "scaling_preview",
            ):
                st.session_state.pop(preview_key, None)
            st.rerun()

    except (ValueError, OSError, ImportError) as exc:
        st.error(f"Could not process the dataset: {exc}")