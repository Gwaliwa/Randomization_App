# pages/6_back_check.py

import io
from typing import Tuple

import numpy as np
import pandas as pd
import streamlit as st


def load_file(file) -> pd.DataFrame:
    """Load CSV or Excel into a DataFrame."""
    if file is None:
        return None
    name = file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file)
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(file)
    else:
        st.error("Unsupported file type. Please upload CSV or Excel.")
        return None


st.title("Back-check consistency check")

st.markdown(
    """
Upload your **ORIGINAL** survey dataset and the corresponding **BACK-CHECK** dataset.  
This tool will merge them on respondent ID (and optionally enumerator) and compare selected questions.
"""
)

# --- File uploads ---
col1, col2 = st.columns(2)

with col1:
    orig_file = st.file_uploader(
        "Upload ORIGINAL dataset (CSV or Excel)",
        type=["csv", "xlsx", "xls"],
        key="orig_file",
    )

with col2:
    bc_file = st.file_uploader(
        "Upload BACK-CHECK dataset (CSV or Excel)",
        type=["csv", "xlsx", "xls"],
        key="bc_file",
    )

if not orig_file or not bc_file:
    st.info("Please upload both ORIGINAL and BACK-CHECK datasets to continue.")
    st.stop()

# --- Read data ---
df_orig = load_file(orig_file)
df_bc = load_file(bc_file)

if df_orig is None or df_bc is None:
    st.stop()

st.subheader("Data preview")

c1, c2 = st.columns(2)
with c1:
    st.markdown("**Original data preview**")
    st.dataframe(df_orig.head())
with c2:
    st.markdown("**Back-check data preview**")
    st.dataframe(df_bc.head())

# --- Column selection ---

st.markdown("---")
st.subheader("ID and Enumerator mapping")

orig_cols = list(df_orig.columns)
bc_cols = list(df_bc.columns)

# ID columns (required)
id_col_orig = st.selectbox(
    "ID column in ORIGINAL data",
    orig_cols,
    key="id_col_orig",
    help="Unique identifier for each respondent in the ORIGINAL dataset.",
)

id_col_bc = st.selectbox(
    "ID column in BACK-CHECK data",
    bc_cols,
    key="id_col_bc",
    help="Unique identifier for each respondent in the BACK-CHECK dataset.",
)

# Enumerator columns (OPTIONAL – allow None)
enum_options_orig = ["(None)"] + orig_cols
enum_options_bc = ["(None)"] + bc_cols

enum_col_orig = st.selectbox(
    "Enumerator column in ORIGINAL data (optional)",
    enum_options_orig,
    key="enum_col_orig",
    help="Column indicating the enumerator/interviewer in the ORIGINAL dataset. "
         "Choose '(None)' if not applicable.",
)

enum_col_bc = st.selectbox(
    "Enumerator column in BACK-CHECK data (optional)",
    enum_options_bc,
    key="enum_col_bc",
    help="Column indicating the enumerator/interviewer in the BACK-CHECK dataset. "
         "Choose '(None)' if not applicable.",
)

# If user picks the same column for ID and Enumerator, silently treat enumerator as None
if enum_col_orig == id_col_orig:
    st.info(
        "You selected the same column for **ID** and **Enumerator** in the ORIGINAL data. "
        "Enumerator will be ignored to avoid duplicate merge keys."
    )
    enum_col_orig = "(None)"

if enum_col_bc == id_col_bc:
    st.info(
        "You selected the same column for **ID** and **Enumerator** in the BACK-CHECK data. "
        "Enumerator will be ignored to avoid duplicate merge keys."
    )
    enum_col_bc = "(None)"

# --- Build keys for merge, ensuring uniqueness ---

keys_orig = [id_col_orig]
keys_bc = [id_col_bc]

if enum_col_orig != "(None)":
    keys_orig.append(enum_col_orig)
if enum_col_bc != "(None)":
    keys_bc.append(enum_col_bc)

# Remove duplicates while preserving order (extra safety)
keys_orig = list(dict.fromkeys(keys_orig))
keys_bc = list(dict.fromkeys(keys_bc))

st.markdown("---")
st.subheader("Variables to compare (questions)")

# Variables to compare: must exist in BOTH datasets
common_cols = sorted(set(df_orig.columns).intersection(df_bc.columns))

# Remove ID/enumerator keys from default suggestions
default_exclusions = set(keys_orig + keys_bc)
question_candidates = [c for c in common_cols if c not in default_exclusions]

question_vars = st.multiselect(
    "Select variables/questions to compare",
    options=question_candidates,
    default=question_candidates[:3] if len(question_candidates) >= 3 else question_candidates,
    help="These variables will be compared between ORIGINAL and BACK-CHECK data.",
)

if not question_vars:
    st.warning("Please select at least one variable to compare.")
    st.stop()

# --- Perform merge safely ---

st.markdown("---")
st.subheader("Merge and comparison")

try:
    # Select columns to keep
    cols_orig = list(dict.fromkeys(keys_orig + question_vars))
    cols_bc = list(dict.fromkeys(keys_bc + question_vars))

    df_orig_sub = df_orig[cols_orig].copy()
    df_bc_sub = df_bc[cols_bc].copy()

    # Merge on keys (use left_on/right_on to avoid 'label not unique' issues)
    merged = df_orig_sub.merge(
        df_bc_sub,
        left_on=keys_orig,
        right_on=keys_bc,
        how="inner",
        suffixes=("_orig", "_bc"),
    )

    if merged.empty:
        st.warning(
            "No matching records were found between ORIGINAL and BACK-CHECK datasets "
            "using the selected ID (and enumerator) columns."
        )
        st.stop()

    st.success(f"Merged {len(merged)} matched records between ORIGINAL and BACK-CHECK data.")

    st.markdown("### Matched data preview")
    st.dataframe(merged.head())

    # --- Basic comparison: means & differences for numeric variables ---

    st.markdown("### Summary comparison for selected variables")

    summary_rows = []
    for q in question_vars:
        col_orig = f"{q}_orig"
        col_bc = f"{q}_bc"

        if col_orig not in merged.columns or col_bc not in merged.columns:
            continue

        # Restrict to rows where at least one is non-missing
        mask = merged[[col_orig, col_bc]].notna().any(axis=1)
        sub = merged.loc[mask, [col_orig, col_bc]]

        # If numeric, compute mean difference
        if pd.api.types.is_numeric_dtype(sub[col_orig]) and pd.api.types.is_numeric_dtype(
            sub[col_bc]
        ):
            mean_orig = sub[col_orig].mean()
            mean_bc = sub[col_bc].mean()
            diff = mean_bc - mean_orig
            abs_diff = abs(diff)

            # Proportion exact match
            exact_match = (sub[col_orig] == sub[col_bc]).mean()

            summary_rows.append(
                {
                    "Variable": q,
                    "Mean (ORIGINAL)": mean_orig,
                    "Mean (BACK-CHECK)": mean_bc,
                    "Difference (BC - Orig)": diff,
                    "Abs difference": abs_diff,
                    "Exact match rate": exact_match,
                    "N compared": len(sub),
                }
            )
        else:
            # For non-numeric, only report match rate
            exact_match = (sub[col_orig] == sub[col_bc]).mean()
            summary_rows.append(
                {
                    "Variable": q,
                    "Mean (ORIGINAL)": np.nan,
                    "Mean (BACK-CHECK)": np.nan,
                    "Difference (BC - Orig)": np.nan,
                    "Abs difference": np.nan,
                    "Exact match rate": exact_match,
                    "N compared": len(sub),
                }
            )

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        st.dataframe(summary_df)
    else:
        st.info(
            "No numeric variables were selected or available for mean comparison. "
            "You may still inspect the merged data above."
        )

except Exception as e:
    st.error(
        "An error occurred while merging or comparing the datasets. "
        "Please check that your ID and Enumerator columns are correctly specified "
        "and that the selected variables exist in both files."
    )
    st.exception(e)
