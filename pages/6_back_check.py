import numpy as np
import pandas as pd
import streamlit as st

st.title("🔁 Back-check Analysis (bcstats-style)")

st.markdown(
    """
This page compares **original survey data vs back-check data**, similar to `bcstats`:

- Match records by ID  
- Compute agreement rates per question  
- Summarize agreement by enumerator  

**Workflow**

1. Upload original and back-check datasets.  
2. Select the ID column (in both) and enumerator columns.  
3. View agreement per variable and per enumerator.
"""
)

file_orig = st.file_uploader(
    "Upload ORIGINAL dataset (CSV or Excel)", type=["csv", "xlsx"], key="orig_file"
)
file_bc = st.file_uploader(
    "Upload BACK-CHECK dataset (CSV or Excel)", type=["csv", "xlsx"], key="bc_file"
)

if file_orig is not None and file_bc is not None:
    if file_orig.name.endswith(".csv"):
        df_orig = pd.read_csv(file_orig)
    else:
        df_orig = pd.read_excel(file_orig)

    if file_bc.name.endswith(".csv"):
        df_bc = pd.read_csv(file_bc)
    else:
        df_bc = pd.read_excel(file_bc)

    st.subheader("Original data preview")
    st.dataframe(df_orig.head())
    st.subheader("Back-check data preview")
    st.dataframe(df_bc.head())

    id_col_orig = st.selectbox("ID column in ORIGINAL data", options=df_orig.columns, key="id_orig")
    id_col_bc = st.selectbox("ID column in BACK-CHECK data", options=df_bc.columns, key="id_bc")

    enum_col_orig = st.selectbox("Enumerator column in ORIGINAL data", options=df_orig.columns, key="enum_orig")
    enum_col_bc = st.selectbox("Enumerator column in BACK-CHECK data", options=df_bc.columns, key="enum_bc")

    # Common variables by name
    common_vars = sorted(list(set(df_orig.columns) & set(df_bc.columns)))
    # Remove ID and enumerator columns from question list
    common_vars = [
        v for v in common_vars
        if v not in [id_col_orig, id_col_bc, enum_col_orig, enum_col_bc]
    ]

    question_vars = st.multiselect(
        "Variables to compare (questions)",
        options=common_vars,
        default=common_vars,
    )

    if question_vars:
        # Merge on IDs
        merged = df_orig[[id_col_orig, enum_col_orig] + question_vars].merge(
            df_bc[[id_col_bc, enum_col_bc] + question_vars],
            left_on=id_col_orig,
            right_on=id_col_bc,
            suffixes=("_orig", "_bc"),
            how="inner",
        )

        st.write(f"Merged rows: {len(merged)}")

        # Agreement per question
        q_rows = []
        for var in question_vars:
            col_o = f"{var}_orig"
            col_b = f"{var}_bc"

            # define agreement (exact match), ignoring missing
            mask_non_missing = merged[col_o].notna() & merged[col_b].notna()
            n_comp = mask_non_missing.sum()
            if n_comp == 0:
                continue
            agree = (merged.loc[mask_non_missing, col_o] == merged.loc[mask_non_missing, col_b]).sum()
            agree_rate = agree / n_comp if n_comp > 0 else np.nan

            q_rows.append(
                {
                    "variable": var,
                    "n_compared": int(n_comp),
                    "n_agree": int(agree),
                    "agreement_rate": agree_rate,
                }
            )

        if q_rows:
            q_table = pd.DataFrame(q_rows)
            st.subheader("Agreement by question")
            st.dataframe(q_table)

        # Agreement by enumerator (using ORIGINAL enumerator)
        enum_rows = []
        for enum, sub in merged.groupby(enum_col_orig):
            row = {"enumerator": enum}
            for var in question_vars:
                col_o = f"{var}_orig"
                col_b = f"{var}_bc"
                mask_non_missing = sub[col_o].notna() & sub[col_b].notna()
                n_comp = mask_non_missing.sum()
                if n_comp == 0:
                    continue
                agree = (sub.loc[mask_non_missing, col_o] == sub.loc[mask_non_missing, col_b]).sum()
                agree_rate = agree / n_comp if n_comp > 0 else np.nan
                row[f"agree_{var}"] = agree_rate
            enum_rows.append(row)

        if enum_rows:
            enum_table = pd.DataFrame(enum_rows)
            st.subheader("Agreement by enumerator (original data)")
            st.dataframe(enum_table)

        # Downloads
        if q_rows:
            q_csv = q_table.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download question agreement table (CSV)",
                data=q_csv,
                file_name="backcheck_agreement_by_question.csv",
                mime="text/csv",
            )
        if enum_rows:
            e_csv = enum_table.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download enumerator agreement table (CSV)",
                data=e_csv,
                file_name="backcheck_agreement_by_enumerator.csv",
                mime="text/csv",
            )
    else:
        st.info("Select at least one question variable to compare.")
else:
    st.info("👆 Upload both ORIGINAL and BACK-CHECK datasets.")
