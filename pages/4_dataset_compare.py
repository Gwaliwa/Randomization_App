import pandas as pd
import streamlit as st

st.title("📁 Dataset Comparison (cfout-style)")

st.markdown(
    """
This page compares **two datasets** similar to Stata's `cfout`:

- Compare number of rows and columns  
- Identify variables present in one dataset but not the other  
- For common numeric variables, compare means  

**Workflow**

1. Upload **Dataset A** and **Dataset B**.  
2. (Optional) Select an ID column.  
3. View comparison summaries.
"""
)

file_a = st.file_uploader(
    "Upload Dataset A (CSV or Excel)", type=["csv", "xlsx"], key="ds_a"
)
file_b = st.file_uploader(
    "Upload Dataset B (CSV or Excel)", type=["csv", "xlsx"], key="ds_b"
)

if file_a is not None and file_b is not None:
    if file_a.name.endswith(".csv"):
        df_a = pd.read_csv(file_a)
    else:
        df_a = pd.read_excel(file_a)

    if file_b.name.endswith(".csv"):
        df_b = pd.read_csv(file_b)
    else:
        df_b = pd.read_excel(file_b)

    st.subheader("Basic info")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Dataset A**")
        st.write(f"Rows: {len(df_a)}")
        st.write(f"Columns: {len(df_a.columns)}")
    with col2:
        st.markdown("**Dataset B**")
        st.write(f"Rows: {len(df_b)}")
        st.write(f"Columns: {len(df_b.columns)}")

    st.subheader("Variable comparison")
    vars_a = set(df_a.columns)
    vars_b = set(df_b.columns)

    only_a = sorted(list(vars_a - vars_b))
    only_b = sorted(list(vars_b - vars_a))
    common = sorted(list(vars_a & vars_b))

    st.markdown("**Variables only in Dataset A**")
    st.write(only_a if only_a else "None")

    st.markdown("**Variables only in Dataset B**")
    st.write(only_b if only_b else "None")

    st.markdown("**Common variables**")
    st.write(common if common else "None")

    # Numeric comparison for common vars
    numeric_common = [
        c for c in common
        if pd.api.types.is_numeric_dtype(df_a[c]) and pd.api.types.is_numeric_dtype(df_b[c])
    ]

    if numeric_common:
        rows = []
        for col in numeric_common:
            rows.append(
                {
                    "variable": col,
                    "mean_A": df_a[col].mean(),
                    "mean_B": df_b[col].mean(),
                    "diff_mean": df_a[col].mean() - df_b[col].mean(),
                }
            )
        comp_table = pd.DataFrame(rows)
        st.subheader("Numeric variable mean comparison")
        st.dataframe(comp_table)

        csv = comp_table.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download numeric comparison (CSV)",
            data=csv,
            file_name="dataset_comparison_numeric.csv",
            mime="text/csv",
        )
    else:
        st.info("No common numeric variables to compare.")
else:
    st.info("👆 Upload both datasets to compare.")
