import numpy as np
import pandas as pd
import streamlit as st

st.title("📈 High-Frequency Checks (ipacheck-style)")

st.markdown(
    """
This page implements **high-frequency checks** inspired by `ipacheck`:

- Basic completeness / missingness  
- Enumerator-level stats (number of interviews, average duration, missingness)  
- Date patterns (interviews per day)  

**Workflow**

1. Upload the latest survey dataset.  
2. Choose enumerator and date/time columns (and optional duration).  
3. Review high-frequency indicators.
"""
)

uploaded_file = st.file_uploader(
    "Upload survey dataset (CSV or Excel)", type=["csv", "xlsx"], key="hf_file"
)

if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("Preview of uploaded data")
    st.dataframe(df.head())

    enum_col = st.selectbox("Enumerator column", options=df.columns)
    date_col = st.selectbox("Interview date column", options=df.columns)
    duration_col = st.selectbox(
        "Interview duration column (optional)",
        options=["<none>"] + list(df.columns),
    )
    if duration_col == "<none>":
        duration_col = None

    # Try to parse date column
    try:
        df["_hf_date"] = pd.to_datetime(df[date_col], errors="coerce").dt.date
    except Exception:
        df["_hf_date"] = pd.NaT

    st.subheader("Overall missingness")
    missing_summary = df.isna().mean().reset_index()
    missing_summary.columns = ["variable", "share_missing"]
    st.dataframe(missing_summary)

    st.subheader("Enumerator-level summary")
    enum_groups = df.groupby(enum_col)
    rows = []
    for enum, sub in enum_groups:
        row = {
            "enumerator": enum,
            "n_interviews": len(sub),
        }
        if duration_col is not None and duration_col in sub.columns:
            # duration might be in seconds/minutes
            try:
                row["avg_duration"] = sub[duration_col].astype(float).mean()
            except Exception:
                row["avg_duration"] = np.nan
        # overall missingness rate across variables
        row["avg_missing_rate"] = sub.isna().mean().mean()
        rows.append(row)
    enum_table = pd.DataFrame(rows)
    st.dataframe(enum_table)

    st.subheader("Interviews per day")
    if df["_hf_date"].notna().any():
        daily_counts = (
            df.groupby("_hf_date")
            .size()
            .reset_index(name="n_interviews")
            .sort_values("_hf_date")
        )
        st.dataframe(daily_counts)
    else:
        st.info("Could not parse dates from the selected date column.")

    # Downloads
    missing_csv = missing_summary.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download missingness summary (CSV)",
        data=missing_csv,
        file_name="hf_missingness_summary.csv",
        mime="text/csv",
    )

    enum_csv = enum_table.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download enumerator summary (CSV)",
        data=enum_csv,
        file_name="hf_enumerator_summary.csv",
        mime="text/csv",
    )

else:
    st.info("👆 Upload a survey dataset to run high-frequency checks.")
