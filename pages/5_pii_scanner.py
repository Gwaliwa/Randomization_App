import re
import pandas as pd
import streamlit as st

st.title("🔐 PII Scanner")

st.markdown(
    """
This page scans a dataset for **potential PII (Personally Identifiable Information)**:

- Emails  
- Phone numbers  
- ID-like patterns  
- GPS coordinates  
- Name-like columns (by column name heuristics)  

**Workflow**

1. Upload a dataset.  
2. The app inspects columns and flags potential PII.  
3. Review and decide what to anonymize.
"""
)

uploaded_file = st.file_uploader(
    "Upload dataset (CSV or Excel)", type=["csv", "xlsx"], key="pii_file"
)

email_pattern = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
phone_pattern = re.compile(r"\+?\d[\d\s\-]{6,}\d")
coord_pattern = re.compile(r"^-?\d{1,3}\.\d+")  # rough latitude/longitude

if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("Preview of uploaded data")
    st.dataframe(df.head())

    results = []
    sample_n = 200  # sample up to 200 rows per column for speed
    for col in df.columns:
        series = df[col].astype(str).dropna()
        sample = series.head(sample_n)

        col_lower = col.lower()
        looks_like_name = any(
            kw in col_lower
            for kw in ["name", "first_name", "last_name", "surname"]
        )
        looks_like_id = any(
            kw in col_lower
            for kw in ["id", "identifier", "national_id", "nid", "passport"]
        )

        has_email = any(bool(email_pattern.search(x)) for x in sample)
        has_phone = any(bool(phone_pattern.search(x)) for x in sample)
        has_coord = any(bool(coord_pattern.search(x)) for x in sample)

        results.append(
            {
                "column": col,
                "possible_name_column": looks_like_name,
                "possible_id_column": looks_like_id,
                "contains_email_pattern": has_email,
                "contains_phone_pattern": has_phone,
                "contains_coordinate_pattern": has_coord,
            }
        )

    pii_df = pd.DataFrame(results)
    st.subheader("PII scan results")
    st.dataframe(pii_df)

    csv = pii_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download PII scan report (CSV)",
        data=csv,
        file_name="pii_scan_report.csv",
        mime="text/csv",
    )
else:
    st.info("👆 Upload a dataset to scan for PII.")
