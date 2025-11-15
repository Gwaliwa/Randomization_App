import streamlit as st

st.set_page_config(
    page_title="RCT Randomization & QA Suite",
    page_icon="🎯",
    layout="wide",
)

st.title("🎯 RCT Randomization & Data QA Suite")

st.markdown(
    """
This app provides a **Python / Streamlit** toolkit for randomized evaluations:

**Design & Assignment**
- randomizr-style assignment (complete / blocked / cluster)
- randtreat-style multi-arm assignment (unequal fractions, misfits)

**Data Quality & Diagnostics**
- Balance checks (baseline comparability)
- Dataset comparison
- PII scanner
- Back-check analysis
- High-frequency checks

Use the sidebar to navigate between pages.
"""
)
