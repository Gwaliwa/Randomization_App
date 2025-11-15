import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

st.title("📊 Balance Checks (Baseline Comparability)")

st.markdown(
    """
This page provides **baseline balance checks** similar to `orth_out` / `ietoolkit`:

- Compare means of covariates between treatment and control  
- Compute standardized mean differences (SMD)  
- Compute t-test p-values  

**Workflow**

1. Upload a baseline dataset with a treatment indicator.  
2. Select the treatment column (0/1) and covariates.  
3. View and download the balance table.
"""
)

uploaded_file = st.file_uploader(
    "Upload baseline dataset (CSV or Excel)", type=["csv", "xlsx"], key="balance_file"
)

if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("Preview of uploaded data")
    st.dataframe(df.head())

    treat_col = st.selectbox("Treatment indicator column (0/1)", options=df.columns)
    covariate_cols = st.multiselect(
        "Covariate columns to check balance on",
        options=[c for c in df.columns if c != treat_col],
    )

    if treat_col and covariate_cols:
        # Ensure treatment is binary
        if df[treat_col].nunique() <= 1:
            st.error("Treatment column must have at least two distinct values (0 and 1).")
        else:
            treat_mask = df[treat_col] == df[treat_col].unique()[0]
            # Better: define as 1 vs 0; assume 1 = treated if present
            if 1 in df[treat_col].unique():
                treat_mask = df[treat_col] == 1
            else:
                # fallback: first unique value = treatment
                treat_mask = df[treat_col] == df[treat_col].unique()[0]

            treated = df[treat_mask]
            control = df[~treat_mask]

            rows = []
            for col in covariate_cols:
                x_t = treated[col].dropna()
                x_c = control[col].dropna()

                if x_t.empty or x_c.empty:
                    continue

                mean_t = x_t.mean()
                mean_c = x_c.mean()
                sd_t = x_t.std(ddof=1)
                sd_c = x_c.std(ddof=1)
                diff = mean_t - mean_c

                # pooled SD for SMD
                n_t = len(x_t)
                n_c = len(x_c)
                s_pooled = np.sqrt(
                    ((n_t - 1) * sd_t**2 + (n_c - 1) * sd_c**2) / (n_t + n_c - 2)
                ) if (n_t + n_c - 2) > 0 else np.nan
                smd = diff / s_pooled if s_pooled not in [0, np.nan] else np.nan

                # t-test (Welch)
                try:
                    tstat, pval = stats.ttest_ind(x_t, x_c, equal_var=False, nan_policy="omit")
                except Exception:
                    tstat, pval = np.nan, np.nan

                rows.append(
                    {
                        "variable": col,
                        "mean_treat": mean_t,
                        "mean_control": mean_c,
                        "diff": diff,
                        "std_diff": smd,
                        "p_value": pval,
                        "n_treat": n_t,
                        "n_control": n_c,
                    }
                )

            if rows:
                bal_table = pd.DataFrame(rows)
                st.subheader("Balance table")
                st.dataframe(bal_table)

                csv = bal_table.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download balance table (CSV)",
                    data=csv,
                    file_name="balance_checks.csv",
                    mime="text/csv",
                )
            else:
                st.warning("No valid covariates selected for balance checks.")
else:
    st.info("👆 Upload a dataset to get started.")
