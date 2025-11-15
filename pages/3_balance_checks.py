import numpy as np
import pandas as pd
import streamlit as st

try:
    from scipy import stats
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

st.title("🔍 Balance Checks (Baseline Comparability)")

st.markdown(
    """
Upload a dataset **after randomization** (with a 0/1 treatment column, e.g. `treat`)
and select baseline covariates to check whether **Treatment** and **Control** groups
are similar **before** the intervention.

This page will compute, for each covariate:

- Mean in Control and Treatment  
- Difference in means  
- Standardized Mean Difference (SMD)  
- t-test p-value (if SciPy is available)  
"""
)

uploaded_file = st.file_uploader(
    "Upload dataset with treatment indicator", type=["csv", "xlsx"], key="bal_file"
)

if uploaded_file is not None:
    # --- Read file ---
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Could not read the uploaded file: {e}")
        st.stop()

    if df.empty:
        st.error("The uploaded file appears to be empty.")
        st.stop()

    st.subheader("Preview of uploaded data")
    st.dataframe(df.head())

    # --- Select treatment column ---
    treat_col = st.selectbox(
        "Treatment indicator column (0/1)",
        options=df.columns,
        index=list(df.columns).index("treat") if "treat" in df.columns else 0,
    )

    # Try to coerce treatment column to numeric 0/1
    try:
        df[treat_col] = pd.to_numeric(df[treat_col], errors="coerce")
    except Exception:
        st.error(f"Treatment column '{treat_col}' could not be converted to numeric.")
        st.stop()

    if not set(df[treat_col].dropna().unique()).issubset({0, 1}):
        st.warning(
            f"Treatment column '{treat_col}' does not look like a 0/1 indicator. "
            "Make sure Treatment = 1 and Control = 0."
        )

    # --- Select covariates ---
    covariates = st.multiselect(
        "Covariate columns to check balance on",
        options=[c for c in df.columns if c != treat_col],
    )

    if st.button("Run balance checks", type="primary"):
        if len(covariates) == 0:
            st.error("Please select at least one covariate.")
            st.stop()

        results = []
        notes = []

        def to_numeric_cov(series: pd.Series):
            """
            Convert covariate to numeric for balance checks.

            - If already numeric → return as float
            - If binary categorical → map one category to 1, the other to 0
            - Else → raise TypeError so we can skip it with a warning
            """
            if pd.api.types.is_numeric_dtype(series):
                return series.astype(float), None

            # Non-numeric: try binary categorical
            uniq = series.dropna().unique()
            if len(uniq) == 2:
                ref = uniq[0]
                num = (series == ref).astype(int)
                note = (
                    f"'{series.name}' treated as binary: "
                    f"{ref} = 1, other = 0."
                )
                return num, note

            raise TypeError(
                f"Covariate '{series.name}' is non-numeric with {len(uniq)} categories; "
                "only numeric or binary categorical variables are supported."
            )

        for cov in covariates:
            try:
                s_orig = df[cov]
                s_num, note = to_numeric_cov(s_orig)

                mask_c = df[treat_col] == 0
                mask_t = df[treat_col] == 1

                x_c = s_num[mask_c].dropna()
                x_t = s_num[mask_t].dropna()

                if x_c.empty or x_t.empty:
                    raise ValueError(
                        f"No non-missing data for '{cov}' in one of the groups."
                    )

                mean_c = x_c.mean()
                mean_t = x_t.mean()
                diff = mean_t - mean_c

                # Pooled SD for SMD
                var_c = x_c.var(ddof=1)
                var_t = x_t.var(ddof=1)
                sd_pooled = np.sqrt((var_c + var_t) / 2) if (var_c + var_t) > 0 else np.nan
                smd = diff / sd_pooled if sd_pooled > 0 else np.nan

                # t-test p-value (if SciPy available)
                if HAVE_SCIPY:
                    t_stat, p_val = stats.ttest_ind(x_t, x_c, equal_var=False, nan_policy="omit")
                else:
                    p_val = np.nan

                results.append(
                    {
                        "covariate": cov,
                        "mean_control": mean_c,
                        "mean_treatment": mean_t,
                        "diff_treat_minus_control": diff,
                        "SMD": smd,
                        "p_value": p_val,
                        "n_control": int(mask_c.sum()),
                        "n_treatment": int(mask_t.sum()),
                    }
                )
                if note:
                    notes.append(note)

            except Exception as e:
                st.warning(f"Skipping '{cov}': {e}")

        if len(results) == 0:
            st.error("No covariates could be processed. Check variable types.")
            st.stop()

        res_df = pd.DataFrame(results)
        st.subheader("Balance table")
        st.dataframe(res_df)

        if notes:
            st.subheader("Notes on covariate coding")
            for n in notes:
                st.info(n)

        # Optional: download results
        csv_bal = res_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download balance results (CSV)",
            data=csv_bal,
            file_name="balance_checks_results.csv",
            mime="text/csv",
        )

else:
    st.info("👆 Upload a dataset with a treatment indicator to run balance checks.")
