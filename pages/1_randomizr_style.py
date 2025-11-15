import numpy as np
import pandas as pd
import streamlit as st

st.title("🧪 randomizr-style Assignment Generator")

st.markdown(
    """
This page mimics key ideas from **`randomizr`**:

- Complete randomization  
- Blocked / stratified randomization  
- Cluster randomization  

**Workflow**

1. Upload a CSV (or Excel) with at least an **ID column**.  
2. Optionally include **block/strata** and/or **cluster** columns.  
3. Choose a design and parameters.  
4. Download the assignment table.
"""
)

uploaded_file = st.file_uploader(
    "Upload dataset (CSV or Excel)", type=["csv", "xlsx"], key="rand_file"
)

if uploaded_file is not None:
    # --- Read file safely ---
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

    # --- Column selection ---
    if len(df.columns) == 0:
        st.error("No columns found in the uploaded dataset.")
        st.stop()

    id_col = st.selectbox("ID column", options=df.columns, key="id_col")

    design_type = st.selectbox(
        "Design type",
        options=[
            "Complete randomization",
            "Blocked / stratified randomization",
            "Cluster randomization",
        ],
    )

    block_col = None
    cluster_col = None

    if design_type == "Blocked / stratified randomization":
        block_col = st.selectbox(
            "Block / strata column",
            options=df.columns,
            key="block_col",
        )

    if design_type == "Cluster randomization":
        cluster_col = st.selectbox(
            "Cluster column",
            options=df.columns,
            key="cluster_col",
        )

    st.markdown("#### Treatment assignment settings")
    assign_mode = st.radio(
        "Specify treatment by:",
        ["Treatment probability (same for all units)", "Fixed number treated"],
        horizontal=True,
    )

    p_treat = None
    n_treated = None

    if assign_mode == "Treatment probability (same for all units)":
        p_treat = st.slider(
            "Treatment probability",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
        )
    else:
        max_n = len(df)
        n_treated = st.number_input(
            "Number of treated units (overall)",
            min_value=1,
            max_value=max_n,
            value=max_n // 2,
            step=1,
        )

    seed = st.number_input(
        "Random seed (optional)",
        min_value=0,
        value=123,
        step=1,
    )

    # ---- Randomization helpers ----

    def complete_randomization(df, p_treat=None, n_treated=None, seed=None):
        rng = np.random.default_rng(seed)
        n = len(df)
        treat = np.zeros(n, dtype=int)

        if n_treated is not None:
            if n_treated > n:
                raise ValueError("Number treated (n_treated) cannot exceed number of rows.")
            idx = rng.choice(n, size=int(n_treated), replace=False)
            treat[idx] = 1
        elif p_treat is not None:
            if not (0 <= p_treat <= 1):
                raise ValueError("Treatment probability must be between 0 and 1.")
            treat = (rng.random(n) < p_treat).astype(int)
        else:
            raise ValueError("Either p_treat or n_treated must be provided.")

        return treat

    def blocked_randomization(df, block_col, p_treat=None, n_treated=None, seed=None):
        if block_col not in df.columns:
            raise ValueError(f"Block column '{block_col}' not found in data.")
        rng = np.random.default_rng(seed)
        treat = np.zeros(len(df), dtype=int)
        groups = df.groupby(block_col, sort=False)

        total_n = len(df)
        for block, idx in groups.groups.items():
            block_idx = np.array(list(idx))
            n_block = len(block_idx)

            if n_block == 0:
                continue

            if n_treated is not None:
                # Allocate proportionally to block size
                n_block_treat = int(round(n_treated * n_block / total_n))
                n_block_treat = min(n_block_treat, n_block)
                if n_block_treat > 0:
                    chosen = rng.choice(block_idx, size=int(n_block_treat), replace=False)
                    treat[chosen] = 1
            elif p_treat is not None:
                if not (0 <= p_treat <= 1):
                    raise ValueError("Treatment probability must be between 0 and 1.")
                chosen = block_idx[rng.random(n_block) < p_treat]
                treat[chosen] = 1
            else:
                raise ValueError("Either p_treat or n_treated must be provided.")

        return treat

    def cluster_randomization(df, cluster_col, p_treat=None, n_treated=None, seed=None):
        if cluster_col not in df.columns:
            raise ValueError(f"Cluster column '{cluster_col}' not found in data.")
        rng = np.random.default_rng(seed)
        clusters = df[cluster_col].dropna().unique()
        n_clusters = len(clusters)

        if n_clusters == 0:
            raise ValueError("No clusters found (all values missing?).")

        if n_treated is not None:
            if n_treated > n_clusters:
                raise ValueError("Number treated clusters cannot exceed number of clusters.")
            treated_clusters = rng.choice(clusters, size=int(n_treated), replace=False)
        elif p_treat is not None:
            if not (0 <= p_treat <= 1):
                raise ValueError("Treatment probability must be between 0 and 1.")
            treated_clusters = [c for c in clusters if rng.random() < p_treat]
        else:
            raise ValueError("Either p_treat or n_treated must be provided.")

        treated_clusters = set(treated_clusters)
        treat = df[cluster_col].apply(lambda c: 1 if c in treated_clusters else 0).to_numpy()
        return treat

    # ---- Generate assignment ----

    if st.button("Generate assignment", type="primary"):
        try:
            df_out = df.copy()

            if design_type == "Complete randomization":
                df_out["treat"] = complete_randomization(
                    df, p_treat=p_treat, n_treated=n_treated, seed=seed
                )

            elif design_type == "Blocked / stratified randomization":
                if block_col is None:
                    raise ValueError("Please select a block/strata column.")
                df_out["treat"] = blocked_randomization(
                    df, block_col=block_col, p_treat=p_treat, n_treated=n_treated, seed=seed
                )

            elif design_type == "Cluster randomization":
                if cluster_col is None:
                    raise ValueError("Please select a cluster column.")
                df_out["treat"] = cluster_randomization(
                    df, cluster_col=cluster_col, p_treat=p_treat, n_treated=n_treated, seed=seed
                )

            st.success("Assignment generated.")

            # --- Assignment summary ---
            st.subheader("Assignment summary")
            try:
                summary = (
                    df_out["treat"]
                    .value_counts(dropna=False)
                    .rename(index={0: "Control (0)", 1: "Treatment (1)"})
                )
                st.write(summary)
            except Exception:
                st.write(df_out["treat"].value_counts(dropna=False))

            # --- Preview with assignment ---
            st.subheader("Preview with assignment")
            cols_to_show = [id_col]
            if block_col is not None:
                cols_to_show.append(block_col)
            if cluster_col is not None:
                cols_to_show.append(cluster_col)
            cols_to_show.append("treat")

            # ✅ Remove duplicate column names while keeping order
            cols_to_show = list(dict.fromkeys(cols_to_show))

            # Only keep columns that actually exist
            cols_to_show = [c for c in cols_to_show if c in df_out.columns]

            st.dataframe(df_out[cols_to_show].head())

            # --- Download ---
            csv = df_out.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download full dataset with assignment (CSV)",
                data=csv,
                file_name="randomizr_style_assignment.csv",
                mime="text/csv",
            )

        except Exception as e:
            st.error(f"❌ Randomization failed: {e}")

else:
    st.info("👆 Upload a dataset to get started.")
