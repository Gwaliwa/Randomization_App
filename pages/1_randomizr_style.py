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
    # Read file
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("Preview of uploaded data")
    st.dataframe(df.head())

    # Column selection
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
                raise ValueError("n_treated cannot exceed number of rows")
            idx = rng.choice(n, size=n_treated, replace=False)
            treat[idx] = 1
        elif p_treat is not None:
            treat = (rng.random(n) < p_treat).astype(int)
        else:
            raise ValueError("Either p_treat or n_treated must be provided.")

        return treat

    def blocked_randomization(df, block_col, p_treat=None, n_treated=None, seed=None):
        rng = np.random.default_rng(seed)
        treat = np.zeros(len(df), dtype=int)
        groups = df.groupby(block_col, sort=False)

        for block, idx in groups.groups.items():
            block_idx = np.array(list(idx))
            n_block = len(block_idx)

            if n_treated is not None:
                # Allocate proportionally to block size
                n_block_treat = int(round(n_treated * n_block / len(df)))
                n_block_treat = min(n_block_treat, n_block)
                chosen = rng.choice(block_idx, size=n_block_treat, replace=False)
                treat[chosen] = 1
            elif p_treat is not None:
                chosen = block_idx[rng.random(n_block) < p_treat]
                treat[chosen] = 1
            else:
                raise ValueError("Either p_treat or n_treated must be provided.")

        return treat

    def cluster_randomization(df, cluster_col, p_treat=None, n_treated=None, seed=None):
        rng = np.random.default_rng(seed)
        clusters = df[cluster_col].unique()
        n_clusters = len(clusters)

        if n_treated is not None:
            if n_treated > n_clusters:
                raise ValueError("n_treated cannot exceed number of clusters")
            treated_clusters = rng.choice(clusters, size=n_treated, replace=False)
        elif p_treat is not None:
            treated_clusters = [c for c in clusters if rng.random() < p_treat]
        else:
            raise ValueError("Either p_treat or n_treated must be provided.")

        treated_clusters = set(treated_clusters)
        treat = df[cluster_col].apply(lambda c: 1 if c in treated_clusters else 0).to_numpy()
        return treat

    # ---- Generate assignment ----

    if st.button("Generate assignment", type="primary"):
        df_out = df.copy()

        if design_type == "Complete randomization":
            df_out["treat"] = complete_randomization(df, p_treat, n_treated, seed)

        elif design_type == "Blocked / stratified randomization":
            df_out["treat"] = blocked_randomization(df, block_col, p_treat, n_treated, seed)

        elif design_type == "Cluster randomization":
            df_out["treat"] = cluster_randomization(df, cluster_col, p_treat, n_treated, seed)

        st.success("Assignment generated.")

        st.subheader("Assignment summary")
        st.write(
            df_out["treat"]
            .value_counts()
            .rename(index={0: "Control (0)", 1: "Treatment (1)"})
        )

        st.subheader("Preview with assignment")
        cols_to_show = [id_col]
        if block_col is not None:
            cols_to_show.append(block_col)
        if cluster_col is not None:
            cols_to_show.append(cluster_col)
        cols_to_show.append("treat")
        st.dataframe(df_out[cols_to_show].head())

        csv = df_out.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download full dataset with assignment (CSV)",
            data=csv,
            file_name="randomizr_style_assignment.csv",
            mime="text/csv",
        )

else:
    st.info("👆 Upload a dataset to get started.")
