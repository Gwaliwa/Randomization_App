import numpy as np
import pandas as pd
import streamlit as st

st.title("🎲 randtreat-style Assignment (Multi-arm, Unequal Fractions & Misfits)")

st.markdown(
    """
This page mimics core ideas from **`randtreat`**:

- Multi-arm randomization (2+ arms)  
- Unequal treatment fractions  
- Stratified (blocked) assignment  
- Misfit handling (units that don't fit exact fractions because of rounding)

**Workflow**

1. Upload a CSV/Excel with an **ID column** (and optional **strata** column).  
2. Choose the number of arms and desired **fractions per arm**.  
3. Choose how to handle **misfits**.  
4. Download assignment with an `arm` column and `misfit` flag.
"""
)

uploaded_file = st.file_uploader(
    "Upload dataset (CSV or Excel)", type=["csv", "xlsx"], key="randtreat_file"
)

if uploaded_file is not None:
    # Read file
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("Preview of uploaded data")
    st.dataframe(df.head())

    id_col = st.selectbox("ID column", options=df.columns, key="rt_id_col")

    strata_col_choice = st.selectbox(
        "Strata / block column (optional)",
        options=["<none>"] + list(df.columns),
        key="rt_strata_col",
    )
    strata_col = None if strata_col_choice == "<none>" else strata_col_choice

    n_arms = st.number_input(
        "Number of treatment arms",
        min_value=2,
        max_value=6,
        value=2,
        step=1,
    )

    st.markdown("#### Desired fractions per arm")
    cols = st.columns(n_arms)
    fractions = []
    default_frac = round(1.0 / n_arms, 3)

    for i in range(n_arms):
        with cols[i]:
            f = st.number_input(
                f"Arm {i} fraction",
                min_value=0.0,
                max_value=1.0,
                value=default_frac,
                step=0.01,
                key=f"frac_{i}",
            )
            fractions.append(f)

    frac_sum = sum(fractions)
    st.write(f"Sum of fractions: **{frac_sum:.3f}** (should be close to 1.0)")
    if not (0.99 <= frac_sum <= 1.01):
        st.warning(
            "⚠️ Fractions do not sum to 1. "
            "Assignment will be scaled proportionally within each stratum."
        )

    misfit_method = st.selectbox(
        "Misfit handling",
        options=[
            "Assign and flag misfits",
            "Drop misfits from output",
            "Leave misfits unassigned (arm = NA)",
        ],
    )

    seed = st.number_input(
        "Random seed (optional)",
        min_value=0,
        value=1234,
        step=1,
    )

    # ---- Helper for multi-arm assignment ----

    def assign_with_fractions(indices, fractions, rng, misfit_method):
        """
        indices: array of row indices for a stratum
        fractions: list of desired fractions (not necessarily summing to 1)
        Returns:
            arm assignment (dict: row_idx -> arm int),
            misfit flag (dict: row_idx -> bool)
        """
        n = len(indices)
        K = len(fractions)

        if n == 0:
            return {}, {}

        # Normalize fractions if they don't sum to 1
        frac_arr = np.array(fractions, dtype=float)
        frac_arr = frac_arr / frac_arr.sum()

        # Target counts per arm (floor)
        target_counts = np.floor(frac_arr * n).astype(int)
        assigned_target = target_counts.sum()
        n_misfits = n - assigned_target

        # Shuffle indices to randomize order
        shuffled = np.array(indices)
        rng.shuffle(shuffled)

        arm_assign = -1 * np.ones(n, dtype=int)
        misfit = np.zeros(n, dtype=bool)

        # Assign target counts deterministically by arm
        start = 0
        for k in range(K):
            end = start + target_counts[k]
            if end > n:
                end = n
            arm_assign[start:end] = k
            start = end

        # Remaining units are misfits
        if n_misfits > 0:
            misfit_idx = np.arange(assigned_target, n)
            misfit[misfit_idx] = True

            if misfit_method == "Assign and flag misfits":
                # Randomly assign misfit units to arms but keep misfit=True
                rand_arms = rng.integers(0, K, size=n_misfits)
                arm_assign[misfit_idx] = rand_arms

            elif misfit_method in [
                "Drop misfits from output",
                "Leave misfits unassigned (arm = NA)",
            ]:
                # arm remains -1, misfit=True
                pass
            else:
                raise ValueError("Unknown misfit method")

        # Map back to original row indices
        arm_by_row = {}
        misfit_by_row = {}
        for order_pos, row_idx in enumerate(shuffled):
            arm_by_row[row_idx] = int(arm_assign[order_pos])
            misfit_by_row[row_idx] = bool(misfit[order_pos])

        return arm_by_row, misfit_by_row

    # ---- Generate assignment ----

    if st.button("Generate multi-arm assignment", type="primary"):
        rng = np.random.default_rng(seed)
        df_out = df.copy()

        arm_series = pd.Series(index=df.index, dtype="Int64")
        misfit_series = pd.Series(False, index=df.index, dtype=bool)

        if strata_col is None:
            # Single stratum: all rows
            arm_map, misfit_map = assign_with_fractions(
                df.index.to_list(), fractions, rng, misfit_method
            )
            arm_series = arm_series.index.to_series().map(arm_map).astype("Int64")
            misfit_series = misfit_series.index.to_series().map(misfit_map).astype(bool)
        else:
            # Strata-specific assignment
            for stratum, idx in df.groupby(strata_col).groups.items():
                idx_list = list(idx)
                arm_map, misfit_map = assign_with_fractions(
                    idx_list, fractions, rng, misfit_method
                )
                arm_series.loc[idx_list] = pd.Series(arm_map)
                misfit_series.loc[idx_list] = pd.Series(misfit_map)

        # Optionally drop misfits
        if misfit_method == "Drop misfits from output":
            keep_mask = ~misfit_series
            df_out = df_out.loc[keep_mask].copy()
            arm_series = arm_series.loc[keep_mask]
            misfit_series = misfit_series.loc[keep_mask]

        df_out["arm"] = arm_series
        df_out["misfit"] = misfit_series

        st.success("Multi-arm assignment generated.")

        st.subheader("Assignment summary (excluding arm = -1)")
        valid = df_out[df_out["arm"] >= 0]
        st.write(valid["arm"].value_counts().sort_index())

        st.subheader("Preview with assignment")
        cols_to_show = [id_col]
        if strata_col is not None:
            cols_to_show.append(strata_col)
        cols_to_show += ["arm", "misfit"]
        st.dataframe(df_out[cols_to_show].head())

        csv = df_out.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download dataset with assignments (CSV)",
            data=csv,
            file_name="randtreat_style_assignment.csv",
            mime="text/csv",
        )

else:
    st.info("👆 Upload a dataset to get started.")
