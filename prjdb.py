import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Data Preprocessing Dashboard", layout="wide")

st.markdown("""
<style>

/* ===== IMPORT FONT (clean & modern) ===== */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');

/* ===== APPLY FONT EVERYWHERE ===== */
html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
}

/* ===== Main Background (LIGHT CREAM) ===== */
.stApp {
    background-color: #FFF8F0;  /* soft cream */
}

/* ===== Sidebar (soft cream + subtle contrast) ===== */
section[data-testid="stSidebar"] {
    background: linear-gradient(to bottom, #FFF1E6, #FDEBD0);
    padding: 12px;
}

/* ===== Titles ===== */
h1 {
    color: #5D4037;   /* warm brown */
    font-weight: 600;
    letter-spacing: 0.3px;
}

h3 {
    color: #6D4C41;
    font-weight: 500;
}

/* ===== Caption ===== */
.stCaption {
    color: #8D6E63 !important;
    font-size: 14px;
}

/* ===== Buttons (soft elegant green) ===== */
div.stButton > button {
    background-color: #A5D6A7;
    color: #1B5E20;
    border-radius: 10px;
    padding: 8px;
    border: none;
    font-weight: 500;
    transition: 0.3s;
}

div.stButton > button:hover {
    background-color: #81C784;
    color: white;
}

/* ===== Sidebar Step Boxes ===== */
.step-box {
    background: #FFFFFF;
    border-left: 5px solid #D7CCC8;
    padding: 10px;
    border-radius: 10px;
    margin-bottom: 10px;
    font-size: 13px;
}

/* Done */
.done {
    border-left-color: #81C784;
    color: #2E7D32;
    font-weight: 600;
}

/* Pending */
.pending {
    border-left-color: #D7CCC8;
    color: #8D6E63;
}

/* ===== DataFrame ===== */
[data-testid="stDataFrame"] {
    border-radius: 10px;
    border: 1px solid #E0D7CF;
    background-color: #FFFFFF;
}

/* ===== Expander ===== */
.streamlit-expanderHeader {
    font-size: 14px;
    color: #6D4C41;
    font-weight: 500;
}

/* ===== Metrics ===== */
[data-testid="stMetric"] {
    background: #FFFFFF;
    padding: 10px;
    border-radius: 10px;
    border: 1px solid #E0D7CF;
}

/* ===== Alerts ===== */
.stAlert-success {
    background-color: #E8F5E9;
    color: #2E7D32;
}

.stAlert-info {
    background-color: #FFF3E0;
    color: #E65100;
}

.stAlert-warning {
    background-color: #FFF8E1;
    color: #FF6F00;
}

/* ===== Progress Bar ===== */
.stProgress > div > div {
    background-color: #81C784;
}

/* ===== Inputs ===== */
input, textarea {
    border-radius: 8px !important;
}

/* ===== Selectbox ===== */
div[data-baseweb="select"] > div {
    border-radius: 8px;
}

/* ===== Remove top white space ===== */
.block-container {
    padding-top: 0.5rem !important;
}

header {
    visibility: hidden;
}

.main > div {
    padding-top: 0rem;
}

/* ===== Subtle hover ===== */
button:hover {
    transform: scale(1.02);
}

/* ===== Scrollbar ===== */
::-webkit-scrollbar {
    width: 6px;
}
::-webkit-scrollbar-thumb {
    background: #D7CCC8;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# ==== Sidebar: Step Tracker ====
with st.sidebar:
    st.title("Pipeline Steps")
    steps = [
        ("Load Data",      "data_loaded"),
        ("Explore Data",   "data_loaded"),
        ("Visualize",      "data_loaded"),
        ("Drop Columns",   "columns_dropped"),
        ("Split Data",     "dfnum"),
        ("Handle Missing", "missing_handled"),
        ("Standardize",    "standardized"),
        ("Encode",         "encoded"),
    ]
    for label, key in steps:
        done = key in st.session_state and st.session_state[key] is not False
        css = "done" if done else "pending"
        icon = "Done" if done else "Pending"
        st.markdown(f'<div class="step-box {css}">{icon}: {label}</div>', unsafe_allow_html=True)

# ==== Auto-detect encoding suggestion ====
def suggest_encoding(col, series):
    nuniq = series.nunique()
    col_lower = col.lower()
    ordinal_kw = ["grade","level","rank","rating","score","stage","tier",
                  "priority","severity","class","degree","education","satisfaction","frequency"]
    val_set = set(str(v).strip().lower() for v in series.dropna().unique())
    is_binary = (val_set <= {"yes","no"} or val_set <= {"true","false"} or
                 val_set <= {"y","n"} or val_set <= {"male","female"} or val_set <= {"m","f"})
    if nuniq == 2 or is_binary:
        return "Label Encoding", "Binary column — Label Encoding recommended"
    elif any(k in col_lower for k in ordinal_kw) and nuniq <= 10:
        return "Ordinal Encoding", f"{col} suggests a ranked order — Ordinal Encoding recommended"
    elif nuniq <= 6:
        return "One-Hot Encoding", f"Low cardinality ({nuniq} values) — One-Hot Encoding recommended"
    elif nuniq <= 15:
        return "One-Hot Encoding", f"Moderate cardinality ({nuniq} values) — One-Hot possible"
    else:
        return "Drop Column", f"High cardinality ({nuniq} values) — consider dropping"

# ==== Smart Chart Logic ====
def render_smart_chart(df, col1, col2):
    import matplotlib.pyplot as plt

    is_num1 = pd.api.types.is_numeric_dtype(df[col1])
    is_num2 = pd.api.types.is_numeric_dtype(df[col2])

    fig, ax = plt.subplots(figsize=(9, 4))

    if is_num1 and is_num2:
        clean = df[[col1, col2]].dropna()
        ax.scatter(clean[col1], clean[col2], alpha=0.5, s=30, color="#2E86C1", edgecolors="white", linewidth=0.4)
        m, b = np.polyfit(clean[col1], clean[col2], 1)
        x_line = np.linspace(clean[col1].min(), clean[col1].max(), 100)
        ax.plot(x_line, m * x_line + b, color="#E74C3C", linewidth=1.8, linestyle="--", label="Trend")
        ax.set_xlabel(col1)
        ax.set_ylabel(col2)
        ax.set_title(f"Scatter: {col1} vs {col2}")
        ax.legend(fontsize=8)
        chart_note = "📈 Numeric vs Numeric → Scatter plot with trend line"

    elif not is_num1 and is_num2:
        categories = sorted(df[col1].dropna().unique().tolist(), key=str)
        data_by_cat = [df[df[col1] == cat][col2].dropna().values for cat in categories]
        cmap = plt.cm.get_cmap("Set2", len(categories))
        bp = ax.boxplot(data_by_cat, patch_artist=True,
                        medianprops=dict(color="#E74C3C", linewidth=2),
                        whiskerprops=dict(color="#555"),
                        capprops=dict(color="#555"),
                        flierprops=dict(marker="o", color="#BDC3C7", markersize=4, alpha=0.5))
        for patch, i in zip(bp["boxes"], range(len(categories))):
            patch.set_facecolor(cmap(i))
            patch.set_alpha(0.75)
        ax.set_xticks(range(1, len(categories) + 1))
        ax.set_xticklabels([str(c) for c in categories], rotation=20, ha="right")
        ax.set_xlabel(col1)
        ax.set_ylabel(col2)
        ax.set_title(f"Box Plot: {col2} by {col1}")
        chart_note = "📦 Categorical vs Numeric → Box plot"

    elif is_num1 and not is_num2:
        categories = sorted(df[col2].dropna().unique().tolist(), key=str)
        data_by_cat = [df[df[col2] == cat][col1].dropna().values for cat in categories]
        cmap = plt.cm.get_cmap("Set2", len(categories))
        bp = ax.boxplot(data_by_cat, patch_artist=True,
                        medianprops=dict(color="#E74C3C", linewidth=2),
                        whiskerprops=dict(color="#555"),
                        capprops=dict(color="#555"),
                        flierprops=dict(marker="o", color="#BDC3C7", markersize=4, alpha=0.5))
        for patch, i in zip(bp["boxes"], range(len(categories))):
            patch.set_facecolor(cmap(i))
            patch.set_alpha(0.75)
        ax.set_xticks(range(1, len(categories) + 1))
        ax.set_xticklabels([str(c) for c in categories], rotation=20, ha="right")
        ax.set_xlabel(col2)
        ax.set_ylabel(col1)
        ax.set_title(f"Box Plot: {col1} by {col2}")
        chart_note = "📦 Numeric vs Categorical → Box plot"

    else:
        ct = pd.crosstab(df[col1], df[col2])
        cax = ax.imshow(ct.values, aspect="auto", cmap="Blues")
        fig.colorbar(cax, ax=ax, label="Count")
        ax.set_xticks(range(len(ct.columns)))
        ax.set_yticks(range(len(ct.index)))
        ax.set_xticklabels(ct.columns.astype(str), rotation=30, ha="right", fontsize=8)
        ax.set_yticklabels(ct.index.astype(str), fontsize=8)
        for i in range(len(ct.index)):
            for j in range(len(ct.columns)):
                ax.text(j, i, str(ct.values[i, j]), ha="center", va="center", fontsize=8, color="black")
        ax.set_xlabel(col2)
        ax.set_ylabel(col1)
        ax.set_title(f"Heatmap Count: {col1} vs {col2}")
        chart_note = "🟦 Categorical vs Categorical → Heatmap count"

    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    st.caption(chart_note)
    st.pyplot(fig)
    plt.close(fig)

# ==== Main Title ====
st.title("Data Preprocessing Dashboard")
st.caption("Upload your dataset and follow the steps in order to preprocess it.")

# ==== Step 1: Upload CSV ====
st.header("Step 1: Load Data")
uploaded = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded is not None:
    if st.session_state.get("uploaded_filename") != uploaded.name:
        df = pd.read_csv(uploaded)
        st.session_state.clear()
        st.session_state.df = df
        st.session_state.df_original = df.copy()
        st.session_state.data_loaded = True
        st.session_state.uploaded_filename = uploaded.name
        st.session_state.encoded_cols = {}
        st.session_state.encoded = False

if st.session_state.get("data_loaded"):
    st.success(f"Loaded {st.session_state.df.shape[0]} rows x {st.session_state.df.shape[1]} columns")

# ==== Step 2: Explore Data ====
if st.session_state.get("data_loaded"):
    st.header("Step 2: Explore Data")
    df = st.session_state.df

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", int(df.isnull().sum().sum()))

    with st.expander("Preview first 5 rows"):
        st.dataframe(df.head())

    with st.expander("Missing values per column"):
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if missing.empty:
            st.success("No missing values found!")
        else:
            st.dataframe(missing.rename("Missing Count"))

# ==== Step 3: Visualize Data (Fully Automated) ====
if st.session_state.get("data_loaded"):
    st.header("Step 3: Visualize Data")
    df = st.session_state.df

    st.caption("Automatically identifying the two most impactful column relationships in your dataset.")

    if st.button("Auto-Analyze & Generate Charts"):
        from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
        from sklearn.preprocessing import LabelEncoder

        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
        all_cols = df.columns.tolist()

        best_pairs = []

        # --- Strategy 1: Highest absolute Pearson correlation (numeric pairs) ---
        if len(num_cols) >= 2:
            corr = df[num_cols].corr().abs()
            np.fill_diagonal(corr.values, 0)
            max_idx = np.unravel_index(np.argmax(corr.values), corr.shape)
            score = corr.values[max_idx]
            pair = (num_cols[max_idx[0]], num_cols[max_idx[1]])
            best_pairs.append(("corr", pair, score))

        # --- Strategy 2: Mutual Information — best numeric predictor for highest-variance target ---
        if len(num_cols) >= 1 and len(all_cols) >= 2:
            target_col = df[num_cols].var().idxmax() if num_cols else None
            if target_col:
                other_cols = [c for c in all_cols if c != target_col]
                df_temp = df[other_cols + [target_col]].dropna()
                X_encoded = df_temp[other_cols].copy()
                for c in X_encoded.select_dtypes(exclude=np.number).columns:
                    X_encoded[c] = LabelEncoder().fit_transform(X_encoded[c].astype(str))
                y = df_temp[target_col]
                try:
                    mi_scores = mutual_info_regression(X_encoded, y, random_state=0)
                    best_feat_idx = int(np.argmax(mi_scores))
                    best_feat = other_cols[best_feat_idx]
                    mi_score = float(mi_scores[best_feat_idx])
                    pair = (best_feat, target_col)
                    # Only add if not a duplicate of pair 1
                    if not best_pairs or set(pair) != set(best_pairs[0][1]):
                        best_pairs.append(("mi", pair, mi_score))
                except Exception:
                    pass

        # --- Strategy 3: Categorical target — mutual info classification ---
        if len(best_pairs) < 2 and cat_cols and num_cols:
            target_cat = min(cat_cols, key=lambda c: df[c].nunique())
            df_temp = df[num_cols + [target_cat]].dropna()
            y_cat = LabelEncoder().fit_transform(df_temp[target_cat].astype(str))
            try:
                mi_scores = mutual_info_classif(df_temp[num_cols], y_cat, random_state=0)
                best_feat = num_cols[int(np.argmax(mi_scores))]
                pair = (best_feat, target_cat)
                if not any(set(pair) == set(p[1]) for p in best_pairs):
                    best_pairs.append(("mi_cat", pair, float(np.max(mi_scores))))
            except Exception:
                pass

        # --- Fallback: pick first two distinct column pairs ---
        if len(best_pairs) < 2:
            for i in range(len(all_cols)):
                for j in range(i + 1, len(all_cols)):
                    pair = (all_cols[i], all_cols[j])
                    if not any(set(pair) == set(p[1]) for p in best_pairs):
                        best_pairs.append(("fallback", pair, 0.0))
                    if len(best_pairs) >= 2:
                        break
                if len(best_pairs) >= 2:
                    break

        # --- Filter out any columns not present in df before storing ---
        valid_pairs = [
            (strategy, (c1, c2), score)
            for strategy, (c1, c2), score in best_pairs[:2]
            if c1 in df.columns and c2 in df.columns
        ]
        st.session_state.auto_viz_pairs = valid_pairs
        st.rerun()

    # --- Render stored charts (persists across reruns) ---
    if st.session_state.get("auto_viz_pairs"):
        df = st.session_state.df

        # Re-validate pairs against current df columns (guards against columns dropped later)
        best_pairs = [
            (strategy, (c1, c2), score)
            for strategy, (c1, c2), score in st.session_state.auto_viz_pairs
            if c1 in df.columns and c2 in df.columns
        ]

        if not best_pairs:
            st.warning("Previously selected columns were removed. Click 'Auto-Analyze & Generate Charts' again.")
            st.session_state.pop("auto_viz_pairs", None)
        else:
            label_map = {
                "corr":    lambda c1, c2, s: f"📊 **Highest correlation ({s:.2f})** — `{c1}` vs `{c2}`",
                "mi":      lambda c1, c2, s: f"🔍 **Best mutual info predictor (score: {s:.3f})** — `{c1}` → `{c2}`",
                "mi_cat":  lambda c1, c2, s: f"🎯 **Top feature for class target (score: {s:.3f})** — `{c1}` → `{c2}`",
                "fallback":lambda c1, c2, s: f"📈 **`{c1}` vs `{c2}`**",
            }

            chart_cols = st.columns(2)
            for idx, (strategy, (c1, c2), score) in enumerate(best_pairs):
                with chart_cols[idx]:
                    caption_fn = label_map.get(strategy, label_map["fallback"])
                    st.markdown(caption_fn(c1, c2, score))
                    render_smart_chart(df, c1, c2)

# ==== Step 4: Drop Unwanted Columns ====
if st.session_state.get("data_loaded"):
    st.header("Step 4: Drop Unwanted Columns")
    df_current = st.session_state.df

    if st.session_state.get("columns_dropped"):
        dropped = st.session_state.dropped_cols_list
        st.success(f"Dropped: {dropped} — {df_current.shape[1]} columns remaining.")
        st.dataframe(df_current.head())
        if st.button("Redo Column Drop"):
            st.session_state.pop("columns_dropped", None)
            st.session_state.pop("dropped_cols_list", None)
            st.rerun()
    else:
        df_orig = st.session_state.get("df_original", df_current)
        auto_suggested = [c for c in df_orig.columns if df_orig[c].nunique() == len(df_orig)]
        if auto_suggested:
            st.warning(f"Likely ID/Name columns (unique per row): {', '.join(auto_suggested)}")

        cols_to_drop = st.multiselect(
            "Select columns to drop",
            options=df_current.columns.tolist(),
            default=[c for c in auto_suggested if c in df_current.columns]
        )

        if st.button("Drop Selected Columns"):
            if cols_to_drop:
                st.session_state.df = df_current.drop(columns=cols_to_drop)
            st.session_state.columns_dropped = True
            st.session_state.dropped_cols_list = cols_to_drop
            st.session_state.pop("dfnum", None)
            st.session_state.pop("dfcat", None)
            st.session_state.pop("original_cat_cols", None)
            st.session_state.pop("dfcat_checkpoint", None)
            st.session_state.pop("missing_handled", None)
            st.session_state.pop("standardized", None)
            st.session_state.encoded_cols = {}
            st.session_state.encoded = False
            st.rerun()

# ==== Step 5: Split Data ====
if st.session_state.get("columns_dropped"):
    st.header("Step 5: Split Numeric & Categorical")

    if st.button("Split Data"):
        df = st.session_state.df
        dfnum = df.select_dtypes(include=np.number).copy()
        dfcat = df.select_dtypes(exclude=np.number).copy()
        st.session_state.dfnum = dfnum
        st.session_state.dfcat = dfcat
        st.session_state.encoded_cols = {}
        st.session_state.encoded = False
        st.session_state.pop("dfcat_checkpoint", None)
        st.session_state.original_cat_cols = dfcat.columns.tolist()
        st.rerun()

    if "dfnum" in st.session_state:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Numeric Columns")
            st.write(list(st.session_state.dfnum.columns))
        with c2:
            st.subheader("Categorical Columns")
            st.write(list(st.session_state.dfcat.columns))

# ==== Step 6: Handle Missing Values ====
if "dfnum" in st.session_state and "dfcat" in st.session_state:
    st.header("Step 6: Handle Missing Values")

    if st.button("Impute Missing Values"):
        from sklearn.impute import SimpleImputer
        dfnum = st.session_state.dfnum.copy()
        dfcat = st.session_state.dfcat.copy()
        if dfnum.shape[1] > 0:
            dfnum.iloc[:, :] = SimpleImputer(strategy="mean").fit_transform(dfnum)
        if dfcat.shape[1] > 0:
            dfcat.iloc[:, :] = SimpleImputer(strategy="most_frequent").fit_transform(dfcat)
        st.session_state.dfnum = dfnum
        st.session_state.dfcat = dfcat
        st.session_state.dfcat_checkpoint = dfcat.copy()
        st.session_state.missing_handled = True
        st.rerun()

    if st.session_state.get("missing_handled"):
        st.success("Missing values imputed! (Numeric: mean | Categorical: most frequent)")
        combined = pd.concat([st.session_state.dfnum.reset_index(drop=True),
                               st.session_state.dfcat.reset_index(drop=True)], axis=1)
        if combined.isnull().sum().sum() == 0:
            st.success("No missing values remaining!")
        with st.expander("Preview after imputation"):
            c1, c2 = st.columns(2)
            with c1:
                st.caption("Numeric")
                st.dataframe(st.session_state.dfnum.head())
            with c2:
                st.caption("Categorical")
                st.dataframe(st.session_state.dfcat.head())

# ==== Step 7: Standardization ====
if st.session_state.get("missing_handled"):
    st.header("Step 7: Standardize Numeric Data")

    if st.button("Apply StandardScaler"):
        from sklearn.preprocessing import StandardScaler
        dfnum = st.session_state.dfnum.copy()
        dfnum.iloc[:, :] = StandardScaler().fit_transform(dfnum)
        st.session_state.dfnum = dfnum
        st.session_state.standardized = True
        st.rerun()

    if st.session_state.get("standardized"):
        st.success("Standardization complete!")
        with st.expander("Preview standardized data"):
            st.dataframe(st.session_state.dfnum.head())

# ==== Step 8: Encoding ====
if st.session_state.get("missing_handled") and "dfcat" in st.session_state:
    st.header("Step 8: Encode Categorical Variables")

    dfcat = st.session_state.dfcat
    if "encoded_cols" not in st.session_state:
        st.session_state.encoded_cols = {}
    if "original_cat_cols" not in st.session_state:
        st.session_state.original_cat_cols = dfcat.columns.tolist()

    original_cat_cols = st.session_state.original_cat_cols
    current_cat_cols = dfcat.columns.tolist()

    stale_keys = [k for k in st.session_state.encoded_cols if k not in original_cat_cols]
    for k in stale_keys:
        del st.session_state.encoded_cols[k]

    st.subheader("Categorical Columns Overview")
    summary_data = []
    for col in original_cat_cols:
        enc_status = st.session_state.encoded_cols.get(col, "Not encoded")
        already_handled = col in st.session_state.encoded_cols
        if already_handled:
            summary_data.append({"Column": col, "Unique Values": "-", "Sample Values": "-",
                                  "Auto-Suggestion": enc_status, "Status": enc_status})
        elif col in current_cat_cols:
            uniq = dfcat[col].nunique()
            vals = ", ".join(str(v) for v in dfcat[col].unique()[:5])
            if uniq > 5: vals += " ..."
            _, suggestion_msg = suggest_encoding(col, dfcat[col])
            summary_data.append({"Column": col, "Unique Values": uniq, "Sample Values": vals,
                                  "Auto-Suggestion": suggestion_msg, "Status": "Not encoded"})
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True)

    done_count = len(st.session_state.encoded_cols)
    total = len(original_cat_cols)
    st.caption(f"Progress: {done_count}/{total} columns handled")
    st.progress(min(done_count / total, 1.0) if total > 0 else 0)
    st.divider()

    remaining_cols = [c for c in original_cat_cols
                      if c not in st.session_state.encoded_cols and c in current_cat_cols]

    if not remaining_cols:
        st.success("All categorical columns have been encoded!")
    else:
        col_to_encode = st.selectbox("Select column to encode", remaining_cols)
        col_series = dfcat[col_to_encode]
        col_unique_vals = sorted(col_series.unique().tolist(), key=str)
        col_nunique = len(col_unique_vals)

        suggested_type, suggestion_msg = suggest_encoding(col_to_encode, col_series)
        st.info(f"{col_to_encode} — {col_nunique} unique values: {col_unique_vals} | {suggestion_msg}")

        enc_options = ["One-Hot Encoding", "Ordinal Encoding", "Label Encoding", "Drop Column"]
        default_idx = enc_options.index(suggested_type) if suggested_type in enc_options else 0
        enc_type = st.radio("Encoding type", enc_options, index=default_idx, horizontal=True)

        if enc_type == "One-Hot Encoding":
            drop_first = st.toggle("Use drop_first (recommended)", value=True)
            expected_cols = col_nunique - 1 if drop_first else col_nunique
            st.caption(f"Will create {expected_cols} new columns from '{col_to_encode}'")
            if st.button("Apply One-Hot Encoding"):
                dfcat = st.session_state.dfcat.copy()
                dfcat = pd.get_dummies(dfcat, columns=[col_to_encode], dtype=int, drop_first=drop_first)
                st.session_state.dfcat = dfcat
                st.session_state.encoded_cols[col_to_encode] = f"One-Hot (drop_first={drop_first})"
                st.session_state.encoded = True
                st.rerun()

        elif enc_type == "Ordinal Encoding":
            default_order = ", ".join(str(v) for v in col_unique_vals)
            order_input = st.text_input("Category order (lowest to highest)", value=default_order)
            if order_input.strip():
                order_list = [x.strip() for x in order_input.split(",")]
                st.write("Mapping preview:", {v: i for i, v in enumerate(order_list)})
            if st.button("Apply Ordinal Encoding"):
                if order_input.strip():
                    from sklearn.preprocessing import OrdinalEncoder
                    order_list = [x.strip() for x in order_input.split(",")]
                    try:
                        dfcat = st.session_state.dfcat.copy()
                        dfcat[col_to_encode] = OrdinalEncoder(categories=[order_list]).fit_transform(dfcat[[col_to_encode]])
                        st.session_state.dfcat = dfcat
                        st.session_state.encoded_cols[col_to_encode] = "Ordinal"
                        st.session_state.encoded = True
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
                else:
                    st.warning("Please enter the category order.")

        elif enc_type == "Label Encoding":
            manual_map = {}
            cols_ui = st.columns(min(col_nunique, 4))
            for i, val in enumerate(col_unique_vals):
                with cols_ui[i % len(cols_ui)]:
                    manual_map[val] = st.number_input(f"'{val}' ->", value=i, step=1, key=f"lbl_{col_to_encode}_{val}")
            st.write("Mapping preview:", manual_map)
            if st.button("Apply Label Encoding"):
                dfcat = st.session_state.dfcat.copy()
                dfcat[col_to_encode] = dfcat[col_to_encode].map(manual_map)
                if dfcat[col_to_encode].isnull().any():
                    st.error("Some values could not be mapped.")
                else:
                    st.session_state.dfcat = dfcat
                    st.session_state.encoded_cols[col_to_encode] = "Label"
                    st.session_state.encoded = True
                    st.rerun()

        elif enc_type == "Drop Column":
            st.warning(f"This will permanently remove '{col_to_encode}' from the dataset.")
            if st.button("Confirm Drop"):
                dfcat = st.session_state.dfcat.copy()
                st.session_state.dfcat = dfcat.drop(columns=[col_to_encode])
                st.session_state.encoded_cols[col_to_encode] = "Dropped"
                st.rerun()

    if st.session_state.get("encoded_cols"):
        if st.button("Reset All Encodings"):
            st.session_state.dfcat = st.session_state.dfcat_checkpoint.copy()
            st.session_state.encoded_cols = {}
            st.session_state.encoded = False
            st.rerun()

# ==== Step 9: Export ====
if st.session_state.get("encoded") or st.session_state.get("standardized"):
    st.header("Step 9: Export Processed Data")

    dfnum = st.session_state.get("dfnum", pd.DataFrame())
    dfcat = st.session_state.get("dfcat", pd.DataFrame())
    valid_num_cols = st.session_state.df.columns.tolist()
    dfnum = dfnum[[c for c in dfnum.columns if c in valid_num_cols]]
    df_final = pd.concat([dfnum.reset_index(drop=True), dfcat.reset_index(drop=True)], axis=1)

    st.success(f"Final dataset: {df_final.shape[0]} rows x {df_final.shape[1]} columns")
    st.dataframe(df_final.head())
    st.download_button(
        label="Download Processed CSV",
        data=df_final.to_csv(index=False).encode("utf-8"),
        file_name="processed_data.csv",
        mime="text/csv"
    )
