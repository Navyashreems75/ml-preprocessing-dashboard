import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Marketing Campaign ML", layout="wide")

st.markdown("""
<style>
    /* Page background */
    .stApp { background-color: #f0f4ff; }

    /* Sidebar */
    section[data-testid="stSidebar"] { background-color: #dce8ff; }

    /* Section headers */
    h1 { color: #1e3a8a; }
    h2 { color: #1e3a8a; background-color: #dbeafe; padding: 8px 14px; border-radius: 8px; }
    h3 { color: #1e40af; }

    /* Buttons */
    div.stButton > button {
        background-color: #16a34a;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 8px 18px;
        font-weight: 600;
    }
    div.stButton > button:hover {
        background-color: #15803d;
        color: white;
    }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background-color: #dbeafe;
        border-radius: 10px;
        padding: 12px;
        border: 1px solid #93c5fd;
    }

    /* Divider */
    hr { border-color: #bfdbfe; }

    /* Expander */
    details { background-color: #eff6ff; border-radius: 8px; padding: 4px; }
</style>
""", unsafe_allow_html=True)

st.title(" Marketing Campaign — Preprocessing & Model Training")

# ── Load Data ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_raw():
    return pd.read_csv(r"C:\Users\User\Downloads\marketing_campaign.csv")

df_raw = load_raw()

if "df" not in st.session_state:
    st.session_state.df = df_raw.copy()
if "steps_done" not in st.session_state:
    st.session_state.steps_done = []
if "model" not in st.session_state:
    st.session_state.model = None
if "feature_cols" not in st.session_state:
    st.session_state.feature_cols = None
if "example" not in st.session_state:
    st.session_state.example = None

# ── STEP 1: Dataset Basics ─────────────────────────────────────────────────────
st.header("1️⃣ Dataset Basics")
st.dataframe(df_raw.head(), use_container_width=True)
st.write(f"**Shape:** {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")
st.write("**Columns:**", list(df_raw.columns))

with st.expander("Show df.info()"):
    import io
    buf = io.StringIO()
    df_raw.info(buf=buf)
    st.text(buf.getvalue())

# ── STEP 2: Missing Values ─────────────────────────────────────────────────────
st.divider()
st.header("2️⃣ Missing Values")
missing = st.session_state.df.isnull().sum()
st.dataframe(missing.rename("Missing Count").to_frame(), use_container_width=True)

if "fill_income" not in st.session_state.steps_done:
    if st.button("Fill Missing Income with Median"):
        st.session_state.df["Income"] = st.session_state.df["Income"].fillna(
            st.session_state.df["Income"].median()
        )
        st.session_state.steps_done.append("fill_income")
        st.rerun()
else:
    st.success("✅ Missing Income values filled with median")
    st.dataframe(st.session_state.df.isnull().sum().rename("Missing Count").to_frame(), use_container_width=True)

# ── STEP 3: Feature Engineering ───────────────────────────────────────────────
st.divider()
st.header("3️⃣ Feature Engineering")

if "create_age" not in st.session_state.steps_done:
    if st.button("Create Age Column (2024 - Year_Birth)"):
        st.session_state.df["Age"] = 2024 - st.session_state.df["Year_Birth"]
        st.session_state.steps_done.append("create_age")
        st.rerun()
else:
    st.success("✅ Age column created")

if "create_spending" not in st.session_state.steps_done:
    if st.button("Create Total_Spending Column"):
        cols = ["MntWines","MntFruits","MntMeatProducts","MntFishProducts","MntSweetProducts","MntGoldProds"]
        st.session_state.df["Total_Spending"] = st.session_state.df[cols].sum(axis=1)
        st.session_state.steps_done.append("create_spending")
        st.rerun()
else:
    st.success("✅ Total_Spending column created")

if {"create_age","create_spending"}.issubset(st.session_state.steps_done):
    st.dataframe(st.session_state.df[["Age","Total_Spending"]].head(), use_container_width=True)

# ── STEP 4: Drop Unused Columns ───────────────────────────────────────────────
st.divider()
st.header("4️⃣ Drop Unused Columns")
st.write("Dropping: `ID`, `Year_Birth`, `Dt_Customer`, `Z_CostContact`, `Z_Revenue`")

if "drop_cols" not in st.session_state.steps_done:
    if st.button("Drop Columns"):
        st.session_state.df = st.session_state.df.drop(
            ["ID","Year_Birth","Dt_Customer","Z_CostContact","Z_Revenue"], axis=1, errors="ignore"
        )
        st.session_state.steps_done.append("drop_cols")
        st.rerun()
else:
    st.success(f"✅ Columns dropped — new shape: {st.session_state.df.shape}")

# ── STEP 5: Encode Categoricals ───────────────────────────────────────────────
st.divider()
st.header("5️⃣ Encode Categorical Columns")
st.write("One-hot encoding `Education` and `Marital_Status` (drop_first=True)")

if "encode" not in st.session_state.steps_done:
    if st.button("Encode Categoricals"):
        st.session_state.df = pd.get_dummies(
            st.session_state.df, columns=["Education","Marital_Status"], drop_first=True
        )
        st.session_state.steps_done.append("encode")
        st.rerun()
else:
    st.success(f"✅ Encoded — new shape: {st.session_state.df.shape}")
    st.dataframe(st.session_state.df.head(), use_container_width=True)

# ── STEP 6: Scale Features ────────────────────────────────────────────────────
st.divider()
st.header("6️⃣ Scale Numerical Features")
st.write("StandardScaler on: `Income`, `Age`, `Recency`, `Total_Spending`, `NumWebVisitsMonth`")

if "scale" not in st.session_state.steps_done:
    if st.button("Scale Features"):
        num_cols = ["Income","Age","Recency","Total_Spending","NumWebVisitsMonth"]
        existing = [c for c in num_cols if c in st.session_state.df.columns]
        scaler = StandardScaler()
        st.session_state.df[existing] = scaler.fit_transform(st.session_state.df[existing])
        st.session_state.scaler = scaler          # save fitted scaler
        st.session_state.steps_done.append("scale")
        st.rerun()
else:
    st.success("✅ Features scaled")
    st.dataframe(st.session_state.df.head(), use_container_width=True)

# ── Reset ──────────────────────────────────────────────────────────────────────
st.divider()
if st.button("🔄 Reset All Preprocessing"):
    st.session_state.df = df_raw.copy()
    st.session_state.steps_done = []
    st.session_state.model = None
    st.session_state.feature_cols = None
    st.session_state.scaler = None
    st.session_state.example = None
    st.rerun()

# ── STEP 7: Model Training ────────────────────────────────────────────────────
st.divider()
st.header("7️⃣ Model Training & Evaluation")

required = {"fill_income","create_age","create_spending","drop_cols","encode","scale"}
if not required.issubset(set(st.session_state.steps_done)):
    st.warning("⚠️ Complete all preprocessing steps above first.")
else:
    model_choice = st.selectbox("Select Model", ["Logistic Regression", "Random Forest", "XGBoost"])

    if st.button("Train & Evaluate"):
        df_model = st.session_state.df
        y = df_model["Response"]
        X = df_model.drop("Response", axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if model_choice == "Logistic Regression":
            model = LogisticRegression(C=0.5, penalty="l2", class_weight="balanced", max_iter=1000)
        elif model_choice == "Random Forest":
            model = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight="balanced", random_state=42)
        else:
            from xgboost import XGBClassifier
            model = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.5, random_state=42, eval_metric="logloss")

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        # save model + feature columns for prediction
        st.session_state.model = model
        st.session_state.feature_cols = list(X.columns)

        acc = accuracy_score(y_test, preds)
        f1  = f1_score(y_test, preds)
        cm  = confusion_matrix(y_test, preds)

        col_a, col_b = st.columns(2)
        col_a.metric("Accuracy", f"{acc:.4f}")
        col_b.metric("F1 Score", f"{f1:.4f}")

        st.subheader("Confusion Matrix")
        cm_df = pd.DataFrame(
            cm,
            index=["Actual: No Response", "Actual: Responded"],
            columns=["Pred: No Response", "Pred: Responded"]
        )
        st.dataframe(cm_df, use_container_width=True)

# ── STEP 8: Single Customer Prediction ────────────────────────────────────────
st.divider()
st.header("8️⃣ Predict for a Single Customer")

if st.session_state.model is None:
    st.warning("⚠️ Train a model first (Step 7).")
else:
    st.write("Fill in the customer details below and click **Predict**.")

    # ── Example buttons ──
    ex1, ex2, _ = st.columns([1, 1, 4])
    if ex1.button("📥 Example: Will Respond"):
        st.session_state.example = "respond"
        st.rerun()
    if ex2.button("📥 Example: Won't Respond"):
        st.session_state.example = "no_respond"
        st.rerun()

    # set values based on stored example
    if st.session_state.example == "respond":
        default_income, default_age, default_recency = 90000, 50, 5
        default_spending, default_web, default_kid   = 1800,  2,  0
        default_teen, default_edu, default_marital   = 0, "PhD", "Single"
    elif st.session_state.example == "no_respond":
        default_income, default_age, default_recency = 20000, 35, 90
        default_spending, default_web, default_kid   = 80,    8,  2
        default_teen, default_edu, default_marital   = 1, "Basic", "Married"
    else:
        default_income, default_age, default_recency = 55000, 40, 30
        default_spending, default_web, default_kid   = 500,   5,  1
        default_teen, default_edu, default_marital   = 0, "Graduation", "Single"

    c1, c2, c3 = st.columns(3)
    income         = c1.number_input("Income",           min_value=0,  max_value=200000, value=default_income)
    age            = c2.number_input("Age",              min_value=18, max_value=100,    value=default_age)
    recency        = c3.number_input("Recency (days)",   min_value=0,  max_value=100,    value=default_recency)

    c4, c5, c6 = st.columns(3)
    total_spending  = c4.number_input("Total Spending",    min_value=0,    max_value=3000,   value=default_spending)
    web_visits      = c5.number_input("Web Visits/Month",  min_value=0,    max_value=20,     value=default_web)
    kidhome         = c6.number_input("Kids at Home",      min_value=0,    max_value=3,      value=default_kid)

    c7, c8, c9 = st.columns(3)
    teenhome        = c7.number_input("Teens at Home",     min_value=0,    max_value=3,      value=default_teen)
    education       = c8.selectbox("Education",  ["Graduation", "PhD", "Master", "Basic", "2n Cycle"], index=["Graduation", "PhD", "Master", "Basic", "2n Cycle"].index(default_edu))
    marital_status  = c9.selectbox("Marital Status", ["Single", "Married", "Together", "Divorced", "Widow", "Alone", "Absurd", "YOLO"], index=["Single", "Married", "Together", "Divorced", "Widow", "Alone", "Absurd", "YOLO"].index(default_marital))

    if st.button("🔮 Predict"):
        # use the same scaler fitted during Step 6
        num_cols = ["Income", "Age", "Recency", "Total_Spending", "NumWebVisitsMonth"]
        raw_num = np.array([[income, age, recency, total_spending, web_visits]], dtype=float)
        scaled_num = st.session_state.scaler.transform(raw_num)[0]

        # build a zero-filled row matching all feature columns
        row = dict.fromkeys(st.session_state.feature_cols, 0)

        # fill scaled numerics
        for col_name, val in zip(num_cols, scaled_num):
            if col_name in row:
                row[col_name] = val

        # fill remaining simple numerics (unscaled)
        for col_name, val in [("Kidhome", kidhome), ("Teenhome", teenhome)]:
            if col_name in row:
                row[col_name] = val

        # fill one-hot encoded education
        edu_map = {
            "PhD":      "Education_PhD",
            "Master":   "Education_Master",
            "Basic":    "Education_Basic",
            "2n Cycle": "Education_2n Cycle",
        }
        if education in edu_map and edu_map[education] in row:
            row[edu_map[education]] = 1

        # fill one-hot encoded marital status
        marital_map = {
            "Married":  "Marital_Status_Married",
            "Single":   "Marital_Status_Single",
            "Together": "Marital_Status_Together",
            "Divorced": "Marital_Status_Divorced",
            "Widow":    "Marital_Status_Widow",
            "Alone":    "Marital_Status_Alone",
            "Absurd":   "Marital_Status_Absurd",
            "YOLO":     "Marital_Status_YOLO",
        }
        if marital_status in marital_map and marital_map[marital_status] in row:
            row[marital_map[marital_status]] = 1

        input_df = pd.DataFrame([row])
        prediction = st.session_state.model.predict(input_df)[0]
        proba      = st.session_state.model.predict_proba(input_df)[0]

        st.subheader("Prediction Result")
        if prediction == 1:
            st.success(f"✅ This customer is likely to **RESPOND** to the campaign!  (confidence: {proba[1]*100:.1f}%)")
        else:
            st.error(f"❌ This customer is likely to **NOT respond** to the campaign.  (confidence: {proba[0]*100:.1f}%)")
