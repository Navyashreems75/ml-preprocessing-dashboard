
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Page Title
st.title("Customer Dashboard")

# Load Data
df = pd.read_csv("data.csv")

# ---------------------------
# KPI SECTION
# ---------------------------

st.header("Summary")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Customers", len(df))
col2.metric("Average Age", round(df["Age"].mean(), 1))
col3.metric("Average Salary", int(df["Salary"].mean()))
col4.metric("Total Purchased", df[df["Purchased"] == "Yes"].shape[0])

st.divider()

# ---------------------------
# CHART SECTION
# ---------------------------

st.header("Purchase Distribution")

fig1, ax1 = plt.subplots()
df["Purchased"].value_counts().plot(kind="bar", ax=ax1)
ax1.set_xlabel("Purchased")
ax1.set_ylabel("Count")

st.pyplot(fig1)

st.divider()

st.header("Age vs Salary")

fig2, ax2 = plt.subplots()
ax2.scatter(df["Age"], df["Salary"])
ax2.set_xlabel("Age")
ax2.set_ylabel("Salary")

st.pyplot(fig2)

st.divider()

st.header("Data Table")
st.dataframe(df)