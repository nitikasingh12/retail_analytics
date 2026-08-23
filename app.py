"""
app.py
-------
Streamlit dashboard for the Retail Sales & Customer Analytics project.

Run locally:   streamlit run app.py
Deploy:        push this repo to GitHub -> streamlit.io/cloud -> point at app.py
               (requirements.txt already lists the exact dependencies needed)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

sns.set_theme(style="whitegrid", palette="deep")

st.set_page_config(page_title="Retail Sales & Customer Analytics", layout="wide")


# ---------------------------------------------------------------------------
# DATA LOADING (cached so re-runs on filter change are instant)
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/cleaned_transactions.csv", parse_dates=["OrderDate"])
    rfm = pd.read_csv("data/rfm_table.csv")
    retention = pd.read_csv("data/cohort_retention.csv", index_col=0)
    return df, rfm, retention


df, rfm, retention = load_data()

st.title("🛒 Retail Sales & Customer Analytics Dashboard")
st.caption("End-to-end analysis: sales performance, RFM customer segmentation, and cohort retention.")

# ---------------------------------------------------------------------------
# SIDEBAR FILTERS
# ---------------------------------------------------------------------------
st.sidebar.header("Filters")

date_min, date_max = df["OrderDate"].min().date(), df["OrderDate"].max().date()
date_range = st.sidebar.date_input("Order date range", value=(date_min, date_max), min_value=date_min, max_value=date_max)

categories = st.sidebar.multiselect("Category", sorted(df["Category"].unique()), default=sorted(df["Category"].unique()))
regions = st.sidebar.multiselect("Region", sorted(df["Region"].unique()), default=sorted(df["Region"].unique()))

if len(date_range) == 2:
    start, end = date_range
else:
    start, end = date_min, date_max

mask = (
    (df["OrderDate"].dt.date >= start)
    & (df["OrderDate"].dt.date <= end)
    & (df["Category"].isin(categories))
    & (df["Region"].isin(regions))
)
fdf = df[mask]

if fdf.empty:
    st.warning("No data for the selected filters. Adjust the filters in the sidebar.")
    st.stop()

# ---------------------------------------------------------------------------
# KPI ROW
# ---------------------------------------------------------------------------
total_revenue = fdf["NetSales"].sum()
total_orders = fdf["OrderID"].nunique()
total_customers = fdf["CustomerID"].nunique()
aov = total_revenue / total_orders if total_orders else 0

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Revenue", f"₹{total_revenue:,.0f}")
k2.metric("Total Orders", f"{total_orders:,}")
k3.metric("Unique Customers", f"{total_customers:,}")
k4.metric("Avg Order Value", f"₹{aov:,.0f}")

st.divider()

# ---------------------------------------------------------------------------
# TABS
# ---------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📈 Sales Overview", "🧩 RFM Segmentation", "🔁 Cohort Retention"])

with tab1:
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Monthly Revenue Trend")
        monthly = fdf.groupby(fdf["OrderDate"].dt.to_period("M").astype(str))["NetSales"].sum().reset_index()
        monthly.columns = ["Month", "NetSales"]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        sns.lineplot(data=monthly, x="Month", y="NetSales", marker="o", ax=ax)
        ax.set_ylabel("Net Sales (₹)")
        plt.xticks(rotation=60)
        st.pyplot(fig)

    with c2:
        st.subheader("Revenue by Category")
        cat_rev = fdf.groupby("Category")["NetSales"].sum().sort_values(ascending=False).reset_index()
        fig, ax = plt.subplots(figsize=(7, 4.5))
        sns.barplot(data=cat_rev, x="NetSales", y="Category", ax=ax)
        ax.set_xlabel("Net Sales (₹)")
        st.pyplot(fig)

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Revenue by Region")
        region_rev = fdf.groupby("Region")["NetSales"].sum().sort_values(ascending=False).reset_index()
        fig, ax = plt.subplots(figsize=(7, 4.5))
        sns.barplot(data=region_rev, x="Region", y="NetSales", ax=ax)
        ax.set_ylabel("Net Sales (₹)")
        st.pyplot(fig)

    with c4:
        st.subheader("Orders by Payment Method")
        fig, ax = plt.subplots(figsize=(7, 4.5))
        fdf["PaymentMethod"].value_counts().plot.pie(autopct="%1.1f%%", ylabel="", ax=ax)
        st.pyplot(fig)

    st.subheader("Top 10 Products by Revenue")
    top_products = fdf.groupby("Product")["NetSales"].sum().sort_values(ascending=False).head(10).reset_index()
    st.dataframe(top_products, use_container_width=True, hide_index=True)

with tab2:
    st.subheader("What is RFM?")
    st.markdown(
        "Each customer is scored 1-4 on **Recency** (days since last order, lower = better), "
        "**Frequency** (number of orders), and **Monetary** (total spend), then grouped into "
        "actionable segments."
    )

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Customers by Segment")
        seg_counts = rfm["Segment"].value_counts().reset_index()
        seg_counts.columns = ["Segment", "Count"]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        sns.barplot(data=seg_counts, x="Count", y="Segment",
                    order=seg_counts.sort_values("Count", ascending=False)["Segment"], ax=ax)
        st.pyplot(fig)

    with c2:
        st.subheader("Revenue by Segment")
        seg_rev = rfm.groupby("Segment")["Monetary"].sum().sort_values(ascending=False).reset_index()
        fig, ax = plt.subplots(figsize=(7, 4.5))
        sns.barplot(data=seg_rev, x="Monetary", y="Segment", ax=ax)
        ax.set_xlabel("Total Revenue (₹)")
        st.pyplot(fig)

    st.subheader("Explore Customers in a Segment")
    seg_choice = st.selectbox("Choose a segment", sorted(rfm["Segment"].unique()))
    st.dataframe(
        rfm[rfm["Segment"] == seg_choice][["CustomerID", "Recency", "Frequency", "Monetary", "RFM_Score"]]
        .sort_values("Monetary", ascending=False),
        use_container_width=True, hide_index=True,
    )

with tab3:
    st.subheader("Monthly Cohort Retention (%)")
    st.markdown("Each row is a group of customers acquired in the same month; each column shows what % of that cohort was still ordering N months later.")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(retention, annot=True, fmt=".0f", cmap="YlGnBu", cbar_kws={"label": "Retention %"}, ax=ax)
    ax.set_xlabel("Months Since Acquisition")
    ax.set_ylabel("Acquisition Cohort")
    st.pyplot(fig)

st.divider()
st.caption("Built with Python, Pandas, NumPy, Matplotlib, Seaborn & Streamlit.")
