"""
analysis.py
------------
End-to-end analysis pipeline for the Retail Sales & Customer Analytics project.

Stages:
 1. Load + clean raw transactions
 2. Exploratory Data Analysis (sales trends, category/region performance)
 3. RFM Segmentation (Recency, Frequency, Monetary) -- quantile-based, no sklearn
 4. Cohort Retention Analysis (monthly acquisition cohorts)
 5. Saves all charts to charts/ and a cleaned dataset + RFM table to data/

Run: python analysis.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.figsize"] = (10, 6)

CHART_DIR = "charts"
DATA_DIR = "data"


# ---------------------------------------------------------------------------
# 1. LOAD + CLEAN
# ---------------------------------------------------------------------------
def load_and_clean(path=f"{DATA_DIR}/ecommerce_transactions.csv"):
    df = pd.read_csv(path)
    print(f"Raw rows: {len(df)}")

    # Standardize category casing/spacing
    df["Category"] = df["Category"].str.strip().str.title()

    # Parse dates
    df["OrderDate"] = pd.to_datetime(df["OrderDate"], errors="coerce")

    # Drop duplicate orders (same OrderID)
    before = len(df)
    df = df.drop_duplicates(subset="OrderID")
    print(f"Dropped {before - len(df)} duplicate orders")

    # Drop rows with negative/zero NetSales (bad data / refund artifacts)
    before = len(df)
    df = df[df["NetSales"] > 0]
    print(f"Dropped {before - len(df)} rows with non-positive NetSales")

    # Impute missing categorical fields with mode (documented assumption)
    for col in ["Region", "PaymentMethod"]:
        mode_val = df[col].mode()[0]
        n_missing = df[col].isna().sum()
        df[col] = df[col].fillna(mode_val)
        print(f"Filled {n_missing} missing '{col}' values with mode '{mode_val}'")

    # Drop rows with unparseable dates
    df = df.dropna(subset=["OrderDate"])

    df["OrderMonth"] = df["OrderDate"].dt.to_period("M").astype(str)
    df["OrderYear"] = df["OrderDate"].dt.year

    print(f"Clean rows: {len(df)}")
    df.to_csv(f"{DATA_DIR}/cleaned_transactions.csv", index=False)
    return df


# ---------------------------------------------------------------------------
# 2. EXPLORATORY DATA ANALYSIS
# ---------------------------------------------------------------------------
def run_eda(df):
    # -- Monthly revenue trend --
    monthly = df.groupby("OrderMonth")["NetSales"].sum().reset_index()
    plt.figure()
    sns.lineplot(data=monthly, x="OrderMonth", y="NetSales", marker="o")
    plt.xticks(rotation=60)
    plt.title("Monthly Revenue Trend")
    plt.ylabel("Net Sales (₹)")
    plt.tight_layout()
    plt.savefig(f"{CHART_DIR}/monthly_revenue_trend.png", dpi=150)
    plt.close()

    # -- Revenue by category --
    cat_rev = df.groupby("Category")["NetSales"].sum().sort_values(ascending=False).reset_index()
    plt.figure()
    sns.barplot(data=cat_rev, x="NetSales", y="Category")
    plt.title("Revenue by Product Category")
    plt.xlabel("Net Sales (₹)")
    plt.tight_layout()
    plt.savefig(f"{CHART_DIR}/revenue_by_category.png", dpi=150)
    plt.close()

    # -- Revenue by region --
    region_rev = df.groupby("Region")["NetSales"].sum().sort_values(ascending=False).reset_index()
    plt.figure()
    sns.barplot(data=region_rev, x="Region", y="NetSales")
    plt.title("Revenue by Region")
    plt.ylabel("Net Sales (₹)")
    plt.tight_layout()
    plt.savefig(f"{CHART_DIR}/revenue_by_region.png", dpi=150)
    plt.close()

    # -- Payment method distribution --
    plt.figure()
    df["PaymentMethod"].value_counts().plot.pie(autopct="%1.1f%%", ylabel="")
    plt.title("Orders by Payment Method")
    plt.tight_layout()
    plt.savefig(f"{CHART_DIR}/payment_method_share.png", dpi=150)
    plt.close()

    print("EDA charts saved.")
    return {"monthly": monthly, "cat_rev": cat_rev, "region_rev": region_rev}


# ---------------------------------------------------------------------------
# 3. RFM SEGMENTATION (quantile-based, no sklearn)
# ---------------------------------------------------------------------------
def rfm_segmentation(df):
    snapshot_date = df["OrderDate"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("CustomerID").agg(
        Recency=("OrderDate", lambda x: (snapshot_date - x.max()).days),
        Frequency=("OrderID", "nunique"),
        Monetary=("NetSales", "sum"),
    ).reset_index()

    # Score 1 (worst) to 4 (best) using quartiles.
    # Recency is reverse-scored: lower recency (more recent) = higher score.
    rfm["R_Score"] = pd.qcut(rfm["Recency"], 4, labels=[4, 3, 2, 1]).astype(int)
    rfm["F_Score"] = pd.qcut(rfm["Frequency"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
    rfm["M_Score"] = pd.qcut(rfm["Monetary"], 4, labels=[1, 2, 3, 4]).astype(int)

    rfm["RFM_Score"] = rfm["R_Score"] + rfm["F_Score"] + rfm["M_Score"]

    def segment(row):
        r, f, m = row["R_Score"], row["F_Score"], row["M_Score"]
        if r >= 4 and f >= 4:
            return "Champions"
        elif r >= 3 and f >= 3:
            return "Loyal Customers"
        elif r >= 3 and f <= 2:
            return "Potential Loyalist"
        elif r <= 2 and f >= 3:
            return "At Risk"
        elif r <= 2 and f <= 2 and m <= 2:
            return "Hibernating"
        else:
            return "Needs Attention"

    rfm["Segment"] = rfm.apply(segment, axis=1)
    rfm.to_csv(f"{DATA_DIR}/rfm_table.csv", index=False)

    # -- Segment distribution chart --
    seg_counts = rfm["Segment"].value_counts().reset_index()
    seg_counts.columns = ["Segment", "Count"]
    plt.figure()
    sns.barplot(data=seg_counts, x="Count", y="Segment", order=seg_counts.sort_values("Count", ascending=False)["Segment"])
    plt.title("Customer Count by RFM Segment")
    plt.tight_layout()
    plt.savefig(f"{CHART_DIR}/rfm_segment_distribution.png", dpi=150)
    plt.close()

    # -- Segment revenue contribution --
    seg_rev = rfm.groupby("Segment")["Monetary"].sum().sort_values(ascending=False).reset_index()
    plt.figure()
    sns.barplot(data=seg_rev, x="Monetary", y="Segment")
    plt.title("Revenue Contribution by RFM Segment")
    plt.xlabel("Total Revenue (₹)")
    plt.tight_layout()
    plt.savefig(f"{CHART_DIR}/rfm_segment_revenue.png", dpi=150)
    plt.close()

    print("RFM segmentation complete. Segment counts:")
    print(seg_counts.to_string(index=False))
    return rfm


# ---------------------------------------------------------------------------
# 4. COHORT RETENTION ANALYSIS
# ---------------------------------------------------------------------------
def cohort_analysis(df):
    d = df.copy()
    d["OrderPeriod"] = d["OrderDate"].dt.to_period("M")
    d["CohortMonth"] = d.groupby("CustomerID")["OrderDate"].transform("min").dt.to_period("M")

    d["CohortIndex"] = (
        (d["OrderPeriod"].dt.year - d["CohortMonth"].dt.year) * 12
        + (d["OrderPeriod"].dt.month - d["CohortMonth"].dt.month)
    )

    cohort_data = d.groupby(["CohortMonth", "CohortIndex"])["CustomerID"].nunique().reset_index()
    cohort_pivot = cohort_data.pivot(index="CohortMonth", columns="CohortIndex", values="CustomerID")

    cohort_size = cohort_pivot.iloc[:, 0]
    retention = cohort_pivot.divide(cohort_size, axis=0).round(3) * 100

    plt.figure(figsize=(12, 8))
    sns.heatmap(retention, annot=True, fmt=".0f", cmap="YlGnBu", cbar_kws={"label": "Retention %"})
    plt.title("Monthly Cohort Retention (%)")
    plt.xlabel("Months Since Acquisition")
    plt.ylabel("Acquisition Cohort")
    plt.tight_layout()
    plt.savefig(f"{CHART_DIR}/cohort_retention_heatmap.png", dpi=150)
    plt.close()

    retention.to_csv(f"{DATA_DIR}/cohort_retention.csv")
    print("Cohort retention analysis complete.")
    return retention


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = load_and_clean()
    run_eda(df)
    rfm_segmentation(df)
    cohort_analysis(df)
    print("\nAll charts saved to charts/, cleaned data + RFM + cohort tables saved to data/")
