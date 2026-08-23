# Retail Sales & Customer Analytics Dashboard

An end-to-end data analytics project simulating a real e-commerce retailer's transaction
data — cleaning, exploratory analysis, RFM customer segmentation, and cohort retention
analysis, delivered as an interactive Streamlit dashboard.

**Why this project:** RFM segmentation and cohort retention are two of the most
commonly asked-about analyses in Data Analyst interviews (Amazon, Flipkart, Myntra,
Swiggy, and most GCC analytics teams) because they show business-facing thinking, not
just charting. This project builds both from first principles using only Pandas and
NumPy — no black-box ML library.

## Tech Stack
- **Python** — core logic
- **Pandas / NumPy** — data cleaning, aggregation, RFM scoring
- **Matplotlib / Seaborn** — static chart generation (`analysis.py`)
- **Streamlit** — interactive dashboard (`app.py`)

## Project Structure
```
retail_analytics/
├── data/
│   ├── generate_data.py        # synthetic dataset generator (realistic messiness)
│   ├── ecommerce_transactions.csv
│   ├── cleaned_transactions.csv  # output of analysis.py
│   ├── rfm_table.csv             # output of analysis.py
│   └── cohort_retention.csv      # output of analysis.py
├── charts/                     # static PNG charts from analysis.py
├── analysis.py                 # full analysis pipeline (run this first)
├── app.py                      # Streamlit dashboard
├── requirements.txt
└── README.md
```

## How to Run Locally
```bash
pip install -r requirements.txt

# 1. (Optional — a dataset is already included) regenerate synthetic data
python data/generate_data.py

# 2. Run the analysis pipeline — cleans data, produces charts + RFM + cohort tables
python analysis.py

# 3. Launch the dashboard
streamlit run app.py
```

## Methodology

### 1. Data Cleaning
- Standardized inconsistent category casing (`electronics` → `Electronics`)
- Removed duplicate `OrderID`s (simulated double-submitted orders)
- Removed rows with non-positive `NetSales` (refund/entry-error artifacts)
- Imputed missing `Region` / `PaymentMethod` with mode, with counts logged for
  transparency (a real analyst documents every imputation decision)

### 2. RFM Segmentation (no sklearn — pure quantile logic)
- **Recency**: days since each customer's last order
- **Frequency**: number of distinct orders
- **Monetary**: total spend
- Each dimension scored 1–4 via `pd.qcut` quartiles, summed into an `RFM_Score`,
  then mapped to business segments: *Champions, Loyal Customers, Potential Loyalist,
  At Risk, Hibernating, Needs Attention*

### 3. Cohort Retention Analysis
- Customers grouped into monthly acquisition cohorts (by first purchase month)
- Tracks what % of each cohort is still transacting N months later
- Visualized as a retention heatmap — the standard format for churn/retention
  discussions in interviews

## Dashboard Features
- Sidebar filters: date range, category, region (all charts react live)
- KPI cards: total revenue, orders, unique customers, average order value
- Three tabs: Sales Overview, RFM Segmentation (with a per-segment customer explorer),
  and Cohort Retention heatmap

  Deployment link: https://retailanalytics-ktw3tbknpcmcxkb2fycxpk.streamlit.app/

## Deployment
This is a **Streamlit** app (uses `st.cache_data`, `st.pyplot`, interactive widgets),
so it deploys natively on **Streamlit Community Cloud**:
1. Push this folder to a GitHub repo
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app → point at `app.py`
3. Streamlit installs `requirements.txt` automatically and deploys

> Note: this app is not a static site (it runs Python server-side per interaction),
> so it isn't a fit for Vercel's static/serverless hosting model — Streamlit Cloud,
> Render, or Hugging Face Spaces are the right targets for this stack.

## Possible Extensions (good talking points for interviews)
- Swap the synthetic dataset for a real one (Kaggle's Online Retail II dataset works
  with the same schema with minor column renaming)
- Add basket/market association analysis (which products are bought together)
- Add a forecasting tab (moving average or exponential smoothing on monthly revenue)

- Deployment issue
- Link: https://retailanalytics-ktw3tbknpcmcxkb2fycxpk.streamlit.app/
