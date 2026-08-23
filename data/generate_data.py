"""
generate_data.py
-----------------
Generates a realistic synthetic e-commerce transactions dataset.
Includes intentional real-world messiness (nulls, duplicates, inconsistent
casing) so the cleaning step in analysis.py is genuine work, not decoration.

Run: python data/generate_data.py
Output: data/ecommerce_transactions.csv
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

N_CUSTOMERS = 1200
N_TRANSACTIONS = 9000
START_DATE = datetime(2023, 1, 1)
END_DATE = datetime(2024, 12, 31)

categories = {
    "Electronics": ["Wireless Earbuds", "Smartwatch", "Bluetooth Speaker", "Power Bank", "Laptop Sleeve"],
    "Fashion": ["Cotton T-Shirt", "Denim Jacket", "Running Shoes", "Leather Wallet", "Sunglasses"],
    "Home & Kitchen": ["Non-stick Pan", "LED Lamp", "Storage Organizer", "Water Bottle", "Cushion Cover"],
    "Beauty": ["Face Serum", "Lip Balm Set", "Hair Dryer", "Sunscreen SPF50", "Perfume"],
    "Sports": ["Yoga Mat", "Dumbbell Set", "Cricket Bat", "Resistance Bands", "Cycling Gloves"],
    "Books": ["Fiction Novel", "Self-Help Book", "Cookbook", "Business Biography", "Kids Storybook"],
}

regions = ["North", "South", "East", "West", "Central"]
payment_methods = ["Credit Card", "Debit Card", "UPI", "Net Banking", "Cash on Delivery"]

# Customer signup dates (acquisition cohort) and a "loyalty tier" to bias behavior
customer_ids = [f"CUST{str(i).zfill(5)}" for i in range(1, N_CUSTOMERS + 1)]
signup_dates = pd.to_datetime(
    np.random.choice(pd.date_range(START_DATE, END_DATE - timedelta(days=30), freq="D"), N_CUSTOMERS)
)
customer_df = pd.DataFrame({"CustomerID": customer_ids, "SignupDate": signup_dates})
# Some customers are naturally more active (power-law-ish activity)
customer_weight = np.random.pareto(a=2.0, size=N_CUSTOMERS) + 0.2
customer_weight = customer_weight / customer_weight.sum()

rows = []
for i in range(N_TRANSACTIONS):
    cust_idx = np.random.choice(N_CUSTOMERS, p=customer_weight)
    cust_id = customer_ids[cust_idx]
    signup = signup_dates[cust_idx]

    # transaction date must be on/after signup
    days_range = (END_DATE - signup).days
    if days_range <= 0:
        days_range = 1
    txn_date = signup + timedelta(days=int(np.random.exponential(scale=days_range / 3)) % days_range)

    category = np.random.choice(list(categories.keys()), p=[0.28, 0.22, 0.18, 0.12, 0.12, 0.08])
    product = np.random.choice(categories[category])

    base_prices = {
        "Electronics": (800, 4500), "Fashion": (300, 2500), "Home & Kitchen": (200, 2000),
        "Beauty": (150, 1500), "Sports": (250, 3000), "Books": (150, 800),
    }
    low, high = base_prices[category]
    unit_price = round(np.random.uniform(low, high), 2)
    quantity = np.random.choice([1, 1, 1, 2, 2, 3, 4], p=[0.35, 0.2, 0.15, 0.15, 0.08, 0.05, 0.02])
    discount_pct = np.random.choice([0, 0, 5, 10, 15, 20, 30], p=[0.35, 0.15, 0.15, 0.15, 0.1, 0.07, 0.03])

    gross = unit_price * quantity
    net_sales = round(gross * (1 - discount_pct / 100), 2)

    rows.append({
        "OrderID": f"ORD{100000 + i}",
        "CustomerID": cust_id,
        "OrderDate": txn_date.strftime("%Y-%m-%d"),
        "Category": category,
        "Product": product,
        "Region": np.random.choice(regions),
        "PaymentMethod": np.random.choice(payment_methods, p=[0.28, 0.2, 0.32, 0.1, 0.1]),
        "Quantity": quantity,
        "UnitPrice": unit_price,
        "DiscountPct": discount_pct,
        "NetSales": net_sales,
    })

df = pd.DataFrame(rows)

# ---- Inject realistic messiness ----
# 1. Some missing Region / PaymentMethod values
mask_null_region = np.random.rand(len(df)) < 0.02
df.loc[mask_null_region, "Region"] = np.nan
mask_null_pay = np.random.rand(len(df)) < 0.015
df.loc[mask_null_pay, "PaymentMethod"] = np.nan

# 2. Inconsistent category casing/spacing for a subset of rows
messy_idx = df.sample(frac=0.03, random_state=1).index
df.loc[messy_idx, "Category"] = df.loc[messy_idx, "Category"].str.lower()

# 3. Duplicate rows (simulate double-submitted orders)
dupes = df.sample(frac=0.01, random_state=2)
df = pd.concat([df, dupes], ignore_index=True)

# 4. A few negative/zero NetSales from data entry errors (refund artifacts)
err_idx = df.sample(frac=0.005, random_state=3).index
df.loc[err_idx, "NetSales"] = -abs(df.loc[err_idx, "NetSales"])

# 5. Shuffle rows
df = df.sample(frac=1, random_state=7).reset_index(drop=True)

df.to_csv("data/ecommerce_transactions.csv", index=False)
customer_df.to_csv("data/customers.csv", index=False)

print(f"Generated {len(df)} transaction rows across {N_CUSTOMERS} customers.")
print(df.head())
