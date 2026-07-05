#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@license
SPDX-License-Identifier: Apache-2.0
"""

import os
import pandas as pd
import numpy as np

def run_pandas_analysis():
    print("=========================================================")
    print(" AUTOMATED DATA SCIENCE & ANALYTICS PIPELINE (PANDAS)    ")
    print("=========================================================")
    
    # Resolve paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    customers_path = os.path.join(base_dir, "data", "customers.csv")
    orders_path = os.path.join(base_dir, "data", "orders.csv")
    queries_path = os.path.join(base_dir, "data", "support_queries.csv")
    
    if not os.path.exists(customers_path):
        # Fallback to parent path check
        customers_path = os.path.join(os.getcwd(), "data", "customers.csv")
        orders_path = os.path.join(os.getcwd(), "data", "orders.csv")
        queries_path = os.path.join(os.getcwd(), "data", "support_queries.csv")

    if not os.path.exists(customers_path):
        print("Error: CSV datasets not found. Please run the server/generator first.")
        return

    # 1. Load datasets
    df_customers = pd.read_csv(customers_path)
    df_orders = pd.read_csv(orders_path)
    df_queries = pd.read_csv(queries_path)
    
    print(f"Loaded {len(df_customers):,} customer records successfully.")
    print(f"Loaded {len(df_orders):,} order records successfully.")
    print(f"Loaded {len(df_queries):,} support queries successfully.")
    print("---------------------------------------------------------")
    
    # 2. Segments distribution
    print("\n[1] Customer Segments Distribution:")
    segment_counts = df_customers["Customer Segment"].value_counts()
    for seg, count in segment_counts.items():
        pct = (count / len(df_customers)) * 100
        print(f"  - {seg}: {count:,} ({pct:.2f}%)")
        
    # 3. Join Customers and Orders to find Segment-level purchasing metrics
    print("\n[2] Join Analysis - Purchasing Behavior by Customer Segment:")
    df_cust_orders = pd.merge(df_orders, df_customers, on="Customer ID")
    segment_spend = df_cust_orders.groupby("Customer Segment")["Order Amount"].agg(["count", "sum", "mean"])
    print(segment_spend)
    
    # 4. Product categories popularity and revenue
    print("\n[3] Product Performance Matrix:")
    prod_perf = df_orders.groupby("Product Name")["Order Amount"].agg(["count", "sum"]).sort_values(by="sum", ascending=False)
    print(prod_perf)
    
    # 5. Queries category vs. Sentiment cross-tabulation
    print("\n[4] Sentiment vs. Category Crosstab (NLP Alignment):")
    sentiment_crosstab = pd.crosstab(df_queries["Query Category"], df_queries["Sentiment"])
    print(sentiment_crosstab)
    
    # 6. Resolution Rates & Escalations
    print("\n[5] Ticket Escalation & Resolution Efficiency:")
    res_status = df_queries["Resolution Status"].value_counts()
    for res, count in res_status.items():
        pct = (count / len(df_queries)) * 100
        print(f"  - {res}: {count:,} ({pct:.2f}%)")
        
    escalation_rate = (df_queries["Escalation Status"] == "Yes").mean() * 100
    print(f"  - Overall Escalation Rate: {escalation_rate:.2f}%")
    print("=========================================================")

if __name__ == "__main__":
    run_pandas_analysis()
