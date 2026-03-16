"""
Lab 2 — Data Pipeline: Retail Sales Analysis
Module 2 — Programming for AI & Data Science

Complete each function below. Remove the TODO: comments and pass statements
as you implement each function. Do not change the function signatures.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ─── Configuration ────────────────────────────────────────────────────────────

DATA_PATH = 'data/sales_records.csv'
OUTPUT_DIR = 'output'


# ─── Pipeline Functions ───────────────────────────────────────────────────────

def load_data(filepath):
    """Load sales records from CSV. Returns a DataFrame."""
    
    df = pd.read_csv(filepath)
    
    print(f"Loaded {len(df)} records from {filepath}")
    
    return df



def clean_data(df):
    """Handle missing values and fix data types. Returns a cleaned DataFrame."""
    
    df_clean = df.copy()

    # fill missing values with median
    df_clean['quantity'] = df_clean['quantity'].fillna(df_clean['quantity'].median())
    df_clean['unit_price'] = df_clean['unit_price'].fillna(df_clean['unit_price'].median())

    # convert date column
    df_clean['date'] = pd.to_datetime(df_clean['date'], errors='coerce')

    # drop rows where both quantity and unit_price are still missing
    df_clean = df_clean.dropna(subset=['quantity', 'unit_price'], how='all')

    print(f"Cleaned data: {len(df_clean)} records")
    return df_clean


def add_features(df):
    """Add revenue and day_of_week columns. Returns an enriched DataFrame."""
    
    df_enriched = df.copy()

    # create revenue column
    df_enriched['revenue'] = df_enriched['quantity'] * df_enriched['unit_price']

    # create day_of_week column
    df_enriched['day_of_week'] = df_enriched['date'].dt.day_name()

    return df_enriched


def generate_summary(df):
    """Compute summary statistics. Returns a dict."""
    
    summary = {
        'total_revenue': df['revenue'].sum(),
        'avg_order_value': df['revenue'].mean(),
        'top_category': df.groupby('product_category')['revenue'].sum().idxmax(),
        'record_count': len(df)
    }
    
    return summary

def create_visualizations(df, output_dir="output"):
    """Create and save 3 charts to output_dir."""
    
    os.makedirs(output_dir, exist_ok=True)

    # Chart 1: total revenue by product category
    revenue_by_category = df.groupby('product_category')['revenue'].sum()

    fig, ax = plt.subplots()
    revenue_by_category.plot(kind='bar', ax=ax)
    ax.set_title("Total Revenue by Product Category")
    ax.set_ylabel("Revenue")

    fig.savefig(f"{output_dir}/revenue_by_category.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Chart 2: daily revenue trend
    daily_revenue = df.groupby('date')['revenue'].sum()

    fig, ax = plt.subplots()
    daily_revenue.plot(kind='line', ax=ax)
    ax.set_title("Daily Revenue Trend")
    ax.set_ylabel("Revenue")

    fig.savefig(f"{output_dir}/daily_revenue_trend.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Chart 3: average order value by payment method
    avg_order = df.groupby('payment_method')['revenue'].mean()

    fig, ax = plt.subplots()
    avg_order.plot(kind='barh', ax=ax)
    ax.set_title("Average Order Value by Payment Method")
    ax.set_xlabel("Average Revenue")

    fig.savefig(f"{output_dir}/avg_order_by_payment.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
   
def main():
    """Run the full pipeline end-to-end."""
    
    df = load_data("data/sales_records.csv")

    df = clean_data(df)

    df = add_features(df)

    summary = generate_summary(df)

    print("Summary statistics:")
    print(summary)

    create_visualizations(df)

    print("Pipeline complete.")


if __name__ == "__main__":
    main()
