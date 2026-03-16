"""
Lab 2 — Learner Test File

Write your own pytest tests here. You must implement at least 3 test functions:
  - test_load_data_returns_dataframe
  - test_clean_data_no_nulls
  - test_add_features_creates_revenue

The autograder will run your tests as part of the CI check.
"""

import pandas as pd
import numpy as np
import pytest
from pipeline import load_data, clean_data, add_features


# ─── Test 1 ───────────────────────────────────────────────────────────────────

def test_load_data_returns_dataframe():
    """load_data returns a DataFrame with expected columns."""

    df = load_data("data/sales_records.csv")

    # check type
    assert isinstance(df, pd.DataFrame)

    # check not empty
    assert len(df) > 0

    # check expected columns
    expected_columns = ['date', 'store_id', 'product_category', 'quantity', 'unit_price']
    for col in expected_columns:
        assert col in df.columns


# ─── Test 2 ───────────────────────────────────────────────────────────────────

def test_clean_data_no_nulls():
    """After clean_data, quantity and unit_price have no NaN values."""

    df = load_data("data/sales_records.csv")
    cleaned = clean_data(df)

    assert cleaned['quantity'].isna().sum() == 0
    assert cleaned['unit_price'].isna().sum() == 0



# ─── Test 3 ───────────────────────────────────────────────────────────────────

def test_add_features_creates_revenue():
    """add_features creates a revenue column equal to quantity * unit_price."""

    df = load_data("data/sales_records.csv")
    cleaned = clean_data(df)
    enriched = add_features(cleaned)

    # revenue column exists
    assert 'revenue' in enriched.columns

    # check revenue calculation
    expected = enriched['quantity'] * enriched['unit_price']
    pd.testing.assert_series_equal(enriched['revenue'], expected, check_names=False)
