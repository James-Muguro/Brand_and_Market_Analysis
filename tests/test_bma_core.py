import pandas as pd
import numpy as np
from bma.data_ingest import load_data
from bma.cleaning import clean_data
from bma.analysis import segment_customers
from bma.recommendations import recommend_actions


def test_load_data_from_dataframe():
    df = pd.DataFrame({'A': [1, 2, 3]})
    out = load_data(df)
    assert isinstance(out, pd.DataFrame)
    assert out.shape == (3, 1)


def test_clean_data_and_parse_dates():
    df = pd.DataFrame({'Period': ['202001', '202002', '202003'], 'Net Sales': [100, None, 200]})
    cleaned = clean_data(df, date_cols=('Period',))
    assert 'period' in cleaned.columns or 'Period' in cleaned.columns


def test_segmentation_basic():
    df = pd.DataFrame({'brand': ['a', 'b', 'c'], 'v1': [1, 2, 3], 'v2': [3, 2, 1]})
    out = segment_customers(df, features=['v1', 'v2'], n_clusters=2)
    assert '_segment' in out.columns


def test_recommendations_runs():
    df = pd.DataFrame({
        'brand': ['x', 'y', 'z'],
        'net_sales': [100, 80, 60],
        'volume': [10, 8, 6],
        'discounts': [0.1, 0.2, 0.05],
        'period': ['2020-01-01', '2020-01-01', '2020-01-01']
    })
    recs = recommend_actions(df)
    assert isinstance(recs, list)
