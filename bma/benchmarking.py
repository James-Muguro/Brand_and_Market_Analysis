"""
Competitive benchmarking analysis.

This module provides functions for comparing brand performance against
competitor data.
"""
import pandas as pd
from typing import List

def benchmark_performance(
    own_df: pd.DataFrame,
    competitor_df: pd.DataFrame,
    metric: str = 'net_sales',
    brand_col: str = 'brand'
) -> pd.DataFrame:
    """
    Compares brand performance between own data and competitor data.

    Args:
        own_df: DataFrame with your own brand data.
        competitor_df: DataFrame with competitor brand data.
        metric: The metric to compare (e.g., 'net_sales', 'volume').
        brand_col: The name of the brand column.

    Returns:
        A DataFrame with the performance comparison.
    """
    if brand_col not in own_df.columns or brand_col not in competitor_df.columns:
        raise ValueError(f"Brand column '{brand_col}' not found in one of the DataFrames.")
    if metric not in own_df.columns or metric not in competitor_df.columns:
        raise ValueError(f"Metric column '{metric}' not found in one of the DataFrames.")

    own_performance = own_df.groupby(brand_col)[metric].sum().reset_index()
    own_performance['source'] = 'Own'

    competitor_performance = competitor_df.groupby(brand_col)[metric].sum().reset_index()
    competitor_performance['source'] = 'Competitor'

    combined_performance = pd.concat([own_performance, competitor_performance])
    
    # Pivot to get own and competitor side-by-side
    pivot_df = combined_performance.pivot_table(index=brand_col, columns='source', values=metric).fillna(0)
    
    # Calculate market share
    pivot_df['total_market'] = pivot_df['Own'] + pivot_df['Competitor']
    pivot_df['market_share_own'] = (pivot_df['Own'] / pivot_df['total_market']) * 100
    pivot_df['market_share_competitor'] = (pivot_df['Competitor'] / pivot_df['total_market']) * 100
    
    return pivot_df.sort_values(by='total_market', ascending=False)
