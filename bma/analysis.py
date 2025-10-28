"""
Analysis primitives for brand and market data.

This module provides functions for calculating key performance indicators (KPIs),
running customer segmentation, and analyzing various aspects of brand and market
performance.
"""
from typing import Tuple, Optional, List
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


def brand_performance(df: pd.DataFrame, value_col: str = 'net_sales') -> pd.Series:
    """
    Calculates the performance of each brand based on a specified value column.

    Args:
        df: The input DataFrame.
        value_col: The column to use for calculating performance (e.g., 'net_sales', 'volume').

    Returns:
        A pandas Series with the total value for each brand, sorted in descending order.

    Raises:
        ValueError: If the 'brand' or `value_col` column is not in the DataFrame.
    """
    df = df.copy()
    if 'brand' not in df.columns:
        raise ValueError("Input DataFrame must contain a 'brand' column.")
    if value_col not in df.columns:
        raise ValueError(f"Value column '{value_col}' not found in the DataFrame.")
    return df.groupby('brand')[value_col].sum().sort_values(ascending=False)


def client_performance(df: pd.DataFrame, value_col: str = 'net_sales') -> pd.Series:
    """
    Calculates the performance of each client based on a specified value column.

    Args:
        df: The input DataFrame.
        value_col: The column to use for calculating performance (e.g., 'net_sales', 'volume').

    Returns:
        A pandas Series with the total value for each client, sorted in descending order.

    Raises:
        ValueError: If the 'client' or `value_col` column is not in the DataFrame.
    """
    df = df.copy()
    if 'client' not in df.columns:
        raise ValueError("Input DataFrame must contain a 'client' column.")
    if value_col not in df.columns:
        raise ValueError(f"Value column '{value_col}' not found in the DataFrame.")
    return df.groupby('client')[value_col].sum().sort_values(ascending=False)


def discount_impact(df: pd.DataFrame, discount_col: str = 'discounts', volume_col: str = 'volume', groupby: str = 'brand') -> pd.Series:
    """
    Calculates the correlation between discounts and sales volume for each brand or client.

    Args:
        df: The input DataFrame.
        discount_col: The name of the discounts column.
        volume_col: The name of the sales volume column.
        groupby: The column to group by (e.g., 'brand', 'client').

    Returns:
        A pandas Series with the correlation between discounts and volume for each group.

    Raises:
        ValueError: If the required columns are not in the DataFrame.
    """
    df = df.copy()
    if groupby not in df.columns:
        raise ValueError(f"Groupby column '{groupby}' not found in the DataFrame.")
    if discount_col not in df.columns:
        raise ValueError(f"Discount column '{discount_col}' not found in the DataFrame.")
    if volume_col not in df.columns:
        raise ValueError(f"Volume column '{volume_col}' not found in the DataFrame.")

    def corr(x):
        if x[discount_col].isnull().all() or x[volume_col].isnull().all():
            return np.nan
        return x[discount_col].corr(x[volume_col])

    return df.groupby(groupby).apply(corr).rename('discount_volume_corr')


def cost_efficiency(df: pd.DataFrame, cost_cols: Tuple[str, ...] = ('cost_of_goods_sold', 'distribution', 'warehousing'), value_col: str = 'net_sales') -> pd.Series:
    """
    Calculates the cost efficiency for each brand.

    Args:
        df: The input DataFrame.
        cost_cols: A tuple of column names representing different costs.
        value_col: The column to use for calculating the value (e.g., 'net_sales').

    Returns:
        A pandas Series with the cost efficiency for each brand.
    """
    df = df.copy()
    for c in cost_cols:
        if c not in df.columns:
            logger.warning(f"Cost column '{c}' not found in DataFrame. It will be treated as 0.")
            df[c] = 0
    costs = df[list(cost_cols)].sum(axis=1)
    df['_total_costs'] = costs
    return df.groupby('brand').apply(lambda g: g['_total_costs'].sum() / (g[value_col].sum() if g[value_col].sum() != 0 else np.nan))


def product_preferences(df: pd.DataFrame, by: str = 'pack', value_col: str = 'volume') -> dict:
    """
    Determines the most and least popular product pack size for each brand and client.

    Args:
        df: The input DataFrame.
        by: The column representing the product attribute to analyze (e.g., 'pack', 'size').
        value_col: The column to use for determining popularity (e.g., 'volume', 'net_sales').

    Returns:
        A dictionary containing two Series: 'most_popular' and 'least_popular'.

    Raises:
        ValueError: If the required columns are not in the DataFrame.
    """
    group_cols = ['brand', 'client', by]
    for c in group_cols:
        if c not in df.columns:
            raise ValueError(f"Required column '{c}' not found in the DataFrame.")
    t = df.groupby(group_cols)[value_col].sum()
    most = t.groupby(level=[0, 1]).idxmax()
    least = t.groupby(level=[0, 1]).idxmin()
    return {'most_popular': most, 'least_popular': least}


def segment_data(df: pd.DataFrame, features: List[str], model: str = 'kmeans', n_clusters: int = 3, scale: bool = True, **kwargs) -> pd.DataFrame:
    """
    Segments data using different clustering algorithms.

    Args:
        df: The input DataFrame.
        features: A list of column names to use for clustering.
        model: The clustering model to use ('kmeans' or 'dbscan').
        n_clusters: The number of clusters for K-Means.
        scale: Whether to scale the features before clustering.
        **kwargs: Additional arguments for the clustering model.

    Returns:
        The input DataFrame with an additional '_segment' column containing the cluster labels.

    Raises:
        ValueError: If the feature columns are not in the DataFrame, if there are not enough features,
                    or if an unsupported model is selected.
    """
    df = df.copy()
    if not features:
        raise ValueError("Feature list cannot be empty.")
    for feature in features:
        if feature not in df.columns:
            raise ValueError(f"Feature column '{feature}' not found in the DataFrame.")

    X = df[features].copy()
    X = X.fillna(0)

    if X.shape[1] == 0:
        raise ValueError("No feature columns found to perform segmentation.")

    if scale:
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
    else:
        Xs = X.values

    if model == 'kmeans':
        clusterer = KMeans(n_clusters=n_clusters, random_state=42, **kwargs)
    elif model == 'dbscan':
        clusterer = DBSCAN(**kwargs)
    else:
        raise ValueError(f"Unsupported segmentation model: {model}. Choose 'kmeans' or 'dbscan'.")

    labels = clusterer.fit_predict(Xs)
    df['_segment'] = labels
    return df
