"""
Data cleaning and preparation utilities.

This module provides a set of functions for cleaning and preparing the raw
brand and market analysis data. It includes functions for standardizing column
names, parsing dates, handling missing values, and converting currencies.
"""
from typing import Dict, Optional, List, Tuple
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes column names by stripping whitespace, converting to lowercase,
    and replacing spaces with underscores.

    Args:
        df: The input DataFrame.

    Returns:
        A new DataFrame with standardized column names.
    """
    df = df.copy()
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
    return df


def parse_dates(df: pd.DataFrame, date_cols: Tuple[str, ...] = ('period',), fmt: Optional[str] = None) -> pd.DataFrame:
    """
    Parses date columns in a DataFrame.

    Args:
        df: The input DataFrame.
        date_cols: A tuple of column names to parse as dates.
        fmt: The format of the date strings. If None, pandas will infer the format.

    Returns:
        A new DataFrame with the date columns parsed.
    """
    df = df.copy()
    for col in date_cols:
        if col in df.columns:
            series = df[col]
            # Handle numeric YYYYMM (e.g., 202001) or YYYYMMDD integers
            try:
                if pd.api.types.is_integer_dtype(series) or pd.api.types.is_float_dtype(series):
                    # convert to int then to str
                    s = series.astype('Int64').astype(str)
                    # detect YYYYMM or YYYYMMDD by length
                    if s.str.len().isin([6,8]).all():
                        fmt_try = '%Y%m' if s.str.len().iloc[0] == 6 else '%Y%m%d'
                        df[col] = pd.to_datetime(s, format=fmt_try, errors='coerce')
                        continue
                # handle strings that look like YYYYMM
                if series.dtype == object:
                    s = series.astype(str).str.strip()
                    if s.str.match(r'^\d{6}$').any():
                        df[col] = pd.to_datetime(s, format='%Y%m', errors='coerce')
                        continue
                # fallback to pandas parser
                df[col] = pd.to_datetime(series, format=fmt, errors='coerce')
            except Exception as e:
                logger.warning(f"Could not parse date column '{col}': {e}")
                df[col] = pd.to_datetime(df[col], errors='coerce')
    return df


def fill_missing(df: pd.DataFrame, strategy: str = 'median', fill_map: Optional[Dict[str, object]] = None) -> pd.DataFrame:
    """
    Fills missing values in a DataFrame using various strategies.

    Args:
        df: The input DataFrame.
        strategy: The strategy for filling missing numeric values.
                  Can be 'median', 'mean', or 'zero'.
        fill_map: A dictionary mapping column names to specific fill values.

    Returns:
        A new DataFrame with missing values filled.
    """
    df = df.copy()
    if fill_map:
        for k, v in fill_map.items():
            if k in df.columns:
                df[k] = df[k].fillna(v)

    numeric = df.select_dtypes(include=[np.number]).columns
    if strategy == 'median':
        for c in numeric:
            df[c] = df[c].fillna(df[c].median())
    elif strategy == 'mean':
        for c in numeric:
            df[c] = df[c].fillna(df[c].mean())
    elif strategy == 'zero':
        df[numeric] = df[numeric].fillna(0)
    else:
        logger.warning(f"Unknown fill strategy '{strategy}'. No numeric imputation performed.")

    obj_cols = df.select_dtypes(include=['object', 'category']).columns
    for c in obj_cols:
        if df[c].isnull().any():
            try:
                df[c] = df[c].fillna(df[c].mode().iloc[0])
            except IndexError:
                df[c] = df[c].fillna('') # Handle cases where mode is empty
    return df


def clean_data(df: pd.DataFrame, date_cols: Tuple[str, ...] = ('Period',), numeric_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    A pipeline to standardize and clean a DataFrame.

    This function performs the following steps:
    1. Standardizes column names.
    2. Parses date columns.
    3. Coerces numeric columns to numeric types.
    4. Fills missing values.

    Args:
        df: The input DataFrame.
        date_cols: A tuple of date column names to parse.
        numeric_cols: A list of column names to coerce to numeric types.

    Returns:
        A cleaned and prepared DataFrame.
    """
    df = df.copy()
    df = standardize_columns(df)
    lower_date_cols = [c.lower() for c in date_cols]
    df = parse_dates(df, date_cols=tuple(lower_date_cols))

    if numeric_cols:
        for c in numeric_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')

    df = fill_missing(df)
    return df


def convert_currency(df: pd.DataFrame, amount_columns: List[str], rates: Dict[str, float], from_col: str = 'currency', target: str = 'USD') -> pd.DataFrame:
    """
    Converts currency columns to a target currency.

    Args:
        df: The input DataFrame.
        amount_columns: A list of column names containing currency values.
        rates: A dictionary mapping currency codes to their exchange rate against the target currency.
        from_col: The column containing the currency code for each row.
        target: The target currency code.

    Returns:
        A new DataFrame with the converted currency columns.

    Raises:
        ValueError: If the `from_col` is not in the DataFrame.
    """
    df = df.copy()
    if from_col not in df.columns:
        raise ValueError(f"Currency column '{from_col}' not found in the DataFrame.")

    for col in amount_columns:
        if col not in df.columns:
            logger.warning(f"Amount column '{col}' not found in DataFrame. Skipping conversion.")
            continue

        def _conv(row):
            cur = row.get(from_col)
            rate = rates.get(cur)
            if rate is None:
                logger.warning(f"No rate found for currency '{cur}'. Using 1.0.")
                rate = 1.0
            try:
                return float(row[col]) * float(rate)
            except (ValueError, TypeError):
                return row[col]

        df[f"{col}_in_{target}"] = df.apply(_conv, axis=1)
    return df