"""
Data ingestion utilities for various sources and formats.

This module provides a flexible `load_data` function that can ingest data from
pandas DataFrames, local files (CSV, Excel, JSON), and URLs. It automatically
detects the source type and file format, making it easy to load data into a
consistent format for analysis.
"""
from typing import Optional, Union
import pandas as pd
import os
import requests
import logging
from io import StringIO

logger = logging.getLogger(__name__)


def _read_local(path: str, **kwargs) -> pd.DataFrame:
    """
    Reads data from a local file.

    Args:
        path: The path to the file.
        **kwargs: Additional arguments to pass to the pandas reader function.

    Returns:
        A pandas DataFrame with the loaded data.

    Raises:
        ValueError: If the file extension is not supported.
        FileNotFoundError: If the file is not found at the specified path.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found at: {path}")

    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == '.csv':
            return pd.read_csv(path, **kwargs)
        if ext in ('.xls', '.xlsx'):
            return pd.read_excel(path, **kwargs)
        if ext == '.json':
            return pd.read_json(path, **kwargs)
    except Exception as e:
        logger.error(f"Error reading file {path}: {e}")
        raise

    raise ValueError(f"Unsupported file extension: {ext}")


def _read_url(url: str, **kwargs) -> pd.DataFrame:
    """
    Reads data from a URL.

    Args:
        url: The URL to read data from.
        **kwargs: Additional arguments to pass to the pandas reader function.

    Returns:
        A pandas DataFrame with the loaded data.

    Raises:
        ValueError: If the data from the URL cannot be parsed.
        requests.exceptions.RequestException: For network-related errors.
    """
    try:
        if url.lower().endswith('.csv'):
            return pd.read_csv(url, **kwargs)
        if url.lower().endswith('.json'):
            return pd.read_json(url, **kwargs)
    except Exception as e:
        logger.warning(f"Direct pandas read from URL failed: {e}. Falling back to requests.")

    try:
        resp = requests.get(url)
        resp.raise_for_status()
        text = resp.text
        
        try:
            return pd.read_csv(StringIO(text), **kwargs)
        except Exception:
            try:
                return pd.read_json(StringIO(text), **kwargs)
            except Exception as e:
                raise ValueError("Unable to parse remote resource as CSV or JSON.") from e
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching data from URL {url}: {e}")
        raise


def load_data(source: Union[str, pd.DataFrame], **kwargs) -> pd.DataFrame:
    """
    Loads data from various sources.

    This function can load data from:
    - A pandas DataFrame (returns a copy).
    - A local file path (CSV, Excel, JSON).
    - A URL (attempts to parse as CSV or JSON).

    Args:
        source: The data source to load from.
        **kwargs: Additional arguments to pass to the underlying pandas reader.

    Returns:
        A pandas DataFrame containing the loaded data.

    Raises:
        TypeError: If the source is not a DataFrame, string, or supported file-like object.
    """
    if isinstance(source, pd.DataFrame):
        return source.copy()
    if isinstance(source, str):
        if source.startswith(('http://', 'https://')):
            return _read_url(source, **kwargs)
        return _read_local(source, **kwargs)
    
    # Handle file-like objects from Streamlit's file_uploader
    if hasattr(source, 'read'):
        try:
            # Try reading as CSV
            return pd.read_csv(source, **kwargs)
        except Exception:
            try:
                # Reset buffer and try reading as Excel
                source.seek(0)
                return pd.read_excel(source, **kwargs)
            except Exception:
                try:
                    # Reset buffer and try reading as JSON
                    source.seek(0)
                    return pd.read_json(source, **kwargs)
                except Exception as e:
                    raise ValueError("Uploaded file could not be parsed as CSV, Excel, or JSON.") from e

    raise TypeError("source must be a pandas DataFrame, local path, URL string, or a file-like object.")