"""Brand & Market Analysis (bma) package

Small, modular toolkit for brand and market analysis with data ingestion,
cleaning, segmentation, sentiment, forecasting and scenario simulation.
"""

from .data_ingest import load_data
from .cleaning import clean_data, convert_currency
from .analysis import (
    brand_performance,
    client_performance,
    cost_efficiency,
    discount_impact,
    product_preferences,
    segment_data as segment_customers,  # Updated to match actual function name
)
from .sentiment import analyze_sentiment
from .forecast import forecast_series
from .scenario import simulate_scenarios

__all__ = [
    "load_data",
    "clean_data",
    "convert_currency",
    "brand_performance",
    "client_performance",
    "cost_efficiency",
    "discount_impact",
    "product_preferences",
    "segment_customers",
    "analyze_sentiment",
    "forecast_series",
    "simulate_scenarios",
]

