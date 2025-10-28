"""
Scenario simulation utilities for what-if analysis.
"""
from typing import Dict, Union
import pandas as pd
import numpy as np

def simulate_scenarios(df: pd.DataFrame, base_col: str = 'net_sales', scenario: Dict[str, float] = None) -> pd.DataFrame:
    """
    Apply scenario multipliers to the base column and return a DataFrame
    with a column per scenario. scenario is a dict name -> multiplier (e.g. {'base':1.0,'up_10':1.1}).
    """
    df = df.copy()
    if scenario is None:
        scenario = {'base': 1.0}
    out = pd.DataFrame(index=df.index)
    for name, mult in scenario.items():
        out[name] = df[base_col] * float(mult)
    return out


def simulate_impact(
    df: pd.DataFrame,
    price_change_pct: float = 0.0,
    marketing_spend_increase: float = 0.0,
    price_elasticity: float = -1.0,
    marketing_elasticity: float = 0.1,
) -> pd.DataFrame:
    """Simple impact simulator that adjusts volume based on price and marketing.

    - price_change_pct: e.g., -0.1 for -10% price change
    - marketing_spend_increase: absolute extra marketing spend (not used in currency terms, just a proxy)
    - price_elasticity: expected % change in volume per 1% price change
    - marketing_elasticity: expected % change in volume per unit marketing increase (approx)
    Returns a DataFrame with new volume columns.
    """
    df = df.copy()
    if 'volume' not in df.columns:
        raise ValueError('Input data must contain a "volume" column')

    # compute percent changes
    # price impact: delta_volume_pct = price_elasticity * (price_change_pct * 100)
    # but price_change_pct is already a fraction, so use directly
    delta_volume_pct_price = price_elasticity * price_change_pct

    # marketing impact: simple proportional effect of marketing_spend_increase
    # scale marketing by average spend to avoid huge swings
    avg_marketing = df.get('marketing_spend', pd.Series([0])).mean() or 1.0
    marketing_ratio = marketing_spend_increase / (avg_marketing if avg_marketing != 0 else 1.0)
    delta_volume_pct_marketing = marketing_elasticity * marketing_ratio

    df['volume_after_price_change'] = df['volume'] * (1.0 + delta_volume_pct_price)
    df['volume_after_marketing'] = df['volume_after_price_change'] * (1.0 + delta_volume_pct_marketing)

    return df

def simulate_impact(
    df: pd.DataFrame,
    price_change_pct: float = 0.0,
    marketing_spend_increase: float = 0.0,
    price_elasticity: float = -1.5,
    marketing_elasticity: float = 0.2
) -> pd.DataFrame:
    """
    Simulates the impact of pricing and marketing changes on sales volume.

    Args:
        df: The input DataFrame with historical data.
        price_change_pct: The percentage change in price (e.g., 0.1 for a 10% increase).
        marketing_spend_increase: The absolute increase in marketing spend.
        price_elasticity: The price elasticity of demand.
        marketing_elasticity: The marketing elasticity of demand.

    Returns:
        A DataFrame with the simulated sales volume.
    """
    sim_df = df.copy()
    
    # Simulate price impact
    sim_df['volume_after_price_change'] = sim_df['volume'] * (1 + price_elasticity * price_change_pct)
    
    # Simulate marketing impact
    # Assuming marketing spend is proportional to net sales
    initial_marketing_spend = sim_df['net_sales'] * 0.1  # Assuming 10% of net sales
    new_marketing_spend = initial_marketing_spend + marketing_spend_increase
    marketing_spend_pct_change = (new_marketing_spend - initial_marketing_spend) / initial_marketing_spend
    
    sim_df['volume_after_marketing'] = sim_df['volume_after_price_change'] * (1 + marketing_elasticity * marketing_spend_pct_change)
    
    return sim_df