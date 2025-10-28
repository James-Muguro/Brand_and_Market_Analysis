"""
AI-driven recommendations for growth and risk mitigation.

This module uses a machine learning model to classify brands and generate
strategic recommendations.
"""
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def train_recommendation_model(df: pd.DataFrame) -> Optional[LogisticRegression]:
    """
    Trains a simple model to classify brand performance.

    Args:
        df: The input DataFrame with brand data.

    Returns:
        A trained logistic regression model, or None if training fails.
    """
    try:
        # Feature Engineering
        df['yoy_growth'] = df.groupby('brand')['net_sales'].pct_change(periods=12) * 100
        df['discount_ratio'] = df['discounts'] / df['net_sales']
        df['cost_ratio'] = (df['cost_of_goods_sold'] + df['distribution'] + df['warehousing']) / df['net_sales']
        
        features = ['yoy_growth', 'discount_ratio', 'cost_ratio']
        df.dropna(subset=features, inplace=True)

        if df.empty:
            return None

        # Target variable: classify brands into growth, stable, or decline
        def classify_brand(growth):
            if growth > 5:
                return 'Growth'
            elif growth < -5:
                return 'Decline'
            else:
                return 'Stable'

        df['performance_class'] = df['yoy_growth'].apply(classify_brand)
        
        X = df[features]
        y = df['performance_class']

        if len(y.unique()) < 2:
            return None # Not enough classes to train

        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        model = LogisticRegression(random_state=42, class_weight='balanced')
        model.fit(X_train_scaled, y_train)
        
        # Attach scaler to model for later use
        model.scaler = scaler
        
        return model

    except Exception as e:
        print(f"Error training recommendation model: {e}")
        return None


def generate_recommendations(model: LogisticRegression, brand_data: pd.DataFrame) -> List[Dict]:
    """
    Generates recommendations for a brand based on its performance class.

    Args:
        model: The trained classification model.
        brand_data: The data for a single brand.

    Returns:
        A list of recommendation dictionaries.
    """
    if model is None:
        return [{
            'priority': 5,
            'title': 'Recommendation Model Not Available',
            'recommendation': 'Could not generate AI-driven recommendations due to lack of data or model training failure.'
        }]

    try:
        # Feature Engineering for the brand
        brand_data['yoy_growth'] = brand_data['net_sales'].pct_change(periods=12) * 100
        brand_data['discount_ratio'] = brand_data['discounts'] / brand_data['net_sales']
        brand_data['cost_ratio'] = (brand_data['cost_of_goods_sold'] + brand_data['distribution'] + brand_data['warehousing']) / brand_data['net_sales']
        
        features = ['yoy_growth', 'discount_ratio', 'cost_ratio']
        brand_data.dropna(subset=features, inplace=True)

        if brand_data.empty:
            return []

        # Get the latest data point for the brand
        latest_data = brand_data.iloc[-1][features].values.reshape(1, -1)
        latest_data_scaled = model.scaler.transform(latest_data)
        
        prediction = model.predict(latest_data_scaled)[0]
        
        recs = []
        if prediction == 'Growth':
            recs.append({
                'priority': 1,
                'title': f'Capitalize on Growth for {brand_data["brand"].iloc[0]}',
                'recommendation': 'This brand is in a growth phase. Consider increasing marketing spend and expanding distribution to maximize market share. Monitor competitive response.'
            })
        elif prediction == 'Decline':
            recs.append({
                'priority': 1,
                'title': f'Address Decline for {brand_data["brand"].iloc[0]}',
                'recommendation': 'This brand is showing signs of decline. Investigate root causes, such as competitive pressure, changing consumer preferences, or pricing issues. Consider a brand refresh or strategic pivot.'
            })
        else: # Stable
            recs.append({
                'priority': 2,
                'title': f'Maintain Stability for {brand_data["brand"].iloc[0]}',
                'recommendation': 'This brand is stable. Focus on maintaining market position and profitability. Explore opportunities for incremental innovation and cost optimization.'
            })
        return recs

    except Exception as e:
        print(f"Error generating recommendations: {e}")
        return []


def recommend_actions(df: pd.DataFrame, kpis: Dict[str, pd.Series] = None) -> List[Dict]:
    """High-level API used by the dashboard to return prioritized recommendations.

    Strategy:
    - Try to train an ML model and produce brand-level recommendations using it.
    - If model training fails or is not possible, fall back to a simple rule-based
      set of explainable recommendations derived from KPIs passed in `kpis` or
      computed from `df`.
    """
    recs: List[Dict] = []
    kpis = kpis or {}

    # First, try ML-driven recommendations
    try:
        model = train_recommendation_model(df)
        if model is not None and 'brand' in df.columns:
            for brand in df['brand'].unique():
                brand_data = df[df['brand'] == brand]
                recs.extend(generate_recommendations(model, brand_data))
            if recs:
                return sorted(recs, key=lambda r: r.get('priority', 99))
    except Exception:
        # If ML path fails, we'll fall back to rules below
        pass

    # Rule-based fallback (explainable)
    try:
        # Top brands protection
        if 'top_brands' in kpis and not kpis['top_brands'].empty:
            top = kpis['top_brands'].head(1)
            brand = top.index[0]
            recs.append({
                'priority': 1,
                'title': f'Protect revenue for top brand: {brand}',
                'recommendation': f'Prioritize availability and targeted support for {brand} to avoid revenue loss.'
            })

        # Discount dependency
        if 'discount_impact' in kpis:
            dep = kpis['discount_impact'].dropna()
            strong_dep = dep[dep > 0.5]
            for b in strong_dep.index.tolist()[:3]:
                recs.append({
                    'priority': 2,
                    'title': f'Reduce promo dependency for {b}',
                    'recommendation': 'Shift from discounts to loyalty/value-pack strategies and test margin-preserving options.'
                })

        # Cost efficiency
        if 'cost_efficiency' in kpis and not kpis['cost_efficiency'].empty:
            worst = kpis['cost_efficiency'].dropna().sort_values().head(1)
            if not worst.empty:
                brand = worst.index[0]
                recs.append({
                    'priority': 2,
                    'title': f'Improve cost efficiency for {brand}',
                    'recommendation': 'Review COGS and distribution; consider SKU rationalization or packaging optimization.'
                })

    except Exception:
        # If fallback logic errors, return at least an empty list
        pass

    if not recs:
        recs.append({
            'priority': 5,
            'title': 'No actionable recommendations generated',
            'recommendation': 'Data looks stable or insufficient for automated recommendations. Consider manual review.'
        })

    return sorted(recs, key=lambda r: r.get('priority', 99))