"""Streamlit dashboard for interactive brand and market analysis.

This dashboard provides a user-friendly interface for exploring brand and market
data, running various analyses, and visualizing the results.

Run with: `streamlit run bma/dashboard_streamlit.py` from the project root.
"""
import sys
import pathlib
import streamlit as st
import pandas as pd
import yaml

# Ensure project root is on sys.path so `bma` package is importable when running
# `streamlit run bma/dashboard_streamlit.py` from the project root.
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bma.data_ingest import load_data
from bma.cleaning import clean_data, convert_currency
from bma.analysis import brand_performance, client_performance, segment_data
from bma.benchmarking import benchmark_performance
from bma.forecast import forecast_series
from bma.scenario import simulate_impact
from bma.recommendations import recommend_actions
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title='Brand & Market Analysis', layout='wide')

st.title('Brand & Market Analysis')

# Load config
@st.cache_data
def load_config(path='config.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

config = load_config()

# --- Sidebar ---
st.sidebar.title('Configuration')

# --- Data Loading ---
uploaded_file = st.sidebar.file_uploader('Upload Your Dataset', type=['csv', 'xlsx', 'xls', 'json'])

if uploaded_file is not None:
    try:
        df = load_data(uploaded_file)
        df.columns = [c.lower().replace(' ', '_') for c in df.columns]
        # run cleaning pipeline to normalize and parse dates
        try:
            df = clean_data(df, date_cols=(config.get('data', {}).get('time_column', 'Period'),))
        except Exception:
            # cleaning shouldn't block loading; proceed with raw df if it fails
            pass
        st.session_state['df'] = df
        st.success('Dataset loaded successfully!')
    except Exception as e:
        st.error(f'Failed to load or process dataset: {e}')

# Fallback: try to load the default FMCG_data.xlsx from the project root if present
if 'df' not in st.session_state and (ROOT / 'FMCG_data.xlsx').exists():
    try:
        df = load_data(str(ROOT / 'FMCG_data.xlsx'))
        df.columns = [c.lower().replace(' ', '_') for c in df.columns]
        try:
            df = clean_data(df, date_cols=(config.get('data', {}).get('time_column', 'Period'),))
        except Exception:
            pass
        st.session_state['df'] = df
        st.info('Loaded default dataset FMCG_data.xlsx from project root')
    except Exception as e:
        st.warning(f'Failed to load default dataset FMCG_data.xlsx: {e}')

if 'df' in st.session_state:
    df = st.session_state['df']

    # --- Global Filters ---
    st.sidebar.header('Global Filters')
    metric = st.sidebar.selectbox('Primary Metric', options=df.select_dtypes(include=['number']).columns.tolist())
    
    # Currency Conversion
    if 'currency' in config and config['currency'].get('target'):
        if st.sidebar.checkbox(f"Convert to {config['currency']['target']}"):
            currency_config = config['currency']
            amount_cols = [c for c in metric if 'sales' in c or 'cost' in c]
            df = convert_currency(df, amount_columns=amount_cols, **currency_config)
            st.session_state['df'] = df

    # --- Main Tabs ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Performance Overview", 
        "Competitive Benchmarking", 
        "Customer Segmentation", 
        "Forecasting & Scenarios", 
        "AI Recommendations"
    ])

    with tab1:
        st.header('Performance Overview')
        col1, col2 = st.columns(2)
        with col1:
            if 'brand' in df.columns:
                bp = brand_performance(df, value_col=metric)
                fig = px.bar(bp.reset_index(), x='brand', y=metric, title=f'Top Brands by {metric}')
                st.plotly_chart(fig, use_container_width=True)
        with col2:
            if 'client' in df.columns:
                cp = client_performance(df, value_col=metric)
                fig = px.bar(cp.reset_index(), x='client', y=metric, title=f'Top Clients by {metric}')
                st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.header('Competitive Benchmarking')
        competitor_file = st.file_uploader('Upload Competitor Dataset', type=['csv', 'xlsx', 'xls', 'json'], key="competitor")
        if competitor_file is not None:
            try:
                competitor_df = load_data(competitor_file)
                competitor_df.columns = [c.lower().replace(' ', '_') for c in competitor_df.columns]
                
                st.write("Competitor Data Preview")
                st.dataframe(competitor_df.head())

                if st.button("Run Benchmarking Analysis"):
                    benchmark_results = benchmark_performance(df, competitor_df, metric=metric)
                    st.write("Benchmarking Results")
                    st.dataframe(benchmark_results)

                    market_share_df = benchmark_results[['market_share_own', 'market_share_competitor']].reset_index()
                    market_share_df = market_share_df.melt(id_vars='brand', var_name='source', value_name='market_share')
                    
                    fig = px.bar(market_share_df, x='brand', y='market_share', color='source', title='Market Share Comparison', barmode='group')
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Failed to process competitor data: {e}")

    with tab3:
        st.header('Customer Segmentation')
        features = st.multiselect('Select features for segmentation', options=df.select_dtypes(include=['number']).columns.tolist(), default=df.select_dtypes(include=['number']).columns.tolist()[:2])
        seg_model = st.selectbox('Select segmentation model', options=['kmeans', 'dbscan'])
        
        model_params = {}
        if seg_model == 'kmeans':
            n_clusters = st.slider('Number of clusters', 2, 10, 3)
            model_params['n_clusters'] = n_clusters
        elif seg_model == 'dbscan':
            eps = st.slider('Epsilon (eps)', 0.1, 5.0, 0.5)
            min_samples = st.slider('Minimum samples', 1, 10, 5)
            model_params['eps'] = eps
            model_params['min_samples'] = min_samples

        if st.button('Run Segmentation'):
            if len(features) >= 2:
                seg_df = segment_data(df, features=features, model=seg_model, **model_params)
                st.write("Segment Counts")
                st.dataframe(seg_df['_segment'].value_counts().rename('count'))
                
                fig = px.scatter(seg_df, x=features[0], y=features[1], color='_segment', title='Customer Segments', labels={'_segment': 'Segment'}, hover_data=df.columns)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info('Please select at least two features for segmentation.')

    with tab4:
        st.header('Forecasting & Scenarios')
        time_col = config.get('data', {}).get('time_column', 'period')
        if time_col in df.columns:
            df[time_col] = pd.to_datetime(df[time_col])
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader('Trend Forecasting')
                ts_brand = st.selectbox('Choose brand for forecasting', options=sorted(df['brand'].unique()) if 'brand' in df.columns else [])
                forecast_model = st.selectbox('Select forecast model', options=['ets', 'arima', 'prophet'])
                forecast_horizon = st.slider('Forecast horizon (months)', 1, 36, 12)

                if st.button('Run Forecast'):
                    if ts_brand:
                        series = df[df['brand'] == ts_brand].set_index(time_col)[metric].resample('M').sum()
                        forecast_df = forecast_series(series, model=forecast_model, periods=forecast_horizon)
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=series.index, y=series, mode='lines', name='Actual'))
                        fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['forecast'], mode='lines', name='Forecast'))
                        if 'lower_bound' in forecast_df.columns and 'upper_bound' in forecast_df.columns:
                            fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['upper_bound'], fill=None, mode='lines', line_color='rgba(0,0,0,0)', name='Upper Bound'))
                            fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['lower_bound'], fill='tonexty', mode='lines', line_color='rgba(0,0,0,0)', name='Lower Bound'))
                        
                        fig.update_layout(title=f'Forecast for {ts_brand}', xaxis_title='Date', yaxis_title=metric)
                        st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader('Impact Simulation')
                price_change_pct = st.slider('Price Change (%)', -25.0, 25.0, 0.0, 0.5) / 100
                marketing_spend_increase = st.number_input('Marketing Spend Increase ($)', value=0)
                price_elasticity = st.number_input('Price Elasticity', value=-1.5)
                marketing_elasticity = st.number_input('Marketing Elasticity', value=0.2)

                if st.button('Simulate Impact'):
                    sim_df = simulate_impact(
                        df,
                        price_change_pct=price_change_pct,
                        marketing_spend_increase=marketing_spend_increase,
                        price_elasticity=price_elasticity,
                        marketing_elasticity=marketing_elasticity
                    )
                    
                    st.write("Simulation Results")
                    st.dataframe(sim_df[['brand', 'volume', 'volume_after_price_change', 'volume_after_marketing']].head())
                    
                    total_volume_before = df['volume'].sum()
                    total_volume_after = sim_df['volume_after_marketing'].sum()
                    volume_change_pct = ((total_volume_after - total_volume_before) / total_volume_before) * 100
                    
                    st.metric(label="Simulated Change in Total Volume", value=f"{volume_change_pct:.2f}%")

        else:
            st.info("No time column found for forecasting and scenarios. Please check your data and `config.yaml`.")

    with tab5:
        st.header('AI-Driven Recommendations')
        if st.button('Generate Recommendations'):
            with st.spinner('Generating recommendations...'):
                try:
                    kpis = {}
                    if 'brand' in df.columns:
                        kpis['top_brands'] = brand_performance(df, value_col=metric)
                    if 'brand' in df.columns:
                        try:
                            kpis['cost_efficiency'] = client_performance(df, value_col=metric)
                        except Exception:
                            kpis['cost_efficiency'] = pd.Series()
                    recs = recommend_actions(df, kpis=kpis)
                    for r in recs:
                        st.markdown(f"**{r['title']}** — {r['recommendation']}")
                except Exception as err:
                    st.error(f'Failed to generate recommendations: {err}')
