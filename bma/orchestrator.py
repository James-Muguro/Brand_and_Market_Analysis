import yaml
import logging
from typing import Dict, Any

from bma.data_ingest import load_data
from bma.cleaning import clean_data
from bma.analysis import (
    brand_performance,
    client_performance,
    discount_impact,
    cost_efficiency,
    segment_data
)
from bma.benchmarking import benchmark_performance
from bma.forecast import forecast_series
from bma.sentiment import analyze_sentiment
from bma.recommendations import train_recommendation_model, generate_recommendations
from bma.scenario import simulate_impact

logger = logging.getLogger(__name__)

class AnalysisOrchestrator:
    """
    Orchestrates the brand and market analysis pipeline.
    """
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.data = None
        self.results = {}
        self.recommendation_model = None

    def run(self, data_path: str, competitor_data_path: str = None):
        """
        Runs the full analysis pipeline.
        """
        self.load_and_clean_data(data_path)
        self.run_analysis()
        if competitor_data_path:
            self.run_benchmarking(competitor_data_path)
        self.run_forecast()
        self.run_sentiment_analysis()
        self.run_recommendations()
        self.run_scenario_simulation()
        return self.results

    def load_and_clean_data(self, data_path: str):
        """
        Loads and cleans the data.
        """
        logger.info("Loading and cleaning data...")
        data_config = self.config.get('data', {})
        self.data = load_data(data_path)
        self.data = clean_data(
            self.data,
            date_cols=data_config.get('date_columns', []),
            numeric_cols=data_config.get('numeric_columns', [])
        )
        self.data.columns = [c.lower().replace(' ', '_') for c in self.data.columns]
        logger.info("Data loaded and cleaned successfully.")

    def run_analysis(self):
        """
        Runs the analysis modules.
        """
        logger.info("Running analysis...")
        analysis_config = self.config.get('analysis', {})
        if analysis_config.get('brand_performance', {}).get('enabled', True):
            self.results['brand_performance'] = brand_performance(
                self.data,
                value_col=analysis_config.get('brand_performance', {}).get('value_col', 'net_sales')
            )
        if analysis_config.get('client_performance', {}).get('enabled', True):
            self.results['client_performance'] = client_performance(
                self.data,
                value_col=analysis_config.get('client_performance', {}).get('value_col', 'net_sales')
            )
        if analysis_config.get('discount_impact', {}).get('enabled', True):
            self.results['discount_impact'] = discount_impact(self.data)
        if analysis_config.get('cost_efficiency', {}).get('enabled', True):
            self.results['cost_efficiency'] = cost_efficiency(self.data)
        if analysis_config.get('segmentation', {}).get('enabled', True):
            segmentation_config = analysis_config.get('segmentation', {})
            self.results['segmentation'] = segment_data(
                self.data,
                features=segmentation_config.get('features', []),
                model=segmentation_config.get('model', 'kmeans'),
                n_clusters=segmentation_config.get('n_clusters', 3),
                **segmentation_config.get('model_params', {})
            )
        logger.info("Analysis complete.")

    def run_benchmarking(self, competitor_data_path: str):
        """
        Runs the benchmarking analysis.
        """
        logger.info("Running benchmarking analysis...")
        benchmarking_config = self.config.get('benchmarking', {})
        if benchmarking_config.get('enabled', True):
            competitor_data = load_data(competitor_data_path)
            competitor_data = clean_data(
                competitor_data,
                date_cols=self.config.get('data', {}).get('date_columns', []),
                numeric_cols=self.config.get('data', {}).get('numeric_columns', [])
            )
            competitor_data.columns = [c.lower().replace(' ', '_') for c in competitor_data.columns]

            self.results['benchmarking'] = benchmark_performance(
                self.data,
                competitor_data,
                metric=benchmarking_config.get('metric', 'net_sales'),
                brand_col=benchmarking_config.get('brand_col', 'brand')
            )
        logger.info("Benchmarking analysis complete.")

    def run_forecast(self):
        """
        Runs the forecasting module.
        """
        logger.info("Running forecast...")
        forecast_config = self.config.get('forecast', {})
        if forecast_config.get('enabled', True):
            time_col = self.config.get('data', {}).get('time_column', 'period')
            target_col = forecast_config.get('target_column', 'net_sales')
            
            if time_col in self.data.columns and target_col in self.data.columns:
                self.data[time_col] = pd.to_datetime(self.data[time_col])
                
                if 'brand' in self.data.columns:
                    forecast_results = {}
                    for brand in self.data['brand'].unique():
                        series = self.data[self.data['brand'] == brand].set_index(time_col)[target_col].resample('M').sum()
                        try:
                            forecast_results[brand] = forecast_series(
                                series,
                                model=forecast_config.get('model', 'ets'),
                                periods=forecast_config.get('periods', 12),
                                **forecast_config.get('model_params', {})
                            )
                        except Exception as e:
                            logger.error(f"Forecast failed for brand '{brand}': {e}")
                    self.results['forecast'] = forecast_results
            else:
                logger.warning("Time column or target column not found for forecasting.")
        logger.info("Forecast complete.")

    def run_sentiment_analysis(self):
        """
        Runs the sentiment analysis module.
        """
        logger.info("Running sentiment analysis...")
        sentiment_config = self.config.get('sentiment', {})
        if sentiment_config.get('enabled', True) and 'text_column' in sentiment_config:
            if sentiment_config['text_column'] in self.data.columns:
                self.results['sentiment_analysis'] = analyze_sentiment(
                    self.data[sentiment_config['text_column']].dropna().tolist()
                )
            else:
                logger.warning(f"Sentiment analysis text column '{sentiment_config['text_column']}' not found.")
        logger.info("Sentiment analysis complete.")

    def run_recommendations(self):
        """
        Trains the recommendation model and generates recommendations.
        """
        logger.info("Generating recommendations...")
        recommendations_config = self.config.get('recommendations', {})
        if recommendations_config.get('enabled', True):
            if self.recommendation_model is None:
                self.recommendation_model = train_recommendation_model(self.data)
            
            if self.recommendation_model:
                recommendations = []
                if 'brand' in self.data.columns:
                    for brand in self.data['brand'].unique():
                        brand_data = self.data[self.data['brand'] == brand]
                        recommendations.extend(generate_recommendations(self.recommendation_model, brand_data))
                self.results['recommendations'] = recommendations
            else:
                logger.warning("Recommendation model could not be trained. Skipping recommendations.")
        logger.info("Recommendations generated.")

    def run_scenario_simulation(self):
        """
        Runs the scenario simulation.
        """
        logger.info("Running scenario simulation...")
        scenario_config = self.config.get('scenario', {})
        if scenario_config.get('enabled', True):
            self.results['scenario_simulation'] = simulate_impact(
                self.data,
                **scenario_config.get('impact_params', {})
            )
        logger.info("Scenario simulation complete.")