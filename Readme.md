# Brand and Market Analysis Project

## Overview

This project provides a comprehensive and extensible framework for brand and market analysis, enabling businesses to make data-driven decisions. It is designed to be highly modular, customizable, and user-friendly, supporting a wide range of analyses from brand performance to predictive forecasting.

## Core Features

*   **Modular Architecture:** The project is organized into a `bma` package with distinct modules for each analysis task, making it easy to extend and maintain.
*   **Configuration-Driven:** The analysis pipeline is controlled by a central `config.yaml` file, allowing for easy customization of parameters without changing the code.
*   **Multi-Source Data Ingestion:** The `data_ingest` module can load data from various sources, including local files (CSV, Excel, JSON) and URLs.
*   **Automated Data Cleaning:** The `cleaning` module provides a pipeline for standardizing column names, parsing dates, and handling missing values.
*   **Advanced Analysis:** The `analysis` module includes functions for brand and client performance, discount impact, cost efficiency, and customer segmentation.
*   **Interactive Dashboard:** An interactive Streamlit dashboard (`dashboard_streamlit.py`) allows for visual exploration of the data and analysis results.
*   **Extensible Framework:** The new architecture with the `AnalysisOrchestrator` makes it easy to add new analysis modules and integrate them into the pipeline.

## Project Structure

The project has been refactored into a modular `bma` package:

*   `bma/orchestrator.py`: Contains the `AnalysisOrchestrator` class, which runs the entire analysis pipeline based on the configuration.
*   `bma/data_ingest.py`: Handles loading data from various sources.
*   `bma/cleaning.py`: Provides data cleaning and preparation functions.
*   `bma/analysis.py`: Contains the core analysis functions.
*   `bma/forecast.py`: For time series forecasting.
*   `bma/sentiment.py`: For sentiment analysis.
*   `bma/recommendations.py`: For generating strategic recommendations.
*   `bma/scenario.py`: For scenario simulation.
*   `brand_market_analysis.py`: The main entry point for running the analysis from the command line.
*   `dashboard_streamlit.py`: The Streamlit dashboard for interactive analysis.
*   `config.yaml`: The central configuration file for the analysis pipeline.

## Usage

### Installation

Ensure the following libraries are installed:

```bash
pip install -r requirements.txt
```

### Running the Analysis

The analysis can be run from the command line using the `brand_market_analysis.py` script:

```bash
python brand_market_analysis.py config.yaml path/to/your/data.csv
```

### Interactive Dashboard

To start the interactive Streamlit dashboard, run the following command from the project root:

```bash
streamlit run bma/dashboard_streamlit.py
```

## Future Enhancements

This project is designed for continuous improvement. Future enhancements will focus on:

*   **Advanced Forecasting Models:** Integrating more advanced forecasting models like Prophet and ARIMA.
*   **Machine Learning-Powered Recommendations:** Evolving the recommendation engine to use machine learning for more nuanced advice.
*   **Competitive Benchmarking:** Adding a dedicated module for comparing brand performance against competitors.
*   **Enhanced Scenario Simulation:** Expanding the scenario simulation capabilities to model more complex business decisions.

## Conclusion

This in-depth analysis tool offers actionable insights for businesses, guiding strategies related to pricing, inventory management, and targeted marketing campaigns in the retail landscape. The new modular and configuration-driven architecture makes it a powerful and flexible tool for any data analyst or business strategist.

## Acknowledgments

We extend our gratitude to the contributors and the open-source community for their invaluable input.
