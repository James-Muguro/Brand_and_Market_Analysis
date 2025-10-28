import sys
import logging
import yaml
from bma.orchestrator import AnalysisOrchestrator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main(config_path: str, data_path: str):
    """
    Main function to run the analysis.
    """
    try:
        orchestrator = AnalysisOrchestrator(config_path)
        results = orchestrator.run(data_path)
        for k, v in results.items():
            print(f'--- {k} ---')
            try:
                print(v)
            except Exception as err:
                logger.exception('failed to print result %s: %s', k, err)
                print(repr(v))
    except Exception as e:
        logger.exception("An error occurred during the analysis: %s", e)
        sys.exit(1)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python brand_market_analysis.py <path-to-config> <path-to-data>')
        sys.exit(1)
    config_path = sys.argv[1]
    data_path = sys.argv[2]
    main(config_path, data_path)
