"""Sentiment utilities: use VADER for quick sentiment and optional HF pipeline.
"""
from typing import Optional, List
import logging
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _VADER_AVAILABLE = True
except Exception:
    _VADER_AVAILABLE = False


def analyze_sentiment(texts: List[str], engine: str = 'vader') -> pd.DataFrame:
    """Return sentiment scores for a list of texts.

    Supported engines: 'vader'. If unavailable, raises informative error.
    """
    if engine == 'vader':
        if not _VADER_AVAILABLE:
            raise RuntimeError('vaderSentiment not installed. pip install vaderSentiment')
        analyzer = SentimentIntensityAnalyzer()
        rows = []
        for t in texts:
            if t is None:
                rows.append({'text': t, 'neg': None, 'neu': None, 'pos': None, 'compound': None})
                continue
            s = analyzer.polarity_scores(str(t))
            rows.append({'text': t, **s})
        return pd.DataFrame(rows)
    else:
        raise ValueError('Unsupported sentiment engine')
