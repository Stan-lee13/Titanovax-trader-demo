"""
Feature builder module for ML trading system.

This module provides pure functions for extracting technical and market features
from financial time series data. All functions are deterministic and side-effect free.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
import hashlib
import json


def calculate_rolling_returns(df: pd.DataFrame, periods: list = [1, 5, 15, 60]) -> pd.DataFrame:
    """
    Calculate rolling returns over specified periods.

    Args:
        df: DataFrame with OHLCV data and 'close' column
        periods: List of periods to calculate returns for

    Returns:
        DataFrame with additional return columns
    """
    result_df = df.copy()

    for period in periods:
        col_name = f'return_{period}'
        result_df[col_name] = result_df['close'].pct_change(periods=period)

    return result_df


def calculate_ema_diff(df: pd.DataFrame, short_period: int = 12, long_period: int = 26) -> pd.DataFrame:
    """
    Calculate difference between short and long EMA.

    Args:
        df: DataFrame with 'close' column
        short_period: Period for short EMA
        long_period: Period for long EMA

    Returns:
        DataFrame with EMA difference column
    """
    result_df = df.copy()

    short_ema = result_df['close'].ewm(span=short_period, adjust=False).mean()
    long_ema = result_df['close'].ewm(span=long_period, adjust=False).mean()

    result_df['ema_diff'] = short_ema - long_ema
    result_df['ema_diff_pct'] = (short_ema - long_ema) / long_ema

    return result_df


def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Calculate Relative Strength Index (RSI).

    Args:
        df: DataFrame with 'close' column
        period: RSI calculation period

    Returns:
        DataFrame with RSI column
    """
    result_df = df.copy()

    # Calculate price changes
    delta = result_df['close'].diff()

    # Separate gains and losses
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)

    # Calculate average gains and losses
    avg_gain = gains.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1/period, adjust=False).mean()

    # Calculate RS and RSI
    rs = avg_gain / avg_loss
    result_df['rsi'] = 100 - (100 / (1 + rs))

    return result_df


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Calculate Average True Range (ATR).

    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        period: ATR calculation period

    Returns:
        DataFrame with ATR column
    """
    result_df = df.copy()

    # Calculate True Range
    high_low = result_df['high'] - result_df['low']
    high_close = np.abs(result_df['high'] - result_df['close'].shift())
    low_close = np.abs(result_df['low'] - result_df['close'].shift())

    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    # Calculate ATR
    result_df['atr'] = tr.ewm(alpha=1/period, adjust=False).mean()

    return result_df


def calculate_volume_zscore(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Calculate volume z-score.

    Args:
        df: DataFrame with 'volume' column
        window: Rolling window for z-score calculation

    Returns:
        DataFrame with volume z-score column
    """
    result_df = df.copy()

    rolling_mean = result_df['volume'].rolling(window=window).mean()
    rolling_std = result_df['volume'].rolling(window=window).std()

    result_df['volume_zscore'] = (result_df['volume'] - rolling_mean) / rolling_std

    return result_df


def calculate_tick_imbalance(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Calculate tick imbalance as a proxy for order flow.

    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        window: Rolling window for imbalance calculation

    Returns:
        DataFrame with tick imbalance column
    """
    result_df = df.copy()

    # Calculate up ticks and down ticks
    up_ticks = (result_df['close'] > result_df['close'].shift()).astype(int)
    down_ticks = (result_df['close'] < result_df['close'].shift()).astype(int)

    # Calculate imbalance
    imbalance = (up_ticks - down_ticks).rolling(window=window).sum()

    result_df['tick_imbalance'] = imbalance

    return result_df


def calculate_volatility_normals(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Calculate normalized volatility measures.

    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        window: Rolling window for volatility calculation

    Returns:
        DataFrame with volatility normal columns
    """
    result_df = df.copy()

    # Calculate daily range
    daily_range = (result_df['high'] - result_df['low']) / result_df['close']

    # Calculate normalized volatility
    vol_mean = daily_range.rolling(window=window).mean()
    vol_std = daily_range.rolling(window=window).std()

    result_df['volatility_normal'] = (daily_range - vol_mean) / vol_std

    return result_df


def calculate_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate time-of-day features.

    Args:
        df: DataFrame with datetime index

    Returns:
        DataFrame with time-based feature columns
    """
    result_df = df.copy()

    # Assuming datetime index
    if isinstance(result_df.index, pd.DatetimeIndex):
        result_df['hour'] = result_df.index.hour
        result_df['day_of_week'] = result_df.index.dayofweek
        result_df['month'] = result_df.index.month
        result_df['quarter'] = result_df.index.quarter
        result_df['is_weekend'] = result_df.index.dayofweek >= 5
        result_df['is_market_open'] = result_df['hour'].between(9, 16)  # Assuming forex hours

    return result_df


def calculate_macro_event_flags(df: pd.DataFrame, news_events: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Calculate macro event flags based on news events.

    Args:
        df: DataFrame with datetime index
        news_events: DataFrame with news events (timestamp, impact, type)

    Returns:
        DataFrame with macro event flag columns
    """
    result_df = df.copy()

    # Initialize flags
    result_df['high_impact_news'] = 0
    result_df['medium_impact_news'] = 0
    result_df['economic_announcement'] = 0
    result_df['central_bank_event'] = 0

    if news_events is not None and not news_events.empty:
        # Convert news timestamps to match data frequency
        for idx, row in df.iterrows():
            # Check for news events in the current period
            matching_news = news_events[
                (news_events['timestamp'] >= idx) &
                (news_events['timestamp'] < idx + pd.Timedelta(minutes=1))  # Assuming 1-minute data
            ]

            if not matching_news.empty:
                for _, news in matching_news.iterrows():
                    if news['impact'] == 'high':
                        result_df.loc[idx, 'high_impact_news'] = 1
                    elif news['impact'] == 'medium':
                        result_df.loc[idx, 'medium_impact_news'] = 1

                    if news['type'] == 'economic':
                        result_df.loc[idx, 'economic_announcement'] = 1
                    elif news['type'] == 'central_bank':
                        result_df.loc[idx, 'central_bank_event'] = 1

    return result_df


def calculate_deterministic_hash(features_dict: Dict[str, Any]) -> str:
    """
    Calculate deterministic hash of features for reproducibility.

    Args:
        features_dict: Dictionary containing feature parameters

    Returns:
        SHA256 hash string
    """
    # Convert to JSON string for deterministic serialization
    features_json = json.dumps(features_dict, sort_keys=True, separators=(',', ':'))

    # Calculate SHA256 hash
    hash_obj = hashlib.sha256()
    hash_obj.update(features_json.encode('utf-8'))

    return f"sha256:{hash_obj.hexdigest()[:16]}"  # Return short hash


def build_features(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    news_events: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Build comprehensive feature set from OHLCV data.

    Args:
        df: DataFrame with OHLCV columns (open, high, low, close, volume)
        symbol: Trading symbol (e.g., 'EURUSD')
        timeframe: Timeframe string (e.g., 'M1', 'M5')
        start_date: Start date for feature calculation
        end_date: End date for feature calculation
        news_events: Optional DataFrame with news events

    Returns:
        DataFrame with all calculated features
    """
    # Filter data by date if specified
    working_df = df.copy()
    if start_date:
        working_df = working_df[working_df.index >= start_date]
    if end_date:
        working_df = working_df[working_df.index <= end_date]

    # Apply all feature calculations in sequence
    features_df = working_df.copy()

    # Technical indicators
    features_df = calculate_rolling_returns(features_df)
    features_df = calculate_ema_diff(features_df)
    features_df = calculate_rsi(features_df)
    features_df = calculate_atr(features_df)
    features_df = calculate_volume_zscore(features_df)
    features_df = calculate_tick_imbalance(features_df)
    features_df = calculate_volatility_normals(features_df)
    features_df = calculate_time_features(features_df)
    features_df = calculate_macro_event_flags(features_df, news_events)

    # Calculate feature hash
    feature_params = {
        'symbol': symbol,
        'timeframe': timeframe,
        'start_date': start_date,
        'end_date': end_date,
        'feature_functions': [
            'rolling_returns', 'ema_diff', 'rsi', 'atr',
            'volume_zscore', 'tick_imbalance', 'volatility_normals',
            'time_features', 'macro_event_flags'
        ]
    }

    features_df.attrs['features_hash'] = calculate_deterministic_hash(feature_params)

    return features_df


def validate_features_df(df: pd.DataFrame) -> Tuple[bool, list]:
    """
    Validate that features DataFrame has required columns and no NaN values.

    Args:
        df: Features DataFrame to validate

    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []

    # Check for required columns
    required_columns = [
        'return_1', 'return_5', 'return_15', 'return_60',
        'ema_diff', 'rsi', 'atr', 'volume_zscore',
        'tick_imbalance', 'volatility_normal', 'hour',
        'day_of_week', 'high_impact_news'
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        errors.append(f"Missing required columns: {missing_columns}")

    # Check for NaN values in key features
    nan_columns = df[required_columns].columns[df[required_columns].isna().any()].tolist()
    if nan_columns:
        errors.append(f"NaN values found in columns: {nan_columns}")

    # Check for infinite values
    inf_columns = df[required_columns].columns[np.isinf(df[required_columns]).any()].tolist()
    if inf_columns:
        errors.append(f"Infinite values found in columns: {inf_columns}")

    return len(errors) == 0, errors
