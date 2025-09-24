"""
Unit tests for feature_builder module.

Tests all feature calculation functions to ensure they are pure and deterministic.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from ml_brain.features.feature_builder import (
    calculate_rolling_returns,
    calculate_ema_diff,
    calculate_rsi,
    calculate_atr,
    calculate_volume_zscore,
    calculate_tick_imbalance,
    calculate_volatility_normals,
    calculate_time_features,
    calculate_macro_event_flags,
    calculate_deterministic_hash,
    build_features,
    validate_features_df
)


@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data for testing."""
    np.random.seed(42)  # For deterministic tests

    # Create 100 periods of sample data
    dates = pd.date_range(start='2023-01-01', periods=100, freq='1min')

    data = {
        'open': np.random.uniform(1.0, 1.1, 100),
        'high': np.random.uniform(1.05, 1.15, 100),
        'low': np.random.uniform(0.95, 1.05, 100),
        'close': np.random.uniform(1.0, 1.1, 100),
        'volume': np.random.uniform(1000, 10000, 100)
    }

    df = pd.DataFrame(data, index=dates)

    # Make sure high > low and close is between high/low
    df['high'] = df[['open', 'close']].max(axis=1) + np.random.uniform(0.001, 0.01, 100)
    df['low'] = df[['open', 'close']].min(axis=1) - np.random.uniform(0.001, 0.01, 100)
    df['close'] = df['close'].clip(lower=df['low'], upper=df['high'])

    return df


@pytest.fixture
def sample_news_data():
    """Create sample news events data for testing."""
    return pd.DataFrame({
        'timestamp': pd.to_datetime([
            '2023-01-01 09:30:00',
            '2023-01-01 10:15:00',
            '2023-01-01 14:30:00'
        ]),
        'impact': ['high', 'medium', 'high'],
        'type': ['economic', 'central_bank', 'economic']
    })


class TestRollingReturns:
    """Test rolling returns calculation."""

    def test_calculate_rolling_returns_basic(self, sample_ohlcv_data):
        """Test basic rolling returns calculation."""
        result = calculate_rolling_returns(sample_ohlcv_data)

        # Check that return columns are added
        expected_columns = ['return_1', 'return_5', 'return_15', 'return_60']
        for col in expected_columns:
            assert col in result.columns

        # Check that first few values are NaN (due to pct_change)
        assert pd.isna(result['return_1'].iloc[0])
        assert pd.isna(result['return_5'].iloc[4])

        # Check that later values are calculated
        assert not pd.isna(result['return_1'].iloc[10])

    def test_calculate_rolling_returns_custom_periods(self, sample_ohlcv_data):
        """Test rolling returns with custom periods."""
        custom_periods = [2, 10]
        result = calculate_rolling_returns(sample_ohlcv_data, periods=custom_periods)

        expected_columns = ['return_2', 'return_10']
        for col in expected_columns:
            assert col in result.columns

    def test_calculate_rolling_returns_pure_function(self, sample_ohlcv_data):
        """Test that function is pure (no side effects)."""
        original_df = sample_ohlcv_data.copy()
        result = calculate_rolling_returns(sample_ohlcv_data)

        # Original DataFrame should be unchanged
        pd.testing.assert_frame_equal(sample_ohlcv_data, original_df)


class TestEMADiff:
    """Test EMA difference calculation."""

    def test_calculate_ema_diff_basic(self, sample_ohlcv_data):
        """Test basic EMA difference calculation."""
        result = calculate_ema_diff(sample_ohlcv_data)

        # Check that EMA columns are added
        assert 'ema_diff' in result.columns
        assert 'ema_diff_pct' in result.columns

        # Check that values are reasonable
        assert not result['ema_diff'].isna().all()
        assert result['ema_diff_pct'].between(-1, 1).all()  # Should be reasonable percentages

    def test_calculate_ema_diff_custom_periods(self, sample_ohlcv_data):
        """Test EMA difference with custom periods."""
        result = calculate_ema_diff(sample_ohlcv_data, short_period=5, long_period=10)

        assert 'ema_diff' in result.columns
        assert 'ema_diff_pct' in result.columns

    def test_calculate_ema_diff_pure_function(self, sample_ohlcv_data):
        """Test that function is pure."""
        original_df = sample_ohlcv_data.copy()
        result = calculate_ema_diff(sample_ohlcv_data)

        pd.testing.assert_frame_equal(sample_ohlcv_data, original_df)


class TestRSI:
    """Test RSI calculation."""

    def test_calculate_rsi_basic(self, sample_ohlcv_data):
        """Test basic RSI calculation."""
        result = calculate_rsi(sample_ohlcv_data)

        assert 'rsi' in result.columns

        # RSI should be between 0 and 100
        assert result['rsi'].between(0, 100).all()

        # Check that early values are NaN
        assert pd.isna(result['rsi'].iloc[13])  # RSI needs 14 periods

    def test_calculate_rsi_custom_period(self, sample_ohlcv_data):
        """Test RSI with custom period."""
        result = calculate_rsi(sample_ohlcv_data, period=7)

        assert 'rsi' in result.columns
        assert result['rsi'].between(0, 100).all()

    def test_calculate_rsi_pure_function(self, sample_ohlcv_data):
        """Test that function is pure."""
        original_df = sample_ohlcv_data.copy()
        result = calculate_rsi(sample_ohlcv_data)

        pd.testing.assert_frame_equal(sample_ohlcv_data, original_df)


class TestATR:
    """Test ATR calculation."""

    def test_calculate_atr_basic(self, sample_ohlcv_data):
        """Test basic ATR calculation."""
        result = calculate_atr(sample_ohlcv_data)

        assert 'atr' in result.columns
        assert (result['atr'] >= 0).all()  # ATR should be non-negative

    def test_calculate_atr_custom_period(self, sample_ohlcv_data):
        """Test ATR with custom period."""
        result = calculate_atr(sample_ohlcv_data, period=7)

        assert 'atr' in result.columns
        assert (result['atr'] >= 0).all()

    def test_calculate_atr_pure_function(self, sample_ohlcv_data):
        """Test that function is pure."""
        original_df = sample_ohlcv_data.copy()
        result = calculate_atr(sample_ohlcv_data)

        pd.testing.assert_frame_equal(sample_ohlcv_data, original_df)


class TestVolumeZscore:
    """Test volume z-score calculation."""

    def test_calculate_volume_zscore_basic(self, sample_ohlcv_data):
        """Test basic volume z-score calculation."""
        result = calculate_volume_zscore(sample_ohlcv_data)

        assert 'volume_zscore' in result.columns

        # Check that early values are NaN (due to rolling window)
        assert pd.isna(result['volume_zscore'].iloc[19])  # 20-period window

    def test_calculate_volume_zscore_custom_window(self, sample_ohlcv_data):
        """Test volume z-score with custom window."""
        result = calculate_volume_zscore(sample_ohlcv_data, window=10)

        assert 'volume_zscore' in result.columns
        assert pd.isna(result['volume_zscore'].iloc[9])  # 10-period window

    def test_calculate_volume_zscore_pure_function(self, sample_ohlcv_data):
        """Test that function is pure."""
        original_df = sample_ohlcv_data.copy()
        result = calculate_volume_zscore(sample_ohlcv_data)

        pd.testing.assert_frame_equal(sample_ohlcv_data, original_df)


class TestTickImbalance:
    """Test tick imbalance calculation."""

    def test_calculate_tick_imbalance_basic(self, sample_ohlcv_data):
        """Test basic tick imbalance calculation."""
        result = calculate_tick_imbalance(sample_ohlcv_data)

        assert 'tick_imbalance' in result.columns

        # Values should be reasonable (-window to +window)
        assert result['tick_imbalance'].abs().max() <= 20  # 20-period window

    def test_calculate_tick_imbalance_custom_window(self, sample_ohlcv_data):
        """Test tick imbalance with custom window."""
        result = calculate_tick_imbalance(sample_ohlcv_data, window=10)

        assert 'tick_imbalance' in result.columns
        assert result['tick_imbalance'].abs().max() <= 10

    def test_calculate_tick_imbalance_pure_function(self, sample_ohlcv_data):
        """Test that function is pure."""
        original_df = sample_ohlcv_data.copy()
        result = calculate_tick_imbalance(sample_ohlcv_data)

        pd.testing.assert_frame_equal(sample_ohlcv_data, original_df)


class TestVolatilityNormals:
    """Test volatility normals calculation."""

    def test_calculate_volatility_normals_basic(self, sample_ohlcv_data):
        """Test basic volatility normals calculation."""
        result = calculate_volatility_normals(sample_ohlcv_data)

        assert 'volatility_normal' in result.columns

    def test_calculate_volatility_normals_custom_window(self, sample_ohlcv_data):
        """Test volatility normals with custom window."""
        result = calculate_volatility_normals(sample_ohlcv_data, window=10)

        assert 'volatility_normal' in result.columns

    def test_calculate_volatility_normals_pure_function(self, sample_ohlcv_data):
        """Test that function is pure."""
        original_df = sample_ohlcv_data.copy()
        result = calculate_volatility_normals(sample_ohlcv_data)

        pd.testing.assert_frame_equal(sample_ohlcv_data, original_df)


class TestTimeFeatures:
    """Test time features calculation."""

    def test_calculate_time_features_basic(self, sample_ohlcv_data):
        """Test basic time features calculation."""
        result = calculate_time_features(sample_ohlcv_data)

        expected_columns = ['hour', 'day_of_week', 'month', 'quarter', 'is_weekend', 'is_market_open']
        for col in expected_columns:
            assert col in result.columns

        # Check hour range
        assert result['hour'].between(0, 23).all()

        # Check day of week range
        assert result['day_of_week'].between(0, 6).all()

    def test_calculate_time_features_pure_function(self, sample_ohlcv_data):
        """Test that function is pure."""
        original_df = sample_ohlcv_data.copy()
        result = calculate_time_features(sample_ohlcv_data)

        pd.testing.assert_frame_equal(sample_ohlcv_data, original_df)


class TestMacroEventFlags:
    """Test macro event flags calculation."""

    def test_calculate_macro_event_flags_basic(self, sample_ohlcv_data, sample_news_data):
        """Test basic macro event flags calculation."""
        result = calculate_macro_event_flags(sample_ohlcv_data, sample_news_data)

        expected_columns = ['high_impact_news', 'medium_impact_news', 'economic_announcement', 'central_bank_event']
        for col in expected_columns:
            assert col in result.columns

        # Check that flags are binary
        assert result['high_impact_news'].isin([0, 1]).all()
        assert result['medium_impact_news'].isin([0, 1]).all()

    def test_calculate_macro_event_flags_no_news(self, sample_ohlcv_data):
        """Test macro event flags with no news events."""
        result = calculate_macro_event_flags(sample_ohlcv_data, None)

        expected_columns = ['high_impact_news', 'medium_impact_news', 'economic_announcement', 'central_bank_event']
        for col in expected_columns:
            assert col in result.columns
            assert (result[col] == 0).all()  # All should be 0

    def test_calculate_macro_event_flags_pure_function(self, sample_ohlcv_data, sample_news_data):
        """Test that function is pure."""
        original_df = sample_ohlcv_data.copy()
        result = calculate_macro_event_flags(sample_ohlcv_data, sample_news_data)

        pd.testing.assert_frame_equal(sample_ohlcv_data, original_df)


class TestDeterministicHash:
    """Test deterministic hash calculation."""

    def test_calculate_deterministic_hash_consistency(self):
        """Test that hash is consistent for same input."""
        params1 = {'symbol': 'EURUSD', 'timeframe': 'M1'}
        params2 = {'symbol': 'EURUSD', 'timeframe': 'M1'}

        hash1 = calculate_deterministic_hash(params1)
        hash2 = calculate_deterministic_hash(params2)

        assert hash1 == hash2

    def test_calculate_deterministic_hash_different_inputs(self):
        """Test that different inputs produce different hashes."""
        params1 = {'symbol': 'EURUSD', 'timeframe': 'M1'}
        params2 = {'symbol': 'GBPUSD', 'timeframe': 'M1'}

        hash1 = calculate_deterministic_hash(params1)
        hash2 = calculate_deterministic_hash(params2)

        assert hash1 != hash2


class TestBuildFeatures:
    """Test main build_features function."""

    def test_build_features_basic(self, sample_ohlcv_data, sample_news_data):
        """Test basic feature building."""
        result = build_features(
            sample_ohlcv_data,
            symbol='EURUSD',
            timeframe='M1',
            news_events=sample_news_data
        )

        # Check that all expected feature columns are present
        expected_features = [
            'return_1', 'return_5', 'return_15', 'return_60',
            'ema_diff', 'ema_diff_pct', 'rsi', 'atr', 'volume_zscore',
            'tick_imbalance', 'volatility_normal', 'hour', 'day_of_week',
            'month', 'quarter', 'is_weekend', 'is_market_open',
            'high_impact_news', 'medium_impact_news', 'economic_announcement', 'central_bank_event'
        ]

        for feature in expected_features:
            assert feature in result.columns

        # Check that features_hash is set
        assert 'features_hash' in result.attrs

    def test_build_features_no_news(self, sample_ohlcv_data):
        """Test feature building without news events."""
        result = build_features(
            sample_ohlcv_data,
            symbol='EURUSD',
            timeframe='M1',
            news_events=None
        )

        # All news flags should be 0
        news_flags = ['high_impact_news', 'medium_impact_news', 'economic_announcement', 'central_bank_event']
        for flag in news_flags:
            assert (result[flag] == 0).all()

    def test_build_features_with_date_filter(self, sample_ohlcv_data, sample_news_data):
        """Test feature building with date filtering."""
        start_date = '2023-01-01 00:30:00'
        end_date = '2023-01-01 00:50:00'

        result = build_features(
            sample_ohlcv_data,
            symbol='EURUSD',
            timeframe='M1',
            start_date=start_date,
            end_date=end_date,
            news_events=sample_news_data
        )

        # Result should be filtered by date
        assert result.index.min() >= pd.to_datetime(start_date)
        assert result.index.max() <= pd.to_datetime(end_date)


class TestValidateFeatures:
    """Test feature validation function."""

    def test_validate_features_df_valid(self, sample_ohlcv_data, sample_news_data):
        """Test validation of valid features DataFrame."""
        features_df = build_features(
            sample_ohlcv_data,
            symbol='EURUSD',
            timeframe='M1',
            news_events=sample_news_data
        )

        is_valid, errors = validate_features_df(features_df)

        assert is_valid
        assert len(errors) == 0

    def test_validate_features_df_missing_columns(self, sample_ohlcv_data):
        """Test validation with missing columns."""
        # Create DataFrame with missing required columns
        incomplete_df = sample_ohlcv_data[['close', 'volume']].copy()

        is_valid, errors = validate_features_df(incomplete_df)

        assert not is_valid
        assert len(errors) > 0
        assert any('Missing required columns' in error for error in errors)

    def test_validate_features_df_with_nan(self):
        """Test validation with NaN values."""
        # Create DataFrame with NaN values in required columns
        df_with_nan = pd.DataFrame({
            'return_1': [1.0, np.nan, 3.0],
            'rsi': [50.0, 60.0, np.nan],
            'volume_zscore': [1.0, 2.0, 3.0]
        })

        is_valid, errors = validate_features_df(df_with_nan)

        assert not is_valid
        assert any('NaN values found' in error for error in errors)


if __name__ == '__main__':
    pytest.main([__file__])
