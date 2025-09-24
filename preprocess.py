#!/usr/bin/env python3
"""
Data Preprocessing and Feature Engineering Pipeline for TitanovaX Trading System
Converts raw tick data to engineered features for ML models
"""

import pandas as pd
import numpy as np
from pathlib import Path
import talib
from datetime import datetime, timedelta
import json
from typing import Dict, List, Tuple, Optional
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/logs/preprocessing.log'),
        logging.StreamHandler()
    ]
)

class DataPreprocessor:
    def __init__(self, data_dir='data', output_dir='data/processed'):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.timeframes = {
            '1m': '1Min',
            '5m': '5Min',
            '15m': '15Min',
            '30m': '30Min',
            '1h': '1H',
            '4h': '4H',
            '1d': '1D'
        }

    def load_tick_data(self, symbol: str, data_type: str = 'fx') -> pd.DataFrame:
        """Load tick data for a symbol"""

        if data_type == 'fx':
            data_path = self.data_dir / 'raw' / 'fx' / symbol
        else:
            data_path = self.data_dir / 'raw' / 'crypto' / symbol.lower()

        if not data_path.exists():
            raise FileNotFoundError(f"No data found for {symbol} at {data_path}")

        # Load all parquet files for this symbol
        all_files = list(data_path.rglob('*.parquet'))

        if not all_files:
            raise FileNotFoundError(f"No parquet files found for {symbol}")

        # Read and combine all files
        dfs = []
        for file in all_files:
            try:
                df = pd.read_parquet(file)
                dfs.append(df)
            except Exception as e:
                logging.warning(f"Failed to read {file}: {e}")

        if not dfs:
            raise ValueError(f"No valid data files found for {symbol}")

        combined_df = pd.concat(dfs, ignore_index=True)

        # Sort by timestamp and remove duplicates
        combined_df = combined_df.sort_values('timestamp').drop_duplicates()

        logging.info(f"Loaded {len(combined_df)} ticks for {symbol}")
        return combined_df

    def ticks_to_bars(self, tick_df: pd.DataFrame, timeframe: str = '1m') -> pd.DataFrame:
        """Convert tick data to OHLCV bars"""

        if tick_df.empty:
            return pd.DataFrame()

        # Set timestamp as index
        tick_df = tick_df.set_index('timestamp')

        # Handle different data formats
        if 'mid_price' in tick_df.columns:
            price_col = 'mid_price'
        elif 'price' in tick_df.columns:
            price_col = 'price'
        elif 'ask' in tick_df.columns and 'bid' in tick_df.columns:
            price_col = 'ask'  # Use ask for OHLC
            tick_df['price'] = tick_df['ask']
        else:
            raise ValueError("No price column found in tick data")

        # Resample to create bars
        resampled = tick_df[price_col].resample(timeframe).ohlc()

        # Calculate volume (if available)
        if 'quantity' in tick_df.columns:
            volume = tick_df['quantity'].resample(timeframe).sum()
            resampled['volume'] = volume

        # Calculate spread (if available)
        if 'spread' in tick_df.columns:
            spread = tick_df['spread'].resample(timeframe).mean()
            resampled['spread'] = spread

        # Calculate number of trades
        if 'trade_id' in tick_df.columns:
            trade_count = tick_df['trade_id'].resample(timeframe).count()
            resampled['trade_count'] = trade_count

        resampled = resampled.dropna()

        logging.info(f"Created {len(resampled)} {timeframe} bars from {len(tick_df)} ticks")
        return resampled

    def calculate_technical_indicators(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""

        df = bars.copy()

        # Basic price indicators
        if 'close' in df.columns:
            # Moving averages
            df['sma_5'] = df['close'].rolling(window=5).mean()
            df['sma_10'] = df['close'].rolling(window=10).mean()
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()

            # Exponential moving averages
            df['ema_8'] = df['close'].ewm(span=8).mean()
            df['ema_21'] = df['close'].ewm(span=21).mean()
            df['ema_50'] = df['close'].ewm(span=50).mean()

            # RSI
            df['rsi_14'] = self.calculate_rsi(df['close'], 14)

            # MACD
            df['macd'], df['macd_signal'], df['macd_hist'] = self.calculate_macd(df['close'])

            # Bollinger Bands
            df['bb_upper'], df['bb_middle'], df['bb_lower'] = self.calculate_bollinger_bands(df['close'])

            # ATR (Average True Range)
            df['atr_14'] = self.calculate_atr(df, 14)

        # Volume indicators
        if 'volume' in df.columns:
            df['volume_sma_10'] = df['volume'].rolling(window=10).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma_10']

        # Price momentum
        if 'close' in df.columns:
            df['momentum_5'] = df['close'].pct_change(5)
            df['momentum_10'] = df['close'].pct_change(10)
            df['momentum_20'] = df['close'].pct_change(20)

        # Volatility
        if 'close' in df.columns:
            df['volatility_10'] = df['close'].pct_change().rolling(10).std()
            df['volatility_20'] = df['close'].pct_change().rolling(20).std()

        return df

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        try:
            return talib.RSI(prices.values, timeperiod=period)
        except:
            # Fallback implementation
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))

    def calculate_macd(self, prices: pd.Series, fast=12, slow=26, signal=9):
        """Calculate MACD indicator"""
        try:
            macd, signal, hist = talib.MACD(prices.values, fastperiod=fast, slowperiod=slow, signalperiod=signal)
            return macd, signal, hist
        except:
            # Fallback implementation
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            macd = ema_fast - ema_slow
            signal_line = macd.ewm(span=signal).mean()
            histogram = macd - signal_line
            return macd, signal_line, histogram

    def calculate_bollinger_bands(self, prices: pd.Series, period=20, std_dev=2):
        """Calculate Bollinger Bands"""
        try:
            upper, middle, lower = talib.BBANDS(prices.values, timeperiod=period, nbdevup=std_dev, nbdevdn=std_dev, matype=0)
            return upper, middle, lower
        except:
            # Fallback implementation
            middle = prices.rolling(period).mean()
            std = prices.rolling(period).std()
            upper = middle + (std * std_dev)
            lower = middle - (std * std_dev)
            return upper, middle, lower

    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        try:
            return talib.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=period)
        except:
            # Fallback implementation
            high = df['high']
            low = df['low']
            close = df['close']

            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())

            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            return tr.rolling(period).mean()

    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""

        if not isinstance(df.index, pd.DatetimeIndex):
            return df

        df = df.copy()

        # Time of day
        df['hour'] = df.index.hour
        df['minute'] = df.index.minute
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter

        # Time-based flags
        df['is_market_open'] = df['hour'].between(0, 23)  # 24/7 for crypto, adjust for FX
        df['is_weekend'] = df['day_of_week'].isin([5, 6])

        # Session indicators (simplified)
        df['is_asian_session'] = df['hour'].between(0, 8)
        df['is_european_session'] = df['hour'].between(8, 16)
        df['is_us_session'] = df['hour'].between(16, 23)

        # Market volatility patterns
        df['is_high_volatility_period'] = df['hour'].isin([0, 1, 2, 14, 15, 16])  # Common high vol periods

        return df

    def add_market_regime_features(self, df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
        """Add market regime detection features"""

        df = df.copy()

        if 'close' in df.columns:
            # Trend strength
            df['trend_strength'] = self.calculate_trend_strength(df['close'], lookback)

            # Volatility regime
            df['volatility_regime'] = self.calculate_volatility_regime(df['close'], lookback)

            # Volume regime
            if 'volume' in df.columns:
                df['volume_regime'] = self.calculate_volume_regime(df['volume'], lookback)

            # Price momentum regime
            df['momentum_regime'] = self.calculate_momentum_regime(df['close'], lookback)

        return df

    def calculate_trend_strength(self, prices: pd.Series, lookback: int) -> pd.Series:
        """Calculate trend strength (0-1, higher = stronger trend)"""
        # Simple linear regression slope over lookback period
        def trend_slope(x):
            y = np.arange(len(x))
            if len(x) < 2:
                return 0
            slope = np.polyfit(y, x, 1)[0]
            return abs(slope) / x.std() if x.std() > 0 else 0

        return prices.rolling(lookback).apply(trend_slope, raw=False)

    def calculate_volatility_regime(self, prices: pd.Series, lookback: int) -> pd.Series:
        """Calculate volatility regime (0-1, higher = more volatile)"""
        returns = prices.pct_change()
        vol = returns.rolling(lookback).std()
        return vol / vol.rolling(lookback * 4).mean()  # Normalized volatility

    def calculate_volume_regime(self, volume: pd.Series, lookback: int) -> pd.Series:
        """Calculate volume regime (0-1, higher = higher volume)"""
        vol_avg = volume.rolling(lookback).mean()
        vol_std = volume.rolling(lookback).std()
        return volume / (vol_avg + vol_std)  # Z-score like measure

    def calculate_momentum_regime(self, prices: pd.Series, lookback: int) -> pd.Series:
        """Calculate momentum regime (directional bias)"""
        momentum = prices.pct_change(lookback)
        return (momentum > 0).astype(int)  # 1 for up, 0 for down

    def create_multi_horizon_labels(self, df: pd.DataFrame, horizons: List[str] = ['5m', '30m', '1h', '24h']) -> pd.DataFrame:
        """Create labels for multiple prediction horizons"""

        df = df.copy()

        if 'close' not in df.columns:
            return df

        # Forward returns for each horizon
        for horizon in horizons:
            if horizon == '5m':
                periods = 5
            elif horizon == '30m':
                periods = 30
            elif horizon == '1h':
                periods = 60
            elif horizon == '24h':
                periods = 1440  # 24 hours in minutes
            else:
                continue

            # Future return
            future_returns = df['close'].pct_change(periods).shift(-periods)
            df[f'future_return_{horizon}'] = future_returns

            # Binary classification labels (up/down)
            df[f'label_up_{horizon}'] = (future_returns > 0).astype(int)
            df[f'label_down_{horizon}'] = (future_returns < 0).astype(int)

            # Multi-class labels (strong up, up, neutral, down, strong down)
            df[f'label_trend_{horizon}'] = pd.cut(
                future_returns,
                bins=[-np.inf, -0.01, -0.002, 0.002, 0.01, np.inf],
                labels=[-2, -1, 0, 1, 2]
            )

        return df

    def create_spike_labels(self, df: pd.DataFrame, threshold: float = 0.005) -> pd.DataFrame:
        """Create labels for price spike detection"""

        df = df.copy()

        if 'close' not in df.columns:
            return df

        # Calculate returns
        returns = df['close'].pct_change()

        # Spike detection (large moves in short time)
        df['price_spike_up'] = (returns > threshold).astype(int)
        df['price_spike_down'] = (returns < -threshold).astype(int)

        # Volume spikes
        if 'volume' in df.columns:
            volume_avg = df['volume'].rolling(20).mean()
            df['volume_spike'] = (df['volume'] > volume_avg * 2).astype(int)

        return df

    def process_symbol(self, symbol: str, data_type: str = 'fx', timeframes: List[str] = None) -> Dict:
        """Process all data for a single symbol"""

        if timeframes is None:
            timeframes = ['1m', '5m', '15m', '1h']

        logging.info(f"Processing {symbol}...")

        try:
            # Load tick data
            tick_df = self.load_tick_data(symbol, data_type)

            results = {}

            for timeframe in timeframes:
                logging.info(f"Converting {symbol} to {timeframe} bars...")

                # Convert to bars
                bars = self.ticks_to_bars(tick_df, timeframe)

                if bars.empty:
                    logging.warning(f"No bars created for {symbol} {timeframe}")
                    continue

                # Calculate technical indicators
                bars = self.calculate_technical_indicators(bars)

                # Add time features
                bars = self.add_time_features(bars)

                # Add market regime features
                bars = self.add_market_regime_features(bars)

                # Create labels
                bars = self.create_multi_horizon_labels(bars)
                bars = self.create_spike_labels(bars)

                # Save processed data
                output_file = self.output_dir / f"{symbol}_{timeframe}_processed.parquet"
                bars.to_parquet(output_file)

                results[timeframe] = {
                    'file': str(output_file),
                    'rows': len(bars),
                    'columns': len(bars.columns),
                    'date_range': {
                        'start': bars.index.min().isoformat(),
                        'end': bars.index.max().isoformat()
                    }
                }

                logging.info(f"Saved {symbol} {timeframe}: {len(bars)} rows, {len(bars.columns)} columns")

            return {
                'symbol': symbol,
                'success': True,
                'results': results
            }

        except Exception as e:
            logging.error(f"Error processing {symbol}: {e}")
            return {
                'symbol': symbol,
                'success': False,
                'error': str(e)
            }

    def process_all_symbols(self, symbols: List[str] = None, data_type: str = 'fx', max_workers: int = 4) -> Dict:
        """Process all symbols in parallel"""

        if symbols is None:
            if data_type == 'fx':
                symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD']
            else:
                symbols = ['btcusdt', 'ethusdt', 'bnbusdt']

        logging.info(f"Starting batch processing of {len(symbols)} symbols...")

        results = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {
                executor.submit(self.process_symbol, symbol, data_type): symbol
                for symbol in symbols
            }

            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    result = future.result()
                    results[symbol] = result
                    logging.info(f"Completed {symbol}: {'Success' if result['success'] else 'Failed'}")

                except Exception as e:
                    logging.error(f"Error processing {symbol}: {e}")
                    results[symbol] = {
                        'symbol': symbol,
                        'success': False,
                        'error': str(e)
                    }

        # Create summary
        successful = [r for r in results.values() if r['success']]
        failed = [r for r in results.values() if not r['success']]

        summary = {
            'total_symbols': len(symbols),
            'successful': len(successful),
            'failed': len(failed),
            'total_files_created': sum(len(r.get('results', {})) for r in successful),
            'results': results
        }

        # Save summary
        summary_file = self.output_dir / 'processing_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        logging.info(f"Batch processing complete: {len(successful)}/{len(symbols)} symbols successful")
        return summary

def main():
    """Main function for data preprocessing"""

    print("=== TitanovaX Data Preprocessing Pipeline ===")

    preprocessor = DataPreprocessor()

    # Process FX symbols
    print("\nProcessing FX symbols...")
    fx_results = preprocessor.process_all_symbols(
        symbols=['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD'],
        data_type='fx',
        max_workers=2
    )

    # Process crypto symbols
    print("\nProcessing crypto symbols...")
    crypto_results = preprocessor.process_all_symbols(
        symbols=['btcusdt', 'ethusdt'],
        data_type='crypto',
        max_workers=2
    )

    print("\n=== Processing Complete ===")
    print(f"FX: {fx_results['successful']}/{fx_results['total_symbols']} symbols processed")
    print(f"Crypto: {crypto_results['successful']}/{crypto_results['total_symbols']} symbols processed")

    print("\nProcessed data saved to: data/processed/")
    print("Summary saved to: data/processed/processing_summary.json")

if __name__ == "__main__":
    main()
