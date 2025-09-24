#!/usr/bin/env python3
"""
Technical Indicators Module for ML Brain
Advanced technical indicators for feature engineering
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from scipy import signal
from scipy.stats import zscore

@dataclass
class TechnicalIndicators:
    """Technical indicators data class"""
    rsi: float
    macd: float
    macd_signal: float
    macd_histogram: float
    bollinger_upper: float
    bollinger_lower: float
    bollinger_position: float
    stochastic_k: float
    stochastic_d: float
    williams_r: float
    adx: float
    cci: float
    atr: float
    volume_sma: float
    volume_ratio: float
    price_sma_ratio: float
    volatility: float
    momentum: float
    
class TechnicalIndicatorsCalculator:
    """Advanced technical indicators calculator"""
    
    def __init__(self):
        self.logger = None
    
    def calculate_all(self, prices: np.ndarray, volumes: np.ndarray, 
                     highs: np.ndarray, lows: np.ndarray) -> TechnicalIndicators:
        """Calculate all technical indicators"""
        
        # RSI
        rsi = self.calculate_rsi(prices)
        
        # MACD
        macd, macd_signal, macd_histogram = self.calculate_macd(prices)
        
        # Bollinger Bands
        bb_upper, bb_lower, bb_position = self.calculate_bollinger_bands(prices)
        
        # Stochastic
        stoch_k, stoch_d = self.calculate_stochastic(highs, lows, prices)
        
        # Williams %R
        williams_r = self.calculate_williams_r(highs, lows, prices)
        
        # ADX
        adx = self.calculate_adx(highs, lows, prices)
        
        # CCI
        cci = self.calculate_cci(highs, lows, prices)
        
        # ATR
        atr = self.calculate_atr(highs, lows, prices)
        
        # Volume indicators
        volume_sma, volume_ratio = self.calculate_volume_indicators(volumes)
        
        # Price ratios
        price_sma_ratio = self.calculate_price_sma_ratio(prices)
        
        # Volatility
        volatility = self.calculate_volatility(prices)
        
        # Momentum
        momentum = self.calculate_momentum(prices)
        
        return TechnicalIndicators(
            rsi=rsi,
            macd=macd,
            macd_signal=macd_signal,
            macd_histogram=macd_histogram,
            bollinger_upper=bb_upper,
            bollinger_lower=bb_lower,
            bollinger_position=bb_position,
            stochastic_k=stoch_k,
            stochastic_d=stoch_d,
            williams_r=williams_r,
            adx=adx,
            cci=cci,
            atr=atr,
            volume_sma=volume_sma,
            volume_ratio=volume_ratio,
            price_sma_ratio=price_sma_ratio,
            volatility=volatility,
            momentum=momentum
        )
    
    def calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI indicator"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        seed = deltas[:period]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        
        if down == 0:
            return 100.0
        
        rs = up / down
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[float, float, float]:
        """Calculate MACD indicator"""
        exp1 = pd.Series(prices).ewm(span=fast).mean()
        exp2 = pd.Series(prices).ewm(span=slow).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        
        return macd_line.iloc[-1], signal_line.iloc[-1], histogram.iloc[-1]
    
    def calculate_bollinger_bands(self, prices: np.ndarray, period: int = 20, std_dev: int = 2) -> Tuple[float, float, float]:
        """Calculate Bollinger Bands"""
        sma = pd.Series(prices).rolling(window=period).mean()
        std = pd.Series(prices).rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        current_price = prices[-1]
        position = (current_price - lower_band.iloc[-1]) / (upper_band.iloc[-1] - lower_band.iloc[-1])
        
        return upper_band.iloc[-1], lower_band.iloc[-1], position
    
    def calculate_stochastic(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> Tuple[float, float]:
        """Calculate Stochastic indicator"""
        if len(highs) < period:
            return 50.0, 50.0
        
        high_max = np.max(highs[-period:])
        low_min = np.min(lows[-period:])
        
        if high_max == low_min:
            return 50.0, 50.0
        
        k = 100 * (closes[-1] - low_min) / (high_max - low_min)
        d = pd.Series([k]).rolling(window=3).mean().iloc[-1]
        
        return k, d
    
    def calculate_williams_r(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
        """Calculate Williams %R"""
        if len(highs) < period:
            return -50.0
        
        high_max = np.max(highs[-period:])
        low_min = np.min(lows[-period:])
        
        if high_max == low_min:
            return -50.0
        
        williams_r = -100 * (high_max - closes[-1]) / (high_max - low_min)
        return williams_r
    
    def calculate_adx(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
        """Calculate ADX indicator"""
        if len(highs) < period + 1:
            return 25.0
        
        # Calculate True Range
        high_low = highs - lows
        high_close = np.abs(highs - np.roll(closes, 1))
        low_close = np.abs(lows - np.roll(closes, 1))
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = pd.Series(true_range).rolling(window=period).mean().iloc[-1]
        
        # Calculate Directional Movement
        up_move = highs - np.roll(highs, 1)
        down_move = np.roll(lows, 1) - lows
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        plus_di = 100 * (pd.Series(plus_dm).rolling(window=period).mean().iloc[-1] / atr)
        minus_di = 100 * (pd.Series(minus_dm).rolling(window=period).mean().iloc[-1] / atr)
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = pd.Series([dx]).rolling(window=period).mean().iloc[-1]
        
        return adx
    
    def calculate_cci(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 20) -> float:
        """Calculate Commodity Channel Index"""
        if len(highs) < period:
            return 0.0
        
        typical_price = (highs + lows + closes) / 3
        sma_tp = pd.Series(typical_price).rolling(window=period).mean().iloc[-1]
        mean_deviation = pd.Series(np.abs(typical_price - sma_tp)).rolling(window=period).mean().iloc[-1]
        
        if mean_deviation == 0:
            return 0.0
        
        cci = (typical_price[-1] - sma_tp) / (0.015 * mean_deviation)
        return cci
    
    def calculate_atr(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
        """Calculate Average True Range"""
        if len(highs) < period + 1:
            return 0.001
        
        high_low = highs - lows
        high_close = np.abs(highs - np.roll(closes, 1))
        low_close = np.abs(lows - np.roll(closes, 1))
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = pd.Series(true_range).rolling(window=period).mean().iloc[-1]
        
        return atr
    
    def calculate_volume_indicators(self, volumes: np.ndarray, period: int = 20) -> Tuple[float, float]:
        """Calculate volume indicators"""
        if len(volumes) < period:
            return volumes[-1], 1.0
        
        volume_sma = pd.Series(volumes).rolling(window=period).mean().iloc[-1]
        volume_ratio = volumes[-1] / volume_sma if volume_sma > 0 else 1.0
        
        return volume_sma, volume_ratio
    
    def calculate_price_sma_ratio(self, prices: np.ndarray, period: int = 20) -> float:
        """Calculate price to SMA ratio"""
        if len(prices) < period:
            return 1.0
        
        sma = pd.Series(prices).rolling(window=period).mean().iloc[-1]
        return prices[-1] / sma if sma > 0 else 1.0
    
    def calculate_volatility(self, prices: np.ndarray, period: int = 20) -> float:
        """Calculate price volatility"""
        if len(prices) < period:
            return 0.01
        
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns[-period:])
        return volatility
    
    def calculate_momentum(self, prices: np.ndarray, period: int = 10) -> float:
        """Calculate price momentum"""
        if len(prices) < period + 1:
            return 0.0
        
        momentum = (prices[-1] - prices[-period]) / prices[-period]
        return momentum