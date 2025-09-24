#!/usr/bin/env python3
"""
Explainability System for TitanovaX Trading System
Provides post-trade explanations using RAG + LLM integration
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import requests
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import pickle
import logging
from typing import Dict, List, Tuple, Optional
import hashlib
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import base64
from PIL import Image
import io

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/logs/explainability.log'),
        logging.StreamHandler()
    ]
)

class TradeExplainer:
    def __init__(self, models_dir='models', explanations_dir='data/explanations'):
        self.models_dir = Path(models_dir)
        self.explanations_dir = Path(explanations_dir)
        self.explanations_dir.mkdir(parents=True, exist_ok=True)

        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Context database for RAG
        self.context_index = None
        self.context_texts = []
        self.context_embeddings = None

        # Load trade contexts
        self.load_trade_contexts()

    def load_trade_contexts(self):
        """Load trade context database for RAG"""

        context_file = self.explanations_dir / 'trade_contexts.json'

        if context_file.exists():
            with open(context_file, 'r') as f:
                contexts = json.load(f)
        else:
            # Create default contexts
            contexts = {
                'technical_patterns': [
                    "Price broke above resistance level with high volume",
                    "RSI divergence detected - price going down while RSI going up",
                    "Golden cross pattern - 50MA crossed above 200MA",
                    "Price formed double bottom pattern near support level",
                    "Bollinger Bands squeeze followed by expansion",
                    "MACD histogram showing increasing momentum",
                    "Price rejected from overbought RSI levels",
                    "Stochastic oscillator crossover in oversold territory"
                ],
                'market_conditions': [
                    "High volatility during market open",
                    "Low liquidity during Asian session",
                    "News-driven price movement",
                    "End of quarter rebalancing flows",
                    "Central bank announcement impact",
                    "Economic data surprise",
                    "Geopolitical tension affecting markets",
                    "Market maker inventory rebalancing"
                ],
                'risk_factors': [
                    "Wide spread indicating low liquidity",
                    "High volatility may trigger stop losses",
                    "Position size exceeds risk limits",
                    "Correlation with existing positions",
                    "Time of day with historical high volatility",
                    "Weekend gap risk for crypto",
                    "Earnings announcement risk",
                    "Ex-dividend date impact"
                ],
                'strategy_signals': [
                    "Trend following signal confirmed",
                    "Mean reversion opportunity detected",
                    "Momentum breakout confirmed",
                    "Support level bounce expected",
                    "Resistance level rejection likely",
                    "Volatility contraction setup",
                    "Range breakout opportunity",
                    "Fibonacci retracement level"
                ]
            }

            with open(context_file, 'w') as f:
                json.dump(contexts, f, indent=2)

        # Create embeddings for all contexts
        all_texts = []
        all_labels = []

        for category, texts in contexts.items():
            all_texts.extend(texts)
            all_labels.extend([category] * len(texts))

        # Generate embeddings
        embeddings = self.embedding_model.encode(all_texts)

        # Create FAISS index for fast similarity search
        dimension = embeddings.shape[1]
        self.context_index = faiss.IndexFlatIP(dimension)
        self.context_index.add(embeddings.astype(np.float32))

        self.context_texts = all_texts
        self.context_labels = all_labels

        logging.info(f"Loaded {len(all_texts)} context phrases for RAG")

    def get_similar_contexts(self, query: str, top_k: int = 5) -> List[Dict]:
        """Get most similar contexts using RAG"""

        query_embedding = self.embedding_model.encode([query])
        query_embedding = query_embedding.astype(np.float32)

        # Search FAISS index
        scores, indices = self.context_index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if score > 0.3:  # Similarity threshold
                results.append({
                    'text': self.context_texts[idx],
                    'similarity': float(score),
                    'category': self.context_labels[idx]
                })

        return results

    def analyze_technical_factors(self, symbol: str, features: Dict) -> Dict:
        """Analyze technical factors for explanation"""

        analysis = {
            'trend_indicators': [],
            'momentum_indicators': [],
            'volatility_indicators': [],
            'volume_indicators': []
        }

        # Trend analysis
        if 'ema_8' in features and 'ema_21' in features:
            if features['ema_8'] > features['ema_21']:
                analysis['trend_indicators'].append("Short-term EMA above long-term EMA (bullish)")
            else:
                analysis['trend_indicators'].append("Short-term EMA below long-term EMA (bearish)")

        if 'rsi_14' in features:
            rsi = features['rsi_14']
            if rsi > 70:
                analysis['momentum_indicators'].append(f"RSI overbought at {rsi:.1f} (potential reversal)")
            elif rsi < 30:
                analysis['momentum_indicators'].append(f"RSI oversold at {rsi:.1f} (potential bounce)")
            else:
                analysis['momentum_indicators'].append(f"RSI neutral at {rsi:.1f}")

        # MACD analysis
        if 'macd' in features and 'macd_signal' in features:
            macd = features['macd']
            signal = features['macd_signal']

            if macd > signal:
                analysis['momentum_indicators'].append("MACD above signal line (bullish momentum)")
            else:
                analysis['momentum_indicators'].append("MACD below signal line (bearish momentum)")

        # Bollinger Bands
        if all(key in features for key in ['bb_upper', 'bb_middle', 'bb_lower']):
            price = features.get('close', 0)
            upper = features['bb_upper']
            middle = features['bb_middle']
            lower = features['bb_lower']

            if price > upper:
                analysis['volatility_indicators'].append("Price above upper Bollinger Band (overbought)")
            elif price < lower:
                analysis['volatility_indicators'].append("Price below lower Bollinger Band (oversold)")
            else:
                band_width = (upper - lower) / middle
                if band_width < 0.02:
                    analysis['volatility_indicators'].append("Bollinger Bands squeezed (potential breakout)")

        # Volume analysis
        if 'volume' in features and 'volume_sma_10' in features:
            volume = features['volume']
            volume_avg = features['volume_sma_10']

            if volume > volume_avg * 1.5:
                analysis['volume_indicators'].append("Volume spike detected (strong conviction)")
            elif volume < volume_avg * 0.7:
                analysis['volume_indicators'].append("Below average volume (weak participation)")

        # Time analysis
        hour = features.get('hour', 0)
        if 0 <= hour <= 8:
            analysis['trend_indicators'].append("Trading during Asian session (typically lower volatility)")
        elif 8 <= hour <= 16:
            analysis['trend_indicators'].append("Trading during European session (higher volatility)")
        elif 16 <= hour <= 23:
            analysis['trend_indicators'].append("Trading during US session (news-driven moves)")

        return analysis

    def generate_explanation(self, trade_data: Dict, model_prediction: float,
                           features: Dict, market_context: Dict = None) -> str:
        """Generate comprehensive trade explanation"""

        symbol = trade_data.get('symbol', 'UNKNOWN')
        side = trade_data.get('side', 'UNKNOWN')
        confidence = abs(model_prediction - 0.5) * 2

        # Get similar contexts using RAG
        query = f"{side} trade on {symbol} with confidence {confidence:.2f}"
        similar_contexts = self.get_similar_contexts(query, top_k=3)

        # Analyze technical factors
        technical_analysis = self.analyze_technical_factors(symbol, features)

        # Build explanation
        explanation_parts = []

        # Strategy signal
        explanation_parts.append(f"This {side.lower()} signal for {symbol} was generated with {confidence:.2f} confidence.")

        # Model prediction context
        if model_prediction > 0.6:
            explanation_parts.append("The model shows strong bullish conviction based on current market conditions.")
        elif model_prediction < 0.4:
            explanation_parts.append("The model shows strong bearish conviction based on current market conditions.")
        else:
            explanation_parts.append("The model shows moderate confidence with mixed signals.")

        # Technical factors
        if technical_analysis['trend_indicators']:
            explanation_parts.append("Technical indicators suggest: " + "; ".join(technical_analysis['trend_indicators']))

        if technical_analysis['momentum_indicators']:
            explanation_parts.append("Momentum indicators show: " + "; ".join(technical_analysis['momentum_indicators']))

        if technical_analysis['volatility_indicators']:
            explanation_parts.append("Volatility analysis: " + "; ".join(technical_analysis['volatility_indicators']))

        if technical_analysis['volume_indicators']:
            explanation_parts.append("Volume analysis: " + "; ".join(technical_analysis['volume_indicators']))

        # RAG contexts
        if similar_contexts:
            context_texts = [ctx['text'] for ctx in similar_contexts]
            explanation_parts.append("Relevant market contexts: " + "; ".join(context_texts))

        # Market context
        if market_context:
            if market_context.get('volatility', 0) > 0.02:
                explanation_parts.append("Market is experiencing elevated volatility, which may increase both opportunities and risks.")

            if market_context.get('news_events', 0) > 0:
                explanation_parts.append("Economic news events may be impacting market direction.")

        # Risk considerations
        explanation_parts.append("Risk management: Position size should be adjusted based on volatility and correlation with existing positions.")

        # Final recommendation
        if confidence > 0.7:
            explanation_parts.append("High confidence signal - consider normal position sizing.")
        elif confidence > 0.5:
            explanation_parts.append("Moderate confidence signal - consider reduced position sizing.")
        else:
            explanation_parts.append("Low confidence signal - consider waiting for stronger confirmation.")

        return " ".join(explanation_parts)

    def create_chart_annotation(self, symbol: str, trade_data: Dict,
                               features: Dict, timeframe: str = '1h') -> str:
        """Create annotated chart for trade explanation"""

        try:
            # Load recent price data
            data_file = Path('data/processed') / f"{symbol}_{timeframe}_processed.parquet"

            if not data_file.exists():
                return None

            df = pd.read_parquet(data_file)

            # Get recent data (last 100 periods)
            recent_data = df.tail(100).copy()

            if recent_data.empty:
                return None

            # Create chart
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

            # Price chart
            dates = recent_data.index
            ax1.plot(dates, recent_data['close'], label='Price', linewidth=2)

            # Add technical indicators
            if 'ema_21' in recent_data.columns:
                ax1.plot(dates, recent_data['ema_21'], label='EMA 21', alpha=0.7)

            if 'bb_upper' in recent_data.columns and 'bb_lower' in recent_data.columns:
                ax1.plot(dates, recent_data['bb_upper'], label='BB Upper', alpha=0.6, linestyle='--')
                ax1.plot(dates, recent_data['bb_lower'], label='BB Lower', alpha=0.6, linestyle='--')
                ax1.fill_between(dates, recent_data['bb_lower'], recent_data['bb_upper'], alpha=0.1)

            ax1.set_title(f'{symbol} - Technical Analysis')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Volume chart
            if 'volume' in recent_data.columns:
                ax2.bar(dates, recent_data['volume'], alpha=0.7, color='gray')
                ax2.set_ylabel('Volume')
                ax2.grid(True, alpha=0.3)

            # Add trade annotation
            trade_price = trade_data.get('price', recent_data['close'].iloc[-1])
            trade_time = datetime.now()

            # Add vertical line at trade time
            ax1.axvline(x=trade_time, color='red', linestyle='--', alpha=0.7, label='Trade Signal')

            # Add horizontal line at trade price
            ax1.axhline(y=trade_price, color='red', linestyle=':', alpha=0.7, label='Entry Price')

            # Add annotation text
            confidence = abs(trade_data.get('model_prediction', 0.5) - 0.5) * 2
            ax1.annotate(f'{trade_data.get("side", "UNKNOWN")} Signal\nConfidence: {confidence:.2f}',
                        xy=(trade_time, trade_price),
                        xytext=(10, 10),
                        textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8),
                        arrowprops=dict(arrowstyle='->', color='red'))

            plt.tight_layout()

            # Save chart
            chart_file = self.explanations_dir / f"trade_chart_{symbol}_{int(time.time())}.png"
            plt.savefig(chart_file, dpi=150, bbox_inches='tight')
            plt.close()

            logging.info(f"Created annotated chart: {chart_file}")
            return str(chart_file)

        except Exception as e:
            logging.error(f"Error creating chart: {e}")
            return None

    def send_to_telegram(self, explanation: str, chart_path: Optional[str] = None,
                        bot_token: str = None, chat_id: str = None) -> bool:
        """Send explanation to Telegram"""

        if not bot_token or not chat_id:
            logging.warning("Telegram credentials not provided")
            return False

        try:
            from telegram import Bot

            bot = Bot(token=bot_token)

            # Send text message
            bot.send_message(
                chat_id=chat_id,
                text=f"🤖 **TitanovaX Trade Explanation**\n\n{explanation}",
                parse_mode='Markdown'
            )

            # Send chart if available
            if chart_path and Path(chart_path).exists():
                with open(chart_path, 'rb') as photo:
                    bot.send_photo(
                        chat_id=chat_id,
                        photo=photo,
                        caption="📊 Technical analysis chart"
                    )

            logging.info("Explanation sent to Telegram successfully")
            return True

        except Exception as e:
            logging.error(f"Error sending to Telegram: {e}")
            return False

    def save_explanation(self, trade_data: Dict, explanation: str,
                        chart_path: Optional[str] = None) -> str:
        """Save explanation to file"""

        explanation_id = hashlib.md5(f"{trade_data.get('symbol', 'unknown')}_{datetime.now().isoformat()}".encode()).hexdigest()[:12]

        explanation_record = {
            'id': explanation_id,
            'timestamp': datetime.now().isoformat(),
            'trade_data': trade_data,
            'explanation': explanation,
            'chart_path': chart_path
        }

        # Save individual explanation
        explanation_file = self.explanations_dir / f"explanation_{explanation_id}.json"
        with open(explanation_file, 'w') as f:
            json.dump(explanation_record, f, indent=2, default=str)

        # Append to explanations log
        log_file = self.explanations_dir / 'explanations_log.json'
        try:
            if log_file.exists():
                with open(log_file, 'r') as f:
                    explanations_log = json.load(f)
            else:
                explanations_log = []

            explanations_log.append(explanation_record)

            # Keep only last 1000 explanations
            if len(explanations_log) > 1000:
                explanations_log = explanations_log[-1000:]

            with open(log_file, 'w') as f:
                json.dump(explanations_log, f, indent=2, default=str)

        except Exception as e:
            logging.error(f"Error updating explanations log: {e}")

        return explanation_id

    async def explain_trade_async(self, trade_data: Dict, features: Dict,
                                 model_prediction: float, market_context: Dict = None,
                                 telegram_config: Dict = None) -> Dict:
        """Async version of trade explanation"""

        # Generate explanation
        explanation = self.generate_explanation(trade_data, model_prediction, features, market_context)

        # Create chart
        chart_path = None
        if trade_data.get('symbol'):
            chart_path = self.create_chart_annotation(
                trade_data['symbol'], trade_data, features
            )

        # Save explanation
        explanation_id = self.save_explanation(trade_data, explanation, chart_path)

        # Send to Telegram if configured
        telegram_sent = False
        if telegram_config:
            telegram_sent = self.send_to_telegram(
                explanation, chart_path,
                telegram_config.get('bot_token'),
                telegram_config.get('chat_id')
            )

        return {
            'explanation_id': explanation_id,
            'explanation': explanation,
            'chart_path': chart_path,
            'telegram_sent': telegram_sent,
            'timestamp': datetime.now().isoformat()
        }

    def explain_trade(self, trade_data: Dict, features: Dict, model_prediction: float,
                     market_context: Dict = None, telegram_config: Dict = None) -> Dict:
        """Synchronous wrapper for trade explanation"""

        try:
            return asyncio.run(self.explain_trade_async(
                trade_data, features, model_prediction, market_context, telegram_config
            ))
        except Exception as e:
            logging.error(f"Error in trade explanation: {e}")
            return {
                'error': str(e),
                'explanation': 'Unable to generate explanation due to technical error'
            }

def main():
    """Main function for testing explainability system"""

    print("=== TitanovaX Explainability System ===")

    explainer = TradeExplainer()

    # Test data
    test_trade = {
        'symbol': 'EURUSD',
        'side': 'BUY',
        'price': 1.0850,
        'volume': 0.01,
        'timestamp': datetime.now().isoformat()
    }

    test_features = {
        'close': 1.0850,
        'ema_8': 1.0845,
        'ema_21': 1.0830,
        'rsi_14': 65.5,
        'macd': 0.0012,
        'macd_signal': 0.0008,
        'bb_upper': 1.0870,
        'bb_middle': 1.0840,
        'bb_lower': 1.0810,
        'volume': 1500000,
        'volume_sma_10': 1200000,
        'hour': 14,
        'day_of_week': 2
    }

    test_prediction = 0.72

    # Generate explanation
    print("Generating explanation...")
    result = explainer.explain_trade(
        test_trade, test_features, test_prediction
    )

    print("\n=== Generated Explanation ===")
    print(result['explanation'])

    if result.get('chart_path'):
        print(f"Chart saved to: {result['chart_path']}")

    print(f"Explanation ID: {result['explanation_id']}")

if __name__ == "__main__":
    main()
