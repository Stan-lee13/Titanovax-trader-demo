#!/usr/bin/env python3
"""
Walk-Forward Validation Framework for TitanovaX Trading System
Rigorous validation of ML models using time-series cross-validation
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
import hashlib
import pickle

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/logs/walkforward_validation.log'),
        logging.StreamHandler()
    ]
)

class WalkForwardValidator:
    def __init__(self, models_dir='models', validation_dir='data/validation'):
        self.models_dir = Path(models_dir)
        self.validation_dir = Path(validation_dir)
        self.validation_dir.mkdir(parents=True, exist_ok=True)

        # Production thresholds
        self.thresholds = {
            'min_auc': 0.55,
            'min_accuracy': 0.52,
            'min_precision': 0.53,
            'min_f1': 0.52,
            'max_auc_std': 0.08,
            'min_sharpe_ratio': 0.5,
            'max_drawdown': 0.15,
            'min_profit_factor': 1.1
        }

    def load_model_metadata(self, model_path: str) -> Dict:
        """Load model metadata"""

        metadata_file = Path(model_path).with_suffix('.metadata.json')

        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

        with open(metadata_file, 'r') as f:
            return json.load(f)

    def calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""

        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0

    def calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""

        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return abs(drawdown.min())

    def calculate_profit_factor(self, returns: pd.Series) -> float:
        """Calculate profit factor (gross profit / gross loss)"""

        profits = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())

        return profits / losses if losses > 0 else float('inf')

    def simulate_trading(self, predictions: pd.Series, actuals: pd.Series,
                        initial_capital: float = 10000, position_size: float = 0.01) -> Dict:
        """Simulate trading based on predictions"""

        # Create trading signals
        signals = predictions  # 1 for long, 0 for no position

        # Calculate returns (simplified)
        price_changes = actuals.shift(-1)  # Next period return
        trade_returns = signals * price_changes * position_size

        # Account for transaction costs (simplified)
        transaction_cost = 0.0002  # 2 pips spread + commission
        trade_returns = trade_returns - (abs(signals.diff()) * transaction_cost)

        # Calculate cumulative returns
        cumulative_returns = (1 + trade_returns).cumprod()
        total_return = cumulative_returns.iloc[-1] - 1

        # Calculate metrics
        sharpe_ratio = self.calculate_sharpe_ratio(trade_returns)
        max_drawdown = self.calculate_max_drawdown(trade_returns)
        profit_factor = self.calculate_profit_factor(trade_returns)

        # Win rate
        winning_trades = (trade_returns > 0).sum()
        total_trades = (trade_returns != 0).sum()
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'profit_factor': profit_factor,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': total_trades - winning_trades,
            'cumulative_returns': cumulative_returns
        }

    def perform_walk_forward_validation(self, symbol: str, timeframe: str = '1m',
                                      target_horizon: str = '5m', n_splits: int = 10) -> Dict:
        """Perform comprehensive walk-forward validation"""

        logging.info(f"Starting walk-forward validation for {symbol} {timeframe} -> {target_horizon}")

        try:
            # Load processed data
            data_file = Path('data/processed') / f"{symbol}_{timeframe}_processed.parquet"
            if not data_file.exists():
                raise FileNotFoundError(f"Processed data not found: {data_file}")

            df = pd.read_parquet(data_file)

            # Prepare features and target
            feature_columns = [
                'open', 'high', 'low', 'close', 'volume',
                'sma_5', 'sma_10', 'sma_20', 'sma_50',
                'ema_8', 'ema_21', 'ema_50',
                'rsi_14', 'macd', 'macd_signal', 'macd_hist',
                'bb_upper', 'bb_middle', 'bb_lower',
                'atr_14', 'momentum_5', 'momentum_10', 'momentum_20',
                'volatility_10', 'volatility_20',
                'hour', 'day_of_week', 'is_weekend',
                'is_asian_session', 'is_european_session', 'is_us_session',
                'trend_strength', 'volatility_regime', 'momentum_regime'
            ]

            target_column = f'label_up_{target_horizon.replace("m", "m")}' if 'm' in target_horizon else f'label_up_{target_horizon}'

            feature_cols = [col for col in feature_columns if col in df.columns]
            if target_column not in df.columns:
                raise ValueError(f"Target column {target_column} not found")

            X = df[feature_cols].dropna()
            y = df[target_column].loc[X.index]

            # Time series split
            tscv = TimeSeriesSplit(n_splits=n_splits)
            window_results = []

            for i, (train_idx, test_idx) in enumerate(tscv.split(X)):
                logging.info(f"Processing window {i+1}/{n_splits}")

                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                # Train model
                model = xgb.XGBClassifier(
                    max_depth=6,
                    learning_rate=0.1,
                    n_estimators=200,
                    random_state=42,
                    eval_metric='logloss'
                )

                model.fit(X_train, y_train)

                # Make predictions
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                y_pred = model.predict(X_test)

                # Calculate metrics
                auc = roc_auc_score(y_test, y_pred_proba)
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)

                # Simulate trading
                predictions_series = pd.Series(y_pred, index=X_test.index)
                actuals_series = y_test

                trading_results = self.simulate_trading(predictions_series, actuals_series)

                window_result = {
                    'window': i + 1,
                    'train_size': len(X_train),
                    'test_size': len(X_test),
                    'train_start': X_train.index.min().isoformat(),
                    'train_end': X_train.index.max().isoformat(),
                    'test_start': X_test.index.min().isoformat(),
                    'test_end': X_test.index.max().isoformat(),
                    'metrics': {
                        'auc': auc,
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1
                    },
                    'trading': trading_results
                }

                window_results.append(window_result)

            # Calculate aggregate statistics
            auc_scores = [r['metrics']['auc'] for r in window_results]
            accuracy_scores = [r['metrics']['accuracy'] for r in window_results]
            sharpe_scores = [r['trading']['sharpe_ratio'] for r in window_results]
            drawdown_scores = [r['trading']['max_drawdown'] for r in window_results]

            validation_results = {
                'symbol': symbol,
                'timeframe': timeframe,
                'target_horizon': target_horizon,
                'total_windows': len(window_results),
                'overall_metrics': {
                    'auc_mean': np.mean(auc_scores),
                    'auc_std': np.std(auc_scores),
                    'auc_min': np.min(auc_scores),
                    'auc_max': np.max(auc_scores),
                    'accuracy_mean': np.mean(accuracy_scores),
                    'accuracy_std': np.std(accuracy_scores),
                    'sharpe_mean': np.mean(sharpe_scores),
                    'sharpe_std': np.std(sharpe_scores),
                    'drawdown_mean': np.mean(drawdown_scores),
                    'drawdown_max': np.max(drawdown_scores)
                },
                'validation_passed': (
                    np.mean(auc_scores) >= self.thresholds['min_auc'] and
                    np.std(auc_scores) <= self.thresholds['max_auc_std'] and
                    np.mean(sharpe_scores) >= self.thresholds['min_sharpe_ratio'] and
                    np.max(drawdown_scores) <= self.thresholds['max_drawdown']
                ),
                'window_results': window_results,
                'thresholds_used': self.thresholds
            }

            logging.info(f"Walk-forward validation completed for {symbol}")
            logging.info(f"AUC: {validation_results['overall_metrics']['auc_mean']:.4f} ± {validation_results['overall_metrics']['auc_std']:.4f}")
            logging.info(f"Validation passed: {validation_results['validation_passed']}")

            return validation_results

        except Exception as e:
            logging.error(f"Error in walk-forward validation for {symbol}: {e}")
            return {
                'error': str(e),
                'validation_passed': False
            }

    def validate_model(self, model_path: str) -> Dict:
        """Validate a specific model using walk-forward analysis"""

        try:
            metadata = self.load_model_metadata(model_path)
            symbol = metadata['symbol']
            timeframe = metadata['timeframe']
            horizon = metadata['target_horizon']

            logging.info(f"Validating model: {metadata['model_id']}")

            # Perform walk-forward validation
            validation_results = self.perform_walk_forward_validation(
                symbol, timeframe, horizon
            )

            # Create validation report
            report = {
                'model_id': metadata['model_id'],
                'model_info': metadata,
                'validation_results': validation_results,
                'production_ready': validation_results.get('validation_passed', False),
                'validated_at': datetime.now().isoformat(),
                'validation_version': '2.0.0'
            }

            # Save validation report
            validation_file = self.validation_dir / f"{metadata['model_id']}_validation.json"
            with open(validation_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)

            logging.info(f"Validation report saved to {validation_file}")

            return report

        except Exception as e:
            logging.error(f"Error validating model {model_path}: {e}")
            return {
                'error': str(e),
                'production_ready': False
            }

    def validate_all_models(self) -> Dict:
        """Validate all available models"""

        # Find all model metadata files
        metadata_files = list(self.models_dir.glob('*_metadata.json'))

        logging.info(f"Found {len(metadata_files)} models to validate")

        validation_results = {}

        for metadata_file in metadata_files:
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)

                model_id = metadata['model_id']

                logging.info(f"Validating model: {model_id}")

                # Validate model
                result = self.validate_model(str(metadata_file))

                validation_results[model_id] = result

            except Exception as e:
                logging.error(f"Error validating {metadata_file}: {e}")
                validation_results[metadata_file.stem] = {
                    'error': str(e),
                    'production_ready': False
                }

        # Create summary
        production_ready = [r for r in validation_results.values() if r.get('production_ready', False)]
        failed_validation = [r for r in validation_results.values() if not r.get('production_ready', True)]

        summary = {
            'total_models': len(validation_results),
            'production_ready': len(production_ready),
            'failed_validation': len(failed_validation),
            'production_ready_models': [r['model_id'] for r in production_ready],
            'validation_results': validation_results,
            'summary_by_symbol': self.summarize_by_symbol(validation_results)
        }

        # Save summary
        summary_file = self.validation_dir / 'validation_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        logging.info(f"Validation complete: {len(production_ready)}/{len(validation_results)} models ready for production")

        return summary

    def summarize_by_symbol(self, validation_results: Dict) -> Dict:
        """Summarize validation results by symbol"""

        symbol_summary = {}

        for model_id, result in validation_results.items():
            if 'error' in result:
                continue

            symbol = result['model_info']['symbol']

            if symbol not in symbol_summary:
                symbol_summary[symbol] = {
                    'total_models': 0,
                    'production_ready': 0,
                    'best_model': None,
                    'best_sharpe': 0
                }

            symbol_summary[symbol]['total_models'] += 1

            if result.get('production_ready', False):
                symbol_summary[symbol]['production_ready'] += 1

                # Track best model
                avg_sharpe = result['validation_results']['overall_metrics']['sharpe_mean']
                if avg_sharpe > symbol_summary[symbol]['best_sharpe']:
                    symbol_summary[symbol]['best_sharpe'] = avg_sharpe
                    symbol_summary[symbol]['best_model'] = model_id

        return symbol_summary

    def generate_validation_report(self, output_file: str = None) -> str:
        """Generate comprehensive validation report"""

        # Load validation summary
        summary_file = self.validation_dir / 'validation_summary.json'
        if not summary_file.exists():
            return "No validation results found"

        with open(summary_file, 'r') as f:
            summary = json.load(f)

        # Generate report
        report = f"""
# TitanovaX Model Validation Report
Generated: {datetime.now().isoformat()}

## Summary
- Total Models Validated: {summary['total_models']}
- Production Ready: {summary['production_ready']}
- Failed Validation: {summary['failed_validation']}

## Production Ready Models
"""

        for model_id in summary['production_ready_models']:
            result = summary['validation_results'][model_id]
            metrics = result['validation_results']['overall_metrics']

            report += f"""
### {model_id}
- Symbol: {result['model_info']['symbol']}
- Timeframe: {result['model_info']['timeframe']}
- Horizon: {result['model_info']['target_horizon']}
- AUC: {metrics['auc_mean']:.4f} ± {metrics['auc_std']:.4f}
- Sharpe Ratio: {metrics['sharpe_mean']:.4f}
- Max Drawdown: {metrics['drawdown_max']:.4f}
- Production Ready: ✅
"""

        # Symbol breakdown
        report += "\n## Performance by Symbol\n"
        for symbol, info in summary['summary_by_symbol'].items():
            report += f"- {symbol}: {info['production_ready']}/{info['total_models']} models ready\n"

        # Save report
        if output_file is None:
            output_file = self.validation_dir / 'validation_report.md'

        with open(output_file, 'w') as f:
            f.write(report)

        logging.info(f"Validation report saved to {output_file}")
        return report

def main():
    """Main function for model validation"""

    print("=== TitanovaX Walk-Forward Validation ===")

    validator = WalkForwardValidator()

    # Validate all models
    print("Starting validation of all models...")
    results = validator.validate_all_models()

    print("\n=== Validation Summary ===")
    print(f"Total models: {results['total_models']}")
    print(f"Production ready: {results['production_ready']}")
    print(f"Failed validation: {results['failed_validation']}")

    print("\nProduction-ready models:")
    for model_id in results['production_ready_models']:
        print(f"  ✅ {model_id}")

    # Generate report
    print("\nGenerating validation report...")
    report = validator.generate_validation_report()

    print(f"\nReport saved to: {validator.validation_dir}/validation_report.md")
    print("Validation complete!")

if __name__ == "__main__":
    main()
