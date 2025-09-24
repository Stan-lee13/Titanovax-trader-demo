#!/usr/bin/env python3
"""
XGBoost Training Pipeline for TitanovaX Trading System
Trains XGBoost models on engineered features and exports to ONNX
"""

import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import optuna
from optuna.samplers import TPESampler
import json
import pickle
import onnx
import onnxmltools
from onnxmltools.convert import convert_xgboost
from onnxmltools.utils import save_model
import logging
from datetime import datetime
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/logs/xgboost_training.log'),
        logging.StreamHandler()
    ]
)

class XGBoostTrainer:
    def __init__(self, models_dir='models', config_dir='config'):
        self.models_dir = Path(models_dir)
        self.config_dir = Path(config_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.config_dir.mkdir(parents=True, exist_ok=True)

        self.feature_columns = [
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

    def load_processed_data(self, symbol: str, timeframe: str = '1m') -> pd.DataFrame:
        """Load processed data for training"""

        data_file = Path('data/processed') / f"{symbol}_{timeframe}_processed.parquet"

        if not data_file.exists():
            raise FileNotFoundError(f"Processed data not found: {data_file}")

        df = pd.read_parquet(data_file)

        # Remove rows with NaN values in feature columns
        feature_cols = [col for col in self.feature_columns if col in df.columns]
        df = df.dropna(subset=feature_cols)

        logging.info(f"Loaded {len(df)} rows of processed data for {symbol} {timeframe}")
        return df

    def prepare_training_data(self, df: pd.DataFrame, target_horizon: str = '5m',
                           prediction_type: str = 'classification') -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for training"""

        feature_cols = [col for col in self.feature_columns if col in df.columns]

        if prediction_type == 'classification':
            if target_horizon == '5m':
                target_col = 'label_up_5m'
            elif target_horizon == '30m':
                target_col = 'label_up_30m'
            elif target_horizon == '1h':
                target_col = 'label_up_1h'
            else:
                target_col = 'label_up_24h'

            if target_col not in df.columns:
                raise ValueError(f"Target column {target_col} not found")

            X = df[feature_cols]
            y = df[target_col]

        elif prediction_type == 'regression':
            target_col = f'future_return_{target_horizon}'

            if target_col not in df.columns:
                raise ValueError(f"Target column {target_col} not found")

            X = df[feature_cols]
            y = df[target_col]

        else:
            raise ValueError(f"Unknown prediction type: {prediction_type}")

        logging.info(f"Prepared training data: {len(X)} samples, {len(feature_cols)} features")
        return X, y

    def objective(self, trial: optuna.Trial, X_train: pd.DataFrame, y_train: pd.Series,
                  X_val: pd.DataFrame, y_val: pd.Series) -> float:
        """Optuna objective function for hyperparameter optimization"""

        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10)
        }

        model = xgb.XGBClassifier(**params, random_state=42, use_label_encoder=False, eval_metric='logloss')
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        # Make predictions on validation set
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        y_pred = model.predict(X_val)

        # Calculate metrics
        auc = roc_auc_score(y_val, y_pred_proba)
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)

        # Combine metrics (you can adjust weights)
        combined_score = (auc * 0.4 + accuracy * 0.2 + f1 * 0.2 + precision * 0.1 + recall * 0.1)

        return combined_score

    def train_with_optuna(self, X: pd.DataFrame, y: pd.Series, n_trials: int = 50) -> Dict:
        """Train model with Optuna hyperparameter optimization"""

        logging.info(f"Starting Optuna optimization with {n_trials} trials...")

        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=5)

        # Get the last split for final evaluation
        splits = list(tscv.split(X))
        train_idx, val_idx = splits[-1]

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Create Optuna study
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
        study.optimize(lambda trial: self.objective(trial, X_train, y_train, X_val, y_val), n_trials=n_trials)

        logging.info(f"Best trial: {study.best_trial.value}")
        logging.info(f"Best params: {study.best_params}")

        return {
            'best_params': study.best_params,
            'best_score': study.best_trial.value,
            'study': study
        }

    def train_model(self, symbol: str, timeframe: str = '1m', target_horizon: str = '5m',
                   prediction_type: str = 'classification', use_optuna: bool = True,
                   n_trials: int = 50) -> Dict:
        """Train XGBoost model for a symbol"""

        logging.info(f"Training XGBoost model for {symbol} {timeframe} -> {target_horizon}")

        try:
            # Load and prepare data
            df = self.load_processed_data(symbol, timeframe)
            X, y = self.prepare_training_data(df, target_horizon, prediction_type)

            # Split data (time series aware)
            split_point = int(len(X) * 0.8)
            X_train, X_test = X[:split_point], X[split_point:]
            y_train, y_test = y[:split_point], y[split_point:]

            # Hyperparameter optimization
            if use_optuna:
                optuna_results = self.train_with_optuna(X_train, y_train, n_trials)
                best_params = optuna_results['best_params']
            else:
                # Use default parameters
                best_params = {
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'n_estimators': 200,
                    'min_child_weight': 1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'gamma': 0,
                    'reg_alpha': 0,
                    'reg_lambda': 1
                }

            # Train final model
            if prediction_type == 'classification':
                model = xgb.XGBClassifier(**best_params, random_state=42, use_label_encoder=False, eval_metric='logloss')
            else:
                model = xgb.XGBRegressor(**best_params, random_state=42)

            model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

            # Evaluate model
            y_pred_proba = model.predict_proba(X_test)[:, 1] if prediction_type == 'classification' else None
            y_pred = model.predict(X_test)

            if prediction_type == 'classification':
                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred),
                    'recall': recall_score(y_test, y_pred),
                    'f1': f1_score(y_test, y_pred),
                    'auc': roc_auc_score(y_test, y_pred_proba)
                }
            else:
                from sklearn.metrics import mean_squared_error, r2_score
                metrics = {
                    'mse': mean_squared_error(y_test, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                    'r2': r2_score(y_test, y_pred)
                }

            # Create model metadata
            model_id = hashlib.md5(f"{symbol}_{timeframe}_{target_horizon}_{datetime.now().isoformat()}".encode()).hexdigest()[:16]

            model_info = {
                'model_id': model_id,
                'symbol': symbol,
                'timeframe': timeframe,
                'target_horizon': target_horizon,
                'prediction_type': prediction_type,
                'algorithm': 'XGBoost',
                'version': '1.0.0',
                'created_at': datetime.now().isoformat(),
                'training_data': {
                    'samples': len(X_train),
                    'features': len(X_train.columns),
                    'test_samples': len(X_test)
                },
                'hyperparameters': best_params,
                'metrics': metrics,
                'feature_columns': list(X_train.columns),
                'feature_importance': dict(zip(X_train.columns, model.feature_importances_))
            }

            # Save model files
            model_filename = f"xgb_{symbol}_{timeframe}_{target_horizon}"
            model_info['files'] = {}

            # Save XGBoost model
            model_path = self.models_dir / f"{model_filename}.json"
            model.save_model(str(model_path))
            model_info['files']['xgboost'] = str(model_path)

            # Save pickle model for ONNX conversion
            pickle_path = self.models_dir / f"{model_filename}.pkl"
            with open(pickle_path, 'wb') as f:
                pickle.dump(model, f)
            model_info['files']['pickle'] = str(pickle_path)

            # Convert to ONNX
            try:
                onnx_model = convert_xgboost(model, initial_types=[('input', X_train.values.dtype)])
                onnx_path = self.models_dir / f"{model_filename}.onnx"
                save_model(onnx_model, str(onnx_path))
                model_info['files']['onnx'] = str(onnx_path)
                logging.info(f"Successfully converted model to ONNX: {onnx_path}")
            except Exception as e:
                logging.warning(f"Failed to convert to ONNX: {e}")
                model_info['files']['onnx'] = None

            # Save metadata
            metadata_path = self.models_dir / f"{model_filename}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(model_info, f, indent=2, default=str)
            model_info['files']['metadata'] = str(metadata_path)

            # Save feature importance plot data
            importance_path = self.models_dir / f"{model_filename}_importance.json"
            importance_data = {
                'features': list(X_train.columns),
                'importance': model.feature_importances_.tolist(),
                'sorted_features': [x for _, x in sorted(zip(model.feature_importances_, X_train.columns), reverse=True)]
            }
            with open(importance_path, 'w') as f:
                json.dump(importance_data, f, indent=2)
            model_info['files']['importance'] = str(importance_path)

            logging.info(f"Model training completed for {symbol}")
            logging.info(f"Test metrics: {metrics}")

            return model_info

        except Exception as e:
            logging.error(f"Error training model for {symbol}: {e}")
            return {'error': str(e)}

    def walk_forward_validation(self, symbol: str, timeframe: str = '1m', target_horizon: str = '5m',
                              window_size_days: int = 90, test_window_days: int = 30) -> Dict:
        """Perform walk-forward validation"""

        logging.info(f"Starting walk-forward validation for {symbol}")

        try:
            df = self.load_processed_data(symbol, timeframe)
            X, y = self.prepare_training_data(df, target_horizon, 'classification')

            # Calculate split points
            total_samples = len(X)
            window_size = window_size_days * 24 * 60 // int(timeframe.replace('m', ''))  # Approximate samples per day
            test_size = test_window_days * 24 * 60 // int(timeframe.replace('m', ''))

            if window_size + test_size > total_samples:
                raise ValueError("Data not sufficient for walk-forward validation")

            # Perform walk-forward validation
            predictions = []
            actuals = []
            model_scores = []

            start_idx = 0
            while start_idx + window_size + test_size <= total_samples:
                train_end = start_idx + window_size
                test_end = train_end + test_size

                X_train = X.iloc[start_idx:train_end]
                y_train = y.iloc[start_idx:train_end]
                X_test = X.iloc[train_end:test_end]
                y_test = y.iloc[train_end:test_end]

                # Train model on current window
                model = xgb.XGBClassifier(max_depth=6, learning_rate=0.1, n_estimators=200, random_state=42)
                model.fit(X_train, y_train)

                # Make predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]

                predictions.extend(y_pred)
                actuals.extend(y_test)

                # Calculate metrics
                auc = roc_auc_score(y_test, y_pred_proba)
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)

                model_scores.append({
                    'window_start': start_idx,
                    'window_end': test_end,
                    'auc': auc,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                })

                logging.info(f"Window {start_idx}-{test_end}: AUC={auc:.4f}, Accuracy={accuracy:.4f}")

                start_idx += test_size  # Rolling window

            # Calculate overall metrics
            overall_auc = roc_auc_score(actuals, predictions)
            overall_accuracy = accuracy_score(actuals, predictions)
            overall_precision = precision_score(actuals, predictions)
            overall_recall = recall_score(actuals, predictions)
            overall_f1 = f1_score(actuals, predictions)

            # Calculate consistency metrics
            auc_scores = [s['auc'] for s in model_scores]
            auc_std = np.std(auc_scores)
            auc_min = min(auc_scores)
            auc_max = max(auc_scores)

            validation_results = {
                'symbol': symbol,
                'timeframe': timeframe,
                'target_horizon': target_horizon,
                'total_windows': len(model_scores),
                'overall_metrics': {
                    'auc': overall_auc,
                    'accuracy': overall_accuracy,
                    'precision': overall_precision,
                    'recall': overall_recall,
                    'f1': overall_f1
                },
                'consistency_metrics': {
                    'auc_mean': np.mean(auc_scores),
                    'auc_std': auc_std,
                    'auc_min': auc_min,
                    'auc_max': auc_max,
                    'auc_range': auc_max - auc_min
                },
                'window_scores': model_scores,
                'validation_passed': overall_auc > 0.55 and auc_std < 0.1  # Thresholds for production
            }

            logging.info(f"Walk-forward validation completed for {symbol}")
            logging.info(f"Overall AUC: {overall_auc:.4f}, Std: {auc_std:.4f}")

            return validation_results

        except Exception as e:
            logging.error(f"Error in walk-forward validation for {symbol}: {e}")
            return {'error': str(e)}

    def train_all_symbols(self, symbols: List[str] = None, timeframes: List[str] = None,
                         target_horizons: List[str] = None) -> Dict:
        """Train models for all symbols and configurations"""

        if symbols is None:
            symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD']

        if timeframes is None:
            timeframes = ['1m', '5m', '15m']

        if target_horizons is None:
            target_horizons = ['5m', '30m', '1h']

        all_results = {}
        total_models = len(symbols) * len(timeframes) * len(target_horizons)

        logging.info(f"Starting training of {total_models} models...")

        for symbol in symbols:
            symbol_results = {}

            for timeframe in timeframes:
                timeframe_results = {}

                for horizon in target_horizons:
                    try:
                        logging.info(f"Training {symbol} {timeframe} -> {horizon}")

                        # Train model
                        model_result = self.train_model(symbol, timeframe, horizon)

                        # Validate model
                        validation_result = self.walk_forward_validation(symbol, timeframe, horizon)

                        # Combine results
                        combined_result = {
                            'training': model_result,
                            'validation': validation_result,
                            'production_ready': (model_result.get('metrics', {}).get('auc', 0) > 0.55 and
                                              validation_result.get('validation_passed', False))
                        }

                        timeframe_results[horizon] = combined_result

                    except Exception as e:
                        logging.error(f"Error training {symbol} {timeframe} {horizon}: {e}")
                        timeframe_results[horizon] = {'error': str(e)}

                symbol_results[timeframe] = timeframe_results

            all_results[symbol] = symbol_results

        # Save summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_models_attempted': total_models,
            'results': all_results
        }

        summary_file = self.models_dir / 'training_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        logging.info(f"Training completed. Summary saved to {summary_file}")
        return summary

def main():
    """Main function for XGBoost training"""

    print("=== TitanovaX XGBoost Training Pipeline ===")

    trainer = XGBoostTrainer()

    # Train models for major symbols
    symbols = ['EURUSD', 'GBPUSD', 'XAUUSD']

    print("Starting model training...")
    results = trainer.train_all_symbols(
        symbols=symbols,
        timeframes=['1m', '5m'],
        target_horizons=['5m', '1h']
    )

    print("\n=== Training Summary ===")
    total_production_ready = 0
    total_models = 0

    for symbol, symbol_results in results['results'].items():
        print(f"\n{symbol}:")
        for timeframe, timeframe_results in symbol_results.items():
            for horizon, result in timeframe_results.items():
                total_models += 1
                if result.get('production_ready', False):
                    total_production_ready += 1
                    print(f"  ✅ {timeframe} -> {horizon}: Production Ready")
                else:
                    print(f"  ❌ {timeframe} -> {horizon}: Needs Improvement")

    print(f"\nOverall: {total_production_ready}/{total_models} models ready for production")
    print("Models saved to: models/")
    print("Training summary saved to: models/training_summary.json")

if __name__ == "__main__":
    main()
