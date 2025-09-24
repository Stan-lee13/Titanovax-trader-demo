"""
Comprehensive ML Training Pipeline for TitanovaX Trading System
Implements XGBoost, Transformer, and ensemble training with ONNX export
"""

import pandas as pd
import numpy as np
import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import hashlib

# ML Libraries
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import optuna
from optuna.samplers import TPESampler

# Deep Learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ONNX
import onnx
import onnxmltools
from onnxmltools.convert import convert_xgboost, convert_sklearn
from onnxmltools.convert.common.data_types import FloatTensorType

# Custom imports
from ml_brain.features.feature_builder import build_features
from ml_brain.features.technical_indicators import TechnicalIndicators

@dataclass
class TrainingConfig:
    """Configuration for model training"""
    symbol: str
    timeframe: str = "1m"
    target_horizon: str = "5m"
    prediction_type: str = "classification"
    test_size: float = 0.2
    val_size: float = 0.15
    n_trials: int = 100
    models_dir: str = "models"
    data_dir: str = "data"
    
class TransformerModel(nn.Module):
    """Transformer model for time series prediction"""
    
    def __init__(self, input_dim: int, d_model: int = 128, nhead: int = 8, 
                 num_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = self._generate_positional_encoding(1000, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 3)  # up, down, sideways
        )
        
    def _generate_positional_encoding(self, max_len: int, d_model: int):
        """Generate positional encoding"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        x = self.input_projection(x)
        
        # Add positional encoding
        x = x + self.positional_encoding[:, :seq_len, :].to(x.device)
        
        # Transformer encoding
        x = self.transformer(x)
        
        # Use last time step for classification
        x = x[:, -1, :]
        
        # Classification
        return self.classifier(x)

class MLTrainer:
    """Comprehensive ML training pipeline"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        
        # Create directories
        Path(config.models_dir).mkdir(parents=True, exist_ok=True)
        Path("logs").mkdir(parents=True, exist_ok=True)
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler(f'logs/training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(formatter)
        logger.addHandler(console)
        
        return logger
        
    def load_and_prepare_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load and prepare training data"""
        self.logger.info(f"Loading data for {self.config.symbol} {self.config.timeframe}")
        
        # Load raw market data
        data_path = Path(self.config.data_dir) / "raw" / f"{self.config.symbol}_{self.config.timeframe}.parquet"
        
        if not data_path.exists():
            # Generate synthetic data for development
            self.logger.warning(f"Data file not found: {data_path}. Generating synthetic data.")
            return self._generate_synthetic_data()
        
        df = pd.read_parquet(data_path)
        
        # Build features
        df_features = build_features(df, self.config.symbol, self.config.timeframe)
        
        # Create target variable
        target_col = f"label_{self.config.target_horizon}"
        if target_col not in df_features.columns:
            df_features = self._create_target_variable(df_features)
        
        # Select feature columns
        feature_cols = [col for col in df_features.columns 
                       if col not in ['open', 'high', 'low', 'close', 'volume', 
                                    'timestamp', 'label_5m', 'label_30m', 'label_1h', 'label_24h']]
        
        self.feature_columns = feature_cols
        
        X = df_features[feature_cols]
        y = df_features[target_col]
        
        # Remove NaN values
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
        
        self.logger.info(f"Prepared {len(X)} samples with {len(feature_cols)} features")
        
        return X, y
        
    def _generate_synthetic_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Generate synthetic data for development/testing"""
        self.logger.info("Generating synthetic training data")
        
        np.random.seed(42)
        n_samples = 10000
        
        # Generate base price data
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), 
                             periods=n_samples, freq='1min')
        
        # Random walk price generation
        returns = np.random.normal(0.0001, 0.001, n_samples)
        prices = 1.0 * np.exp(np.cumsum(returns))
        
        # Generate OHLCV data
        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices * (1 + np.random.normal(0, 0.0001, n_samples)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.0005, n_samples))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.0005, n_samples))),
            'close': prices,
            'volume': np.random.randint(1000, 10000, n_samples)
        })
        
        # Ensure OHLC consistency
        df['high'] = df[['open', 'high', 'low', 'close']].max(axis=1)
        df['low'] = df[['open', 'high', 'low', 'close']].min(axis=1)
        
        df.set_index('timestamp', inplace=True)
        
        # Build features
        df_features = build_features(df, self.config.symbol, self.config.timeframe)
        
        # Create target variable
        df_features = self._create_target_variable(df_features)
        
        # Select features
        feature_cols = [col for col in df_features.columns 
                       if col not in ['open', 'high', 'low', 'close', 'volume', 'timestamp']]
        
        self.feature_columns = feature_cols
        
        X = df_features[feature_cols]
        y = df_features[f"label_{self.config.target_horizon}"]
        
        # Remove NaN values
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
        
        return X, y
        
    def _create_target_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create target variable for prediction"""
        
        # Calculate future returns
        if self.config.target_horizon == '5m':
            periods = 5
        elif self.config.target_horizon == '30m':
            periods = 30
        elif self.config.target_horizon == '1h':
            periods = 60
        elif self.config.target_horizon == '24h':
            periods = 1440
        else:
            periods = 5
            
        future_returns = df['close'].pct_change(periods=periods).shift(-periods)
        
        # Create classification target
        df[f'label_{self.config.target_horizon}'] = np.where(future_returns > 0.001, 1, 0)
        
        return df
        
    def train_xgboost_with_optuna(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train XGBoost model with Optuna hyperparameter optimization"""
        
        self.logger.info("Training XGBoost model with Optuna optimization")
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, shuffle=False, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, shuffle=False, random_state=42
        )
        
        def objective(trial):
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'random_state': 42
            }
            
            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            
            y_pred = model.predict(X_val)
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            
            auc = roc_auc_score(y_val, y_pred_proba)
            f1 = f1_score(y_val, y_pred)
            
            return 0.7 * auc + 0.3 * f1
        
        # Run Optuna optimization
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
        study.optimize(objective, n_trials=self.config.n_trials, n_jobs=-1)
        
        # Train final model with best parameters
        best_params = study.best_params
        best_params['random_state'] = 42
        
        final_model = xgb.XGBClassifier(**best_params)
        final_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
        # Evaluate on test set
        y_test_pred = final_model.predict(X_test)
        y_test_proba = final_model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_test_pred),
            'precision': precision_score(y_test, y_test_pred),
            'recall': recall_score(y_test, y_test_pred),
            'f1': f1_score(y_test, y_test_pred),
            'auc': roc_auc_score(y_test, y_test_proba),
            'confusion_matrix': confusion_matrix(y_test, y_test_pred).tolist()
        }
        
        self.logger.info(f"XGBoost training completed. AUC: {metrics['auc']:.4f}")
        
        return {
            'model': final_model,
            'metrics': metrics,
            'best_params': best_params,
            'study': study,
            'feature_columns': self.feature_columns
        }
        
    def train_transformer_model(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train Transformer model for time series prediction"""
        
        self.logger.info("Training Transformer model")
        
        # Prepare sequence data
        sequence_length = 60  # Use 60 time steps
        
        # Create sequences
        sequences = []
        targets = []
        
        for i in range(sequence_length, len(X)):
            sequences.append(X.iloc[i-sequence_length:i].values)
            targets.append(y.iloc[i])
        
        sequences = np.array(sequences)
        targets = np.array(targets)
        
        # Split data
        split_idx = int(0.8 * len(sequences))
        X_train, X_test = sequences[:split_idx], sequences[split_idx:]
        y_train, y_test = targets[:split_idx], targets[split_idx:]
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.LongTensor(y_test)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        # Initialize model
        model = TransformerModel(input_dim=X.shape[1])
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Training loop
        model.train()
        for epoch in range(50):  # 50 epochs
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            _, predicted = torch.max(test_outputs.data, 1)
            
            accuracy = (predicted == y_test_tensor).float().mean().item()
            
        self.logger.info(f"Transformer training completed. Accuracy: {accuracy:.4f}")
        
        return {
            'model': model,
            'metrics': {'accuracy': accuracy},
            'sequence_length': sequence_length
        }
        
    def train_ensemble_model(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train ensemble model combining multiple algorithms"""
        
        self.logger.info("Training Ensemble model")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Individual models
        models = [
            ('xgb', xgb.XGBClassifier(n_estimators=100, random_state=42)),
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('lr', LogisticRegression(random_state=42, max_iter=1000))
        ]
        
        # Ensemble model
        ensemble = VotingClassifier(estimators=models, voting='soft')
        ensemble.fit(X_train_scaled, y_train)
        
        # Evaluation
        y_pred = ensemble.predict(X_test_scaled)
        y_pred_proba = ensemble.predict_proba(X_test_scaled)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        self.logger.info(f"Ensemble training completed. AUC: {metrics['auc']:.4f}")
        
        return {
            'model': ensemble,
            'scaler': scaler,
            'metrics': metrics
        }
        
    def export_to_onnx(self, model_dict: Dict[str, Any], model_type: str) -> str:
        """Export trained model to ONNX format"""
        
        self.logger.info(f"Exporting {model_type} model to ONNX")
        
        model_id = f"{self.config.symbol}_{self.config.target_horizon}_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        model_path = Path(self.config.models_dir) / f"{model_id}.onnx"
        metadata_path = Path(self.config.models_dir) / f"{model_id}.metadata.json"
        
        if model_type == 'xgboost':
            model = model_dict['model']
            
            # Convert to ONNX
            initial_type = [('float_input', FloatTensorType([None, len(self.feature_columns)]))]
            onnx_model = convert_xgboost(model, initial_types=initial_type)
            
            # Save ONNX model
            onnxmltools.utils.save_model(onnx_model, str(model_path))
            
        elif model_type == 'ensemble':
            model = model_dict['model']
            
            # Convert to ONNX
            initial_type = [('float_input', FloatTensorType([None, len(self.feature_columns)]))]
            onnx_model = convert_sklearn(model, initial_types=initial_type)
            
            # Save ONNX model
            onnxmltools.utils.save_model(onnx_model, str(model_path))
            
        elif model_type == 'transformer':
            # PyTorch model - save as pickle for now
            # ONNX export for PyTorch models requires more complex handling
            model = model_dict['model']
            torch.save(model.state_dict(), model_path.with_suffix('.pth'))
            self.logger.warning(f"Transformer model saved as PyTorch state dict, not ONNX")
            
        # Save metadata
        metadata = {
            'model_id': model_id,
            'symbol': self.config.symbol,
            'timeframe': self.config.timeframe,
            'target_horizon': self.config.target_horizon,
            'model_type': model_type,
            'feature_columns': self.feature_columns,
            'metrics': model_dict['metrics'],
            'created_at': datetime.now().isoformat(),
            'training_config': self.config.__dict__
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        self.logger.info(f"Model exported: {model_path}")
        
        return model_id
        
    def train_all_models(self) -> Dict[str, str]:
        """Train all model types and export to ONNX"""
        
        self.logger.info("Starting comprehensive model training pipeline")
        
        # Load and prepare data
        X, y = self.load_and_prepare_data()
        
        trained_models = {}
        
        # Train XGBoost model
        try:
            xgb_results = self.train_xgboost_with_optuna(X, y)
            xgb_model_id = self.export_to_onnx(xgb_results, 'xgboost')
            trained_models['xgboost'] = xgb_model_id
        except Exception as e:
            self.logger.error(f"XGBoost training failed: {e}")
            
        # Train Ensemble model
        try:
            ensemble_results = self.train_ensemble_model(X, y)
            ensemble_model_id = self.export_to_onnx(ensemble_results, 'ensemble')
            trained_models['ensemble'] = ensemble_model_id
        except Exception as e:
            self.logger.error(f"Ensemble training failed: {e}")
            
        # Train Transformer model
        try:
            transformer_results = self.train_transformer_model(X, y)
            transformer_model_id = self.export_to_onnx(transformer_results, 'transformer')
            trained_models['transformer'] = transformer_model_id
        except Exception as e:
            self.logger.error(f"Transformer training failed: {e}")
            
        self.logger.info(f"Training pipeline completed. Models: {trained_models}")
        
        return trained_models
        
    def generate_training_report(self, model_results: Dict[str, str]) -> str:
        """Generate comprehensive training report"""
        
        report_path = Path(self.config.models_dir) / f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report = {
            'training_config': self.config.__dict__,
            'models_trained': model_results,
            'timestamp': datetime.now().isoformat(),
            'feature_count': len(self.feature_columns),
            'data_summary': {
                'symbol': self.config.symbol,
                'timeframe': self.config.timeframe,
                'target_horizon': self.config.target_horizon
            }
        }
        
        # Load individual model metrics
        for model_type, model_id in model_results.items():
            metadata_path = Path(self.config.models_dir) / f"{model_id}.metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    report[f'{model_type}_metrics'] = metadata.get('metrics', {})
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        self.logger.info(f"Training report generated: {report_path}")
        
        return str(report_path)

def main():
    """Main training function"""
    
    # Example usage
    config = TrainingConfig(
        symbol="EURUSD",
        timeframe="1m",
        target_horizon="5m",
        prediction_type="classification",
        n_trials=50  # Reduced for faster development
    )
    
    trainer = MLTrainer(config)
    
    # Train all models
    model_results = trainer.train_all_models()
    
    # Generate training report
    report_path = trainer.generate_training_report(model_results)
    
    print(f"Training completed! Models: {model_results}")
    print(f"Report: {report_path}")

if __name__ == "__main__":
    main()