"""
Comprehensive Integration Tests for TitanovaX Trading System
Tests end-to-end workflows and component interactions
"""

import pytest
import asyncio
import json
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import system components
from mt5_executor.mt5_trader import MT5Trader
from mt5_executor.risk_manager import RiskManager
from ml_brain.ensemble_decision_engine import EnsembleDecisionEngine
from ml_brain.adaptive_execution_gate import AdaptiveExecutionGate
from ml_brain.safety_risk_layer import SafetyRiskLayer
from ml_brain.training.trainer import MLTrainer, TrainingConfig
from ml_brain.onnx_server import InferenceServer
from orchestration.telegram_bot import TradingBot
from orchestration.rag_teacher import RAGTeacher


class TestIntegration:
    """Integration tests for the complete trading system"""
    
    @pytest.fixture
    def mock_mt5(self):
        """Mock MT5 connection"""
        with patch('MetaTrader5') as mock_mt5:
            # Mock account info
            mock_mt5.account_info.return_value = MagicMock(
                login=12345,
                balance=10000.0,
                equity=10000.0,
                profit=0.0,
                margin=0.0,
                margin_free=10000.0,
                margin_level=100.0
            )
            
            # Mock symbol info
            mock_mt5.symbol_info.return_value = MagicMock(
                point=0.00001,
                spread=10,
                bid=1.08500,
                ask=1.08510,
                volume_min=0.01,
                volume_max=100.0,
                volume_step=0.01
            )
            
            # Mock positions
            mock_mt5.positions_get.return_value = []
            
            # Mock orders
            mock_mt5.orders_get.return_value = []
            
            yield mock_mt5
    
    @pytest.fixture
    def test_config(self):
        """Test configuration"""
        return {
            'symbol': 'EURUSD',
            'timeframe': '1m',
            'lot_size': 0.1,
            'max_risk_percent': 2.0,
            'stop_loss_pips': 50,
            'take_profit_pips': 100,
            'max_positions': 5,
            'max_drawdown_percent': 10.0
        }
    
    @pytest.fixture
    def sample_market_data(self):
        """Generate sample market data for testing"""
        np.random.seed(42)
        n_samples = 1000
        
        dates = pd.date_range(start=datetime.now() - timedelta(days=1), 
                             periods=n_samples, freq='1min')
        
        # Generate realistic price data
        returns = np.random.normal(0.0001, 0.001, n_samples)
        prices = 1.085 * np.exp(np.cumsum(returns))
        
        data = pd.DataFrame({
            'timestamp': dates,
            'open': prices * (1 + np.random.normal(0, 0.0001, n_samples)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.0005, n_samples))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.0005, n_samples))),
            'close': prices,
            'volume': np.random.randint(1000, 10000, n_samples)
        })
        
        # Ensure OHLC consistency
        data['high'] = data[['open', 'high', 'low', 'close']].max(axis=1)
        data['low'] = data[['open', 'high', 'low', 'close']].min(axis=1)
        
        return data
    
    def test_end_to_end_trading_workflow(self, mock_mt5, test_config, sample_market_data):
        """Test complete trading workflow from signal generation to execution"""
        
        # Initialize components
        trader = MT5Trader(
            symbol=test_config['symbol'],
            lot_size=test_config['lot_size'],
            max_risk_percent=test_config['max_risk_percent']
        )
        
        risk_manager = RiskManager(
            max_risk_percent=test_config['max_risk_percent'],
            max_drawdown_percent=test_config['max_drawdown_percent'],
            max_positions=test_config['max_positions']
        )
        
        decision_engine = EnsembleDecisionEngine(
            model_dir="models",
            config_path="config/ensemble_config.json"
        )
        
        execution_gate = AdaptiveExecutionGate(
            symbol=test_config['symbol'],
            config_path="config/execution_config.json"
        )
        
        safety_layer = SafetyRiskLayer(
            config_path="config/safety_config.json"
        )
        
        # Test signal generation
        latest_data = sample_market_data.iloc[-100:]
        signal = decision_engine.predict(latest_data)
        
        assert signal is not None
        assert 'direction' in signal
        assert 'confidence' in signal
        assert 'timestamp' in signal
        
        # Test execution gate
        gate_decision = execution_gate.should_allow_trade(
            signal=signal,
            current_price=1.08500,
            spread=10
        )
        
        assert gate_decision is not None
        assert 'allow_trade' in gate_decision
        assert 'reason' in gate_decision
        
        # Test safety layer
        trade_request = {
            'symbol': test_config['symbol'],
            'action': 'buy',
            'volume': test_config['lot_size'],
            'price': 1.08500,
            'sl': 1.08450,
            'tp': 1.08600,
            'signal': signal
        }
        
        safety_check = safety_layer.validate_trade_request(trade_request)
        
        assert safety_check is not None
        assert 'approved' in safety_check
        assert 'risk_score' in safety_check
        
        # Test risk manager
        risk_check = risk_manager.can_open_position(trade_request)
        
        assert risk_check is not None
        assert 'can_trade' in risk_check
        assert 'reason' in risk_check
        
        print("✅ End-to-end trading workflow test passed")
    
    def test_ml_training_pipeline(self, sample_market_data):
        """Test complete ML training pipeline"""
        
        # Create test data directory
        test_data_dir = Path("test_data")
        test_data_dir.mkdir(exist_ok=True)
        
        # Save sample data
        data_path = test_data_dir / "EURUSD_1m.parquet"
        sample_market_data.to_parquet(data_path)
        
        # Configure training
        config = TrainingConfig(
            symbol="EURUSD",
            timeframe="1m",
            target_horizon="5m",
            n_trials=10,  # Reduced for testing
            data_dir="test_data",
            models_dir="test_models"
        )
        
        # Train models
        trainer = MLTrainer(config)
        model_results = trainer.train_all_models()
        
        # Verify models were created
        assert len(model_results) > 0
        
        for model_type, model_id in model_results.items():
            model_path = Path("test_models") / f"{model_id}.onnx"
            metadata_path = Path("test_models") / f"{model_id}.metadata.json"
            
            assert model_path.exists() or model_path.with_suffix('.pth').exists()
            assert metadata_path.exists()
            
            # Verify metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                assert 'model_id' in metadata
                assert 'metrics' in metadata
                assert 'feature_columns' in metadata
        
        print("✅ ML training pipeline test passed")
    
    def test_inference_server_integration(self):
        """Test ONNX inference server integration"""
        
        # Create test model
        test_model_path = Path("test_models") / "test_model.onnx"
        test_model_path.parent.mkdir(exist_ok=True)
        
        # Create a simple ONNX model for testing
        import onnx
        from onnx import helper, TensorProto
        
        # Create a simple model
        input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 10])
        output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3])
        
        # Create a simple linear model
        weights = np.random.randn(10, 3).astype(np.float32)
        bias = np.random.randn(3).astype(np.float32)
        
        weights_tensor = helper.make_tensor('weights', TensorProto.FLOAT, [10, 3], weights.flatten())
        bias_tensor = helper.make_tensor('bias', TensorProto.FLOAT, [3], bias.flatten())
        
        matmul_node = helper.make_node('MatMul', ['input', 'weights'], ['matmul_out'])
        add_node = helper.make_node('Add', ['matmul_out', 'bias'], ['output'])
        
        graph = helper.make_graph([matmul_node, add_node], 'test_model', 
                                   [input_tensor], [output_tensor], 
                                   [weights_tensor, bias_tensor])
        
        model = helper.make_model(graph)
        onnx.save(model, str(test_model_path))
        
        # Test inference server
        server = InferenceServer(
            model_dir="test_models",
            port=8001,
            hmac_key="test_key"
        )
        
        # Test prediction
        test_features = np.random.randn(1, 10).astype(np.float32).tolist()
        
        prediction = server.predict({
            'features': test_features,
            'model_type': 'test_model',
            'symbol': 'EURUSD'
        })
        
        assert prediction is not None
        assert 'prediction' in prediction
        assert 'confidence' in prediction
        assert 'timestamp' in prediction
        
        print("✅ Inference server integration test passed")
    
    def test_risk_management_integration(self, test_config):
        """Test risk management across all components"""
        
        risk_manager = RiskManager(
            max_risk_percent=test_config['max_risk_percent'],
            max_drawdown_percent=test_config['max_drawdown_percent'],
            max_positions=test_config['max_positions']
        )
        
        safety_layer = SafetyRiskLayer(
            config_path="config/safety_config.json"
        )
        
        # Test position sizing
        account_balance = 10000.0
        risk_per_trade = 0.02  # 2%
        stop_loss_pips = 50
        
        position_size = risk_manager.calculate_position_size(
            account_balance=account_balance,
            risk_percent=risk_per_trade,
            stop_loss_pips=stop_loss_pips,
            symbol=test_config['symbol']
        )
        
        assert position_size > 0
        assert position_size <= test_config['max_positions']
        
        # Test drawdown protection
        current_drawdown = 0.08  # 8%
        
        can_trade = risk_manager.check_drawdown_limit(current_drawdown)
        assert can_trade is True
        
        current_drawdown = 0.12  # 12% - exceeds limit
        can_trade = risk_manager.check_drawdown_limit(current_drawdown)
        assert can_trade is False
        
        # Test safety layer validation
        trade_request = {
            'symbol': test_config['symbol'],
            'action': 'buy',
            'volume': 1.0,
            'price': 1.08500,
            'sl': 1.08450,
            'tp': 1.08600
        }
        
        validation = safety_layer.validate_trade_request(trade_request)
        
        assert validation is not None
        assert 'approved' in validation
        assert 'risk_score' in validation
        assert 'checks_passed' in validation
        
        print("✅ Risk management integration test passed")
    
    def test_telegram_bot_integration(self):
        """Test Telegram bot integration with trading system"""
        
        # Mock Telegram bot
        with patch('telegram.Bot') as mock_bot:
            mock_bot_instance = MagicMock()
            mock_bot.return_value = mock_bot_instance
            
            # Initialize bot
            trading_bot = TradingBot(
                token="test_token",
                chat_id="test_chat_id",
                config_path="config/bot_config.json"
            )
            
            # Test signal notification
            signal = {
                'symbol': 'EURUSD',
                'direction': 'buy',
                'confidence': 0.85,
                'price': 1.08500,
                'timestamp': datetime.now().isoformat()
            }
            
            trading_bot.notify_signal(signal)
            
            # Verify message was sent
            mock_bot_instance.send_message.assert_called_once()
            call_args = mock_bot_instance.send_message.call_args
            
            assert 'text' in call_args[1]
            assert 'EURUSD' in call_args[1]['text']
            assert 'buy' in call_args[1]['text']
            
            print("✅ Telegram bot integration test passed")
    
    def test_rag_teacher_integration(self):
        """Test RAG teacher integration for market analysis"""
        
        # Mock LLM and vector store
        with patch('langchain.llms.OpenAI') as mock_llm:
            with patch('langchain.vectorstores.Chroma') as mock_vectorstore:
                
                # Setup mocks
                mock_llm_instance = MagicMock()
                mock_llm_instance.predict.return_value = "Market shows bullish momentum"
                mock_llm.return_value = mock_llm_instance
                
                mock_vectorstore_instance = MagicMock()
                mock_vectorstore_instance.similarity_search.return_value = [
                    MagicMock(page_content="Bullish pattern detected"),
                    MagicMock(page_content="Strong support level")
                ]
                mock_vectorstore.return_value = mock_vectorstore_instance
                
                # Initialize RAG teacher
                rag_teacher = RAGTeacher(
                    knowledge_base_path="knowledge_base",
                    config_path="config/rag_config.json"
                )
                
                # Test market analysis
                market_data = {
                    'symbol': 'EURUSD',
                    'price': 1.08500,
                    'volume': 1000,
                    'indicators': {
                        'rsi': 65,
                        'macd': 0.0005,
                        'bollinger_position': 'upper'
                    }
                }
                
                analysis = rag_teacher.analyze_market_conditions(market_data)
                
                assert analysis is not None
                assert 'analysis' in analysis
                assert 'confidence' in analysis
                assert 'recommendations' in analysis
                
                print("✅ RAG teacher integration test passed")
    
    def test_performance_under_load(self, sample_market_data):
        """Test system performance under high load"""
        
        # Initialize components
        decision_engine = EnsembleDecisionEngine(
            model_dir="models",
            config_path="config/ensemble_config.json"
        )
        
        execution_gate = AdaptiveExecutionGate(
            symbol='EURUSD',
            config_path="config/execution_config.json"
        )
        
        # Test multiple concurrent predictions
        start_time = time.time()
        
        for i in range(100):  # 100 predictions
            # Simulate different market conditions
            data_slice = sample_market_data.iloc[-100-i:len(sample_market_data)-i]
            
            signal = decision_engine.predict(data_slice)
            gate_decision = execution_gate.should_allow_trade(
                signal=signal,
                current_price=1.08500 + np.random.normal(0, 0.0001),
                spread=10 + np.random.randint(-2, 3)
            )
            
            assert signal is not None
            assert gate_decision is not None
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / 100
        
        # Performance requirements
        assert avg_time < 0.1  # Less than 100ms per prediction
        assert total_time < 10.0  # Less than 10 seconds total
        
        print(f"✅ Performance test passed - Average: {avg_time:.3f}s per prediction")
    
    def test_error_handling_and_recovery(self):
        """Test system error handling and recovery mechanisms"""
        
        # Test with invalid inputs
        decision_engine = EnsembleDecisionEngine(
            model_dir="nonexistent_models",
            config_path="nonexistent_config.json"
        )
        
        # Should handle missing models gracefully
        result = decision_engine.predict(pd.DataFrame())
        
        assert result is not None
        assert 'error' in result or 'direction' in result
        
        # Test execution gate with invalid data
        execution_gate = AdaptiveExecutionGate(
            symbol='INVALID',
            config_path="nonexistent_config.json"
        )
        
        gate_decision = execution_gate.should_allow_trade(
            signal={'direction': 'buy', 'confidence': 0.8},
            current_price=-1.0,  # Invalid price
            spread=-10  # Invalid spread
        )
        
        assert gate_decision is not None
        assert 'allow_trade' in gate_decision
        
        print("✅ Error handling and recovery test passed")
    
    @pytest.mark.asyncio
    async def test_async_operations(self):
        """Test asynchronous operations"""
        
        # Test async data fetching
        async def fetch_market_data(symbol, timeframe):
            # Simulate async API call
            await asyncio.sleep(0.1)
            return pd.DataFrame({
                'timestamp': [datetime.now()],
                'open': [1.08500],
                'high': [1.08510],
                'low': [1.08490],
                'close': [1.08500],
                'volume': [1000]
            })
        
        # Test concurrent operations
        start_time = time.time()
        
        tasks = [
            fetch_market_data('EURUSD', '1m'),
            fetch_market_data('GBPUSD', '1m'),
            fetch_market_data('USDJPY', '1m')
        ]
        
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        assert len(results) == 3
        assert total_time < 0.5  # Should be much faster than sequential
        
        print(f"✅ Async operations test passed - Time: {total_time:.3f}s")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])