#!/usr/bin/env python3
"""
Core Components Test - Tests the main system components without external dependencies
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'ml-brain'))
sys.path.insert(0, str(Path(__file__).parent / 'mt5_executor'))
sys.path.insert(0, str(Path(__file__).parent / 'orchestration'))

def test_config_manager():
    """Test ConfigManager initialization"""
    try:
        from config.config_manager import ConfigManager
        config = ConfigManager()
        print("✅ ConfigManager initialized successfully")
        return True
    except Exception as e:
        print(f"❌ ConfigManager failed: {e}")
        return False

def test_ml_trainer():
    """Test MLTrainer basic functionality"""
    try:
        # Import from ml-brain directory
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "trainer", 
            Path(__file__).parent / "ml-brain" / "ml_brain" / "training" / "trainer.py"
        )
        trainer_module = importlib.util.module_from_spec(spec)
        sys.modules["trainer"] = trainer_module
        spec.loader.exec_module(trainer_module)
        
        # Test config creation
        config = trainer_module.TrainingConfig(
            symbol='EURUSD',
            timeframe='1m',
            target_horizon='5m',
            prediction_type='classification'
        )
        
        # Test trainer initialization (without actual training)
        trainer = trainer_module.MLTrainer(config)
        print("✅ MLTrainer initialized successfully")
        return True
    except Exception as e:
        print(f"❌ MLTrainer failed: {e}")
        return False

def test_onnx_server():
    """Test ONNX Server basic functionality"""
    try:
        # Import from root directory
        import onnx_server
        from onnx_server import PredictionRequest, SystemHealth
        
        # Test request creation
        request = PredictionRequest(
            symbol="EURUSD",
            timeframe="H1",
            features={'price': 1.0, 'volume': 2.0, 'indicator': 3.0},
            model_name="test_model"
        )
        
        # Test health check
        health = SystemHealth(
            status="healthy",
            models_loaded=5,
            cpu_usage=45.2,
            memory_usage=67.8,
            active_connections=12,
            uptime_seconds=3600
        )
        
        print("✅ ONNX Server components working")
        return True
    except Exception as e:
        print(f"❌ ONNX Server failed: {e}")
        return False

def test_risk_management():
    """Test Risk Management components"""
    try:
        # Import from orchestration directory
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "risk_management", 
            Path(__file__).parent / "orchestration" / "risk_management.py"
        )
        risk_module = importlib.util.module_from_spec(spec)
        sys.modules["risk_management"] = risk_module
        spec.loader.exec_module(risk_module)
        
        # Test risk limits creation
        limits = risk_module.RiskLimits(
            max_position_size=100000,
            max_daily_loss=0.05,
            max_drawdown=0.20,
            max_exposure=0.80,
            max_leverage=2.0,
            min_liquidity_buffer=10000.0,
            max_correlation_risk=0.7,
            var_limit=2000.0
        )
        
        # Test risk manager initialization
        risk_manager = risk_module.RiskManager()
        
        print("✅ Risk Management components working")
        return True
    except Exception as e:
        print(f"❌ Risk Management failed: {e}")
        return False

def test_signal_processor():
    """Test Signal Processing components"""
    try:
        # Import from orchestration directory
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "signal_processor", 
            Path(__file__).parent / "orchestration" / "signal_processor.py"
        )
        signal_module = importlib.util.module_from_spec(spec)
        sys.modules["signal_processor"] = signal_module
        spec.loader.exec_module(signal_module)
        
        # Test signal creation
        signal = signal_module.TradingSignal(
            timestamp="2024-01-01T12:00:00Z",
            symbol="EURUSD",
            signal_type="BUY",
            confidence=0.85,
            strength=0.75,
            features={},
            metadata={}
        )
        
        # Test processor initialization
        processor = signal_module.SignalProcessor()
        
        print("✅ Signal Processing components working")
        return True
    except Exception as e:
        print(f"❌ Signal Processing failed: {e}")
        return False

def main():
    """Run all core component tests"""
    print("🚀 Testing TitanovaX Core Components...")
    print("=" * 50)
    
    tests = [
        test_config_manager,
        test_ml_trainer,
        test_onnx_server,
        test_risk_management,
        test_signal_processor
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All core components are working correctly!")
        return 0
    else:
        print("⚠️  Some components need attention")
        return 1

if __name__ == "__main__":
    sys.exit(main())