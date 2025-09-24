#!/usr/bin/env python3
"""
Simple test script to identify issues
"""

import sys
import traceback

try:
    print("Testing individual components...")

    # Test 1: Import all modules
    print("1. Importing modules...")
    from adaptive_execution_gate import AdaptiveExecutionGate
    from smart_order_router import SmartOrderRouter
    from micro_slippage_model import MicroSlippageModel
    from regime_classifier import RegimeClassifier
    from ensemble_decision_engine import EnsembleDecisionEngine
    from safety_risk_layer import RiskEngine, AnomalyDetector
    from watchdog_self_healing import Watchdog
    print("   ✓ All modules imported successfully")

    # Test 2: Create instances
    print("2. Creating component instances...")
    execution_gate = AdaptiveExecutionGate()
    order_router = SmartOrderRouter()
    slippage_model = MicroSlippageModel()
    print("   ✓ Component instances created")

    # Test 3: Test basic functionality
    print("3. Testing basic functionality...")
    result = execution_gate.adaptive_spread_check("EURUSD")
    print(f"   ✓ Execution gate test: {result['allowed']}")

    estimate = slippage_model.estimate_slippage("EURUSD", "market", 0.1, "normal")
    print(f"   ✓ Slippage model test: {estimate:.2f} pips")

    print("✅ All basic tests passed!")

except Exception as e:
    print(f"❌ Error: {e}")
    print("Full traceback:")
    traceback.print_exc()
    sys.exit(1)
