#!/usr/bin/env python3
"""
FastAPI Inference Server for TitanovaX Trading System
Serves ONNX models for low-latency trading signal generation
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import onnxruntime as rt
import numpy as np
import pandas as pd
from pathlib import Path
import json
import time
import logging
import asyncio
from datetime import datetime
import hashlib
import hmac
import secrets
from concurrent.futures import ThreadPoolExecutor
import psutil
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/logs/inference_server.log'),
        logging.StreamHandler()
    ]
)

# Pydantic models for request/response
class PredictionRequest(BaseModel):
    symbol: str = Field(..., description="Trading symbol (e.g., EURUSD)")
    features: Dict[str, float] = Field(..., description="Feature values for prediction")
    model_id: Optional[str] = Field(None, description="Specific model ID to use")
    horizon: str = Field("5m", description="Prediction horizon (5m, 30m, 1h, 24h)")
    prediction_type: str = Field("classification", description="classification or regression")

class PredictionResponse(BaseModel):
    model_id: str
    symbol: str
    prediction: float
    probability_up: Optional[float] = None
    probability_down: Optional[float] = None
    confidence: float
    processing_time_ms: float
    timestamp: str

class ModelInfo(BaseModel):
    model_id: str
    symbol: str
    timeframe: str
    horizon: str
    type: str
    created_at: str
    metrics: Dict[str, float]

class SystemHealth(BaseModel):
    status: str
    models_loaded: int
    cpu_usage: float
    memory_usage: float
    active_connections: int
    uptime_seconds: float

class InferenceServer:
    def __init__(self, models_dir='models', port=8001):
        self.models_dir = Path(models_dir)
        self.port = port
        self.models = {}
        self.model_metadata = {}
        self.sessions = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.start_time = time.time()
        self.hmac_secret = self.load_hmac_secret()

        # Create FastAPI app
        self.app = FastAPI(
            title="TitanovaX Inference Server",
            description="High-performance ML model inference for trading signals",
            version="2.0.0"
        )

        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Setup routes
        self.setup_routes()

    def load_hmac_secret(self) -> bytes:
        """Load HMAC secret for request signing"""
        secret_file = Path('secrets/hmac_secret.key')
        if secret_file.exists():
            with open(secret_file, 'rb') as f:
                return f.read()
        else:
            # Generate temporary secret for development
            logging.warning("No HMAC secret found, generating temporary secret")
            return secrets.token_bytes(32)

    def load_models(self):
        """Load all available ONNX models"""

        if not self.models_dir.exists():
            logging.error(f"Models directory not found: {self.models_dir}")
            return

        onnx_files = list(self.models_dir.glob('*_*.onnx'))

        for onnx_file in onnx_files:
            try:
                # Load model metadata
                metadata_file = onnx_file.with_suffix('.metadata.json')
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)

                    model_id = metadata['model_id']
                    symbol = metadata['symbol']
                    horizon = metadata['target_horizon']

                    # Load ONNX model
                    session = rt.InferenceSession(str(onnx_file))

                    # Store model info
                    self.models[f"{symbol}_{horizon}"] = {
                        'session': session,
                        'metadata': metadata,
                        'input_name': session.get_inputs()[0].name,
                        'output_name': session.get_outputs()[0].name
                    }

                    self.model_metadata[model_id] = metadata

                    logging.info(f"Loaded model: {symbol} -> {horizon} (ID: {model_id})")

                else:
                    logging.warning(f"No metadata found for {onnx_file}")

            except Exception as e:
                logging.error(f"Failed to load model {onnx_file}: {e}")

        logging.info(f"Loaded {len(self.models)} models")

    def predict(self, symbol: str, features: Dict[str, float], horizon: str = '5m',
               model_id: Optional[str] = None) -> Dict[str, Any]:
        """Make prediction using loaded model"""

        start_time = time.time()

        # Select model
        model_key = f"{symbol}_{horizon}"
        if model_key not in self.models:
            raise HTTPException(status_code=404, detail=f"No model available for {symbol} -> {horizon}")

        model_info = self.models[model_key]
        session = model_info['session']
        metadata = model_info['metadata']

        # Prepare input features
        feature_names = metadata['feature_columns']

        # Ensure all required features are present
        missing_features = set(feature_names) - set(features.keys())
        if missing_features:
            raise HTTPException(
                status_code=400,
                detail=f"Missing features: {missing_features}"
            )

        # Create input array in correct order
        input_data = np.array([[features[feat] for feat in feature_names]], dtype=np.float32)

        # Run inference
        try:
            outputs = session.run(None, {model_info['input_name']: input_data})
            prediction = float(outputs[0][0][0])

            processing_time = (time.time() - start_time) * 1000

            # Calculate confidence
            confidence = abs(prediction - 0.5) * 2  # Scale to 0-1

            result = {
                'model_id': metadata['model_id'],
                'symbol': symbol,
                'prediction': prediction,
                'confidence': confidence,
                'processing_time_ms': processing_time,
                'timestamp': datetime.now().isoformat()
            }

            # Add classification probabilities if available
            if 'probability_up' in outputs[0][0]:
                result['probability_up'] = float(outputs[0][0][1])
                result['probability_down'] = float(outputs[0][0][2])

            return result

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    def verify_hmac_signature(self, request_data: Dict, signature: str) -> bool:
        """Verify HMAC signature of request"""

        if not self.hmac_secret:
            return True  # Skip verification if no secret

        # Create canonical request string
        canonical_request = json.dumps(request_data, separators=(',', ':'), sort_keys=True)

        # Calculate expected signature
        expected_signature = hmac.new(
            self.hmac_secret,
            canonical_request.encode(),
            hashlib.sha256
        ).hexdigest()

        return hmac.compare_digest(expected_signature, signature)

    def get_system_health(self) -> Dict[str, Any]:
        """Get system health metrics"""

        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()

        return {
            'status': 'healthy' if len(self.models) > 0 else 'degraded',
            'models_loaded': len(self.models),
            'cpu_usage': cpu_percent,
            'memory_usage': memory.percent,
            'memory_used_gb': memory.used / (1024**3),
            'memory_available_gb': memory.available / (1024**3),
            'active_connections': len([p for p in psutil.net_connections() if p.status == 'ESTABLISHED']),
            'uptime_seconds': time.time() - self.start_time
        }

    def setup_routes(self):
        """Setup FastAPI routes"""

        @self.app.get("/")
        async def root():
            return {
                "service": "TitanovaX Inference Server",
                "version": "2.0.0",
                "status": "running",
                "models_loaded": len(self.models),
                "uptime": time.time() - self.start_time
            }

        @self.app.get("/health", response_model=SystemHealth)
        async def health_check():
            """System health check endpoint"""
            health = self.get_system_health()
            return SystemHealth(**health)

        @self.app.get("/models", response_model=List[ModelInfo])
        async def list_models():
            """List all available models"""
            models_info = []
            for model_info in self.model_metadata.values():
                models_info.append(ModelInfo(**model_info))
            return models_info

        @self.app.get("/models/{model_id}")
        async def get_model_info(model_id: str):
            """Get information about a specific model"""
            if model_id not in self.model_metadata:
                raise HTTPException(status_code=404, detail="Model not found")
            return self.model_metadata[model_id]

        @self.app.post("/predict", response_model=PredictionResponse)
        async def predict_endpoint(request: PredictionRequest):
            """Make prediction using ML model"""

            try:
                # Make prediction
                result = self.predict(
                    request.symbol,
                    request.features,
                    request.horizon,
                    request.model_id
                )

                return PredictionResponse(**result)

            except Exception as e:
                logging.error(f"Prediction error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/batch_predict")
        async def batch_predict(requests: List[PredictionRequest]):
            """Make batch predictions"""

            async def predict_single(req):
                try:
                    return self.predict(
                        req.symbol,
                        req.features,
                        req.horizon,
                        req.model_id
                    )
                except Exception as e:
                    return {"error": str(e), "symbol": req.symbol}

            # Run predictions concurrently
            tasks = [predict_single(req) for req in requests]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            return {"predictions": results}

        @self.app.post("/signed_predict")
        async def signed_predict(request: PredictionRequest, signature: str):
            """Make prediction with HMAC signature verification"""

            # Verify signature
            request_data = request.dict()
            if not self.verify_hmac_signature(request_data, signature):
                raise HTTPException(status_code=401, detail="Invalid signature")

            # Make prediction
            result = self.predict(
                request.symbol,
                request.features,
                request.horizon,
                request.model_id
            )

            return PredictionResponse(**result)

        @self.app.get("/metrics")
        async def get_metrics():
            """Prometheus-style metrics endpoint"""

            health = self.get_system_health()

            metrics = f"""
# HELP titanovax_models_loaded Number of loaded ML models
# TYPE titanovax_models_loaded gauge
titanovax_models_loaded {len(self.models)}

# HELP titanovax_cpu_usage CPU usage percentage
# TYPE titanovax_cpu_usage gauge
titanovax_cpu_usage {health['cpu_usage']}

# HELP titanovax_memory_usage Memory usage percentage
# TYPE titanovax_memory_usage gauge
titanovax_memory_usage {health['memory_usage']}

# HELP titanovax_uptime_seconds Server uptime in seconds
# TYPE titanovax_uptime_seconds gauge
titanovax_uptime_seconds {health['uptime_seconds']}
"""

            return metrics

    def run(self):
        """Run the FastAPI server"""

        import uvicorn

        logging.info(f"Starting TitanovaX Inference Server on port {self.port}")
        logging.info(f"Models directory: {self.models_dir}")
        logging.info(f"Loaded {len(self.models)} models")

        uvicorn.run(
            self.app,
            host="0.0.0.0",
            port=self.port,
            log_level="info"
        )

def main():
    """Main function to run inference server"""

    print("=== TitanovaX Inference Server v2.0 ===")

    # Initialize server
    server = InferenceServer()

    # Load models
    print("Loading models...")
    server.load_models()

    # Start server
    print("Starting server...")
    server.run()

if __name__ == "__main__":
    main()
