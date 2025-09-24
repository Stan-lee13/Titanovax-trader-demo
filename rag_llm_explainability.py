#!/usr/bin/env python3
"""
RAG + LLM Explainability System for TitanovaX
Provides post-mortem trade explanations with source verification
"""

import logging
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import hashlib
import requests
from collections import defaultdict

@dataclass
class TradeExplanation:
    """Trade explanation with source verification"""
    trade_id: str
    timestamp: datetime
    symbol: str
    action: str
    size: float
    price: float
    pnl: Optional[float]

    # Model inputs and features
    model_inputs: Dict[str, Any]
    feature_importance: Dict[str, float]
    ensemble_votes: Dict[str, Dict[str, float]]
    regime_context: Dict[str, Any]

    # Generated explanation
    plain_language_explanation: str
    confidence_score: float
    reasoning_steps: List[str]

    # Source verification
    source_citations: List[Dict[str, str]]
    hallucination_score: float
    verification_status: str  # "verified", "partial", "unverified"

    # Audit trail
    explanation_model: str
    generation_time_ms: int
    human_feedback: Optional[str] = None

@dataclass
class SourceDocument:
    """Source document for RAG system"""
    id: str
    content: str
    metadata: Dict[str, Any]
    relevance_score: float
    source_type: str  # "market_data", "news", "technical_analysis", "strategy_doc"

class RAGLLMExplainability:
    """RAG + LLM system for trade explanations with verification"""

    def __init__(self, config_path: str = 'config/explainability_config.json'):
        self.config_path = Path(config_path)
        self.explanation_history: List[TradeExplanation] = []
        self.source_documents: Dict[str, SourceDocument] = {}
        self.max_history = 1000

        self.load_config()
        self.setup_logging()

        # Initialize RAG system
        self._initialize_rag_system()

    def load_config(self):
        """Load explainability configuration"""
        default_config = {
            "llm_model": "gpt-4",
            "llm_temperature": 0.1,
            "max_context_length": 4000,
            "min_confidence_threshold": 0.7,
            "source_verification_enabled": True,
            "telegram_enabled": True,
            "telegram_bot_token": "YOUR_TELEGRAM_BOT_TOKEN",
            "telegram_channel_id": "YOUR_CHANNEL_ID",
            "explanation_prompts": {
                "buy_signal": "Explain why this BUY signal was generated. Focus on market conditions, technical indicators, and ensemble model confidence.",
                "sell_signal": "Explain why this SELL signal was generated. Focus on risk factors, market regime, and exit conditions.",
                "hold_signal": "Explain why no position was taken. Focus on uncertainty, risk factors, and market conditions."
            }
        }

        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    self.config = {**default_config, **json.load(f)}
            else:
                self.config = default_config
                self.save_config()
        except Exception as e:
            logging.warning(f"Could not load explainability config: {e}")
            self.config = default_config

    def save_config(self):
        """Save current configuration"""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)

    def setup_logging(self):
        """Setup logging"""
        self.logger = logging.getLogger(__name__)

    def _initialize_rag_system(self):
        """Initialize RAG system with source documents"""
        # In production, this would load from a vector database
        # For now, initialize with basic trading knowledge
        self._load_default_sources()

    def _load_default_sources(self):
        """Load default source documents for RAG"""
        default_sources = [
            {
                "id": "market_regime_trend",
                "content": "In trending markets, momentum strategies perform well. Look for strong directional movement with increasing volume.",
                "metadata": {"regime": "TREND", "strategy": "momentum"},
                "source_type": "strategy_doc"
            },
            {
                "id": "market_regime_range",
                "content": "Range-bound markets favor mean-reversion strategies. Watch for support/resistance levels and oscillators.",
                "metadata": {"regime": "RANGE", "strategy": "mean_reversion"},
                "source_type": "strategy_doc"
            },
            {
                "id": "risk_management",
                "content": "Always respect position sizing limits. Never risk more than 2% of capital per trade. Use stop losses.",
                "metadata": {"topic": "risk"},
                "source_type": "strategy_doc"
            }
        ]

        for source_data in default_sources:
            source = SourceDocument(
                id=source_data["id"],
                content=source_data["content"],
                metadata=source_data["metadata"],
                relevance_score=0.8,
                source_type=source_data["source_type"]
            )
            self.source_documents[source.id] = source

    def generate_trade_explanation(self, trade_data: Dict[str, Any]) -> TradeExplanation:
        """
        Generate comprehensive trade explanation with source verification

        Args:
            trade_data: Dict containing trade information, model inputs, etc.

        Returns:
            TradeExplanation object with full analysis
        """
        start_time = datetime.now()

        # Extract trade information
        trade_id = trade_data.get("trade_id", f"trade_{datetime.now().timestamp()}")
        symbol = trade_data.get("symbol", "UNKNOWN")
        action = trade_data.get("action", "HOLD")
        size = trade_data.get("size", 0.0)
        price = trade_data.get("price", 0.0)

        # Build explanation context
        context = self._build_explanation_context(trade_data)

        # Generate plain language explanation
        explanation_text = self._generate_llm_explanation(action, context)

        # Find relevant source documents
        relevant_sources = self._find_relevant_sources(context)

        # Calculate hallucination score
        hallucination_score = self._calculate_hallucination_score(explanation_text, relevant_sources)

        # Verify explanation against sources
        verification_status = self._verify_explanation_against_sources(explanation_text, relevant_sources)

        # Build reasoning steps
        reasoning_steps = self._build_reasoning_steps(trade_data)

        # Calculate confidence
        confidence_score = self._calculate_confidence_score(trade_data)

        end_time = datetime.now()
        generation_time_ms = (end_time - start_time).total_seconds() * 1000

        explanation = TradeExplanation(
            trade_id=trade_id,
            timestamp=datetime.now(),
            symbol=symbol,
            action=action,
            size=size,
            price=price,
            pnl=trade_data.get("pnl"),

            model_inputs=trade_data.get("model_inputs", {}),
            feature_importance=trade_data.get("feature_importance", {}),
            ensemble_votes=trade_data.get("ensemble_votes", {}),
            regime_context=trade_data.get("regime_context", {}),

            plain_language_explanation=explanation_text,
            confidence_score=confidence_score,
            reasoning_steps=reasoning_steps,

            source_citations=relevant_sources,
            hallucination_score=hallucination_score,
            verification_status=verification_status,

            explanation_model=self.config["llm_model"],
            generation_time_ms=generation_time_ms
        )

        # Add to history
        self.explanation_history.append(explanation)
        if len(self.explanation_history) > self.max_history:
            self.explanation_history = self.explanation_history[-self.max_history:]

        return explanation

    def _build_explanation_context(self, trade_data: Dict[str, Any]) -> Dict[str, Any]:
        """Build context for LLM explanation"""
        context = {
            "symbol": trade_data.get("symbol", "UNKNOWN"),
            "action": trade_data.get("action", "HOLD"),
            "size": trade_data.get("size", 0.0),
            "price": trade_data.get("price", 0.0),
            "timestamp": trade_data.get("timestamp", datetime.now()).isoformat(),

            "regime": trade_data.get("regime_context", {}).get("predicted_regime", "UNKNOWN"),
            "regime_confidence": trade_data.get("regime_context", {}).get("confidence", 0.0),

            "ensemble_confidence": trade_data.get("ensemble_confidence", 0.0),
            "risk_score": trade_data.get("risk_score", 0.0),

            "top_features": dict(list(trade_data.get("feature_importance", {}).items())[:5]),
            "model_votes": trade_data.get("ensemble_votes", {}),

            "market_conditions": {
                "volatility": trade_data.get("volatility", 0.0),
                "spread": trade_data.get("spread", 0.0),
                "volume": trade_data.get("volume", 0.0)
            }
        }

        return context

    def _generate_llm_explanation(self, action: str, context: Dict[str, Any]) -> str:
        """
        Generate plain language explanation using LLM
        In production, this would call actual LLM API
        """
        # For now, generate explanation based on templates and context
        regime = context.get("regime", "UNKNOWN")
        confidence = context.get("ensemble_confidence", 0.0)
        risk = context.get("risk_score", 0.0)

        if action == "BUY":
            explanation = (
                f"This BUY signal for {context['symbol']} was generated in a {regime} market regime "
                f"with {confidence:.1%} ensemble confidence. "
                f"Key factors included: strong momentum indicators, supportive technical patterns, "
                f"and favorable risk-reward ratio. The model detected opportunity for directional "
                f"movement based on recent price action and volume analysis."
            )
        elif action == "SELL":
            explanation = (
                f"This SELL signal for {context['symbol']} was generated due to "
                f"overbought conditions and potential reversal signals in the {regime} regime. "
                f"Risk management protocols triggered position reduction with {confidence:.1%} confidence. "
                f"Factors included: weakening momentum, technical resistance levels, and "
                f"increasing volatility indicators."
            )
        else:  # HOLD
            explanation = (
                f"No position taken for {context['symbol']} due to insufficient confidence "
                f"({confidence:.1%}) or elevated risk factors ({risk:.1%}). "
                f"The ensemble models showed mixed signals in the {regime} regime, "
                f"warranting a wait-and-see approach until clearer directional bias emerges."
            )

        # Add market condition context
        market = context.get("market_conditions", {})
        if market.get("volatility", 0) > 0.001:
            explanation += f" Market volatility ({market['volatility']:.4f}) influenced position sizing decisions."

        return explanation

    def _find_relevant_sources(self, context: Dict[str, Any]) -> List[Dict[str, str]]:
        """Find relevant source documents for explanation"""
        relevant_sources = []

        # Simple keyword matching (in production, use vector similarity)
        keywords = [
            context.get("regime", "").lower(),
            context.get("action", "").lower(),
            "risk" if context.get("risk_score", 0) > 0.5 else "",
            "momentum" if any("momentum" in str(v) for v in context.get("top_features", {}).values()) else ""
        ]

        for source_id, source in self.source_documents.items():
            relevance = 0
            source_keywords = source_id.lower() + " " + source.content.lower()

            for keyword in keywords:
                if keyword and keyword in source_keywords:
                    relevance += 0.2

            if relevance > 0.3:
                relevant_sources.append({
                    "source_id": source_id,
                    "content": source.content,
                    "relevance_score": relevance,
                    "source_type": source.source_type
                })

        return sorted(relevant_sources, key=lambda x: x["relevance_score"], reverse=True)[:3]

    def _calculate_hallucination_score(self, explanation: str, sources: List[Dict[str, str]]) -> float:
        """Calculate hallucination score based on source alignment"""
        if not sources:
            return 1.0  # High hallucination risk without sources

        # Simple overlap scoring (in production, use NLP metrics)
        explanation_words = set(explanation.lower().split())
        source_words = set()

        for source in sources:
            source_words.update(source["content"].lower().split())

        overlap = len(explanation_words.intersection(source_words)) / len(explanation_words) if explanation_words else 0

        return max(0, 1.0 - overlap)  # Lower overlap = higher hallucination risk

    def _verify_explanation_against_sources(self, explanation: str, sources: List[Dict[str, str]]) -> str:
        """Verify explanation against source documents"""
        if not sources:
            return "unverified"

        hallucination_score = self._calculate_hallucination_score(explanation, sources)

        if hallucination_score < 0.2:
            return "verified"
        elif hallucination_score < 0.5:
            return "partial"
        else:
            return "unverified"

    def _build_reasoning_steps(self, trade_data: Dict[str, Any]) -> List[str]:
        """Build reasoning steps from trade data"""
        steps = []

        # Regime analysis step
        regime = trade_data.get("regime_context", {}).get("predicted_regime", "UNKNOWN")
        regime_confidence = trade_data.get("regime_context", {}).get("confidence", 0.0)
        steps.append(f"Market regime analysis: {regime} (confidence: {regime_confidence:.1%})")

        # Model ensemble step
        ensemble_confidence = trade_data.get("ensemble_confidence", 0.0)
        steps.append(f"Ensemble model confidence: {ensemble_confidence:.1%}")

        # Risk assessment step
        risk_score = trade_data.get("risk_score", 0.0)
        steps.append(f"Risk assessment: {risk_score:.1%}")

        # Position sizing step
        size = trade_data.get("size", 0.0)
        steps.append(f"Position sizing: {size:.2f} lots")

        return steps

    def _calculate_confidence_score(self, trade_data: Dict[str, Any]) -> float:
        """Calculate overall confidence score for explanation"""
        # Weighted combination of model confidence, regime confidence, and risk factors
        model_confidence = trade_data.get("ensemble_confidence", 0.0)
        regime_confidence = trade_data.get("regime_context", {}).get("confidence", 0.0)
        risk_score = trade_data.get("risk_score", 0.0)

        # Risk score is inverted (lower risk = higher confidence)
        risk_confidence = max(0, 1.0 - risk_score)

        return (model_confidence * 0.5 + regime_confidence * 0.3 + risk_confidence * 0.2)

    def send_to_telegram(self, explanation: TradeExplanation):
        """Send explanation to Telegram channel"""
        if not self.config.get("telegram_enabled", False):
            return

        message = f"""
🧠 **Trade Explanation: {explanation.symbol}**

**Action:** {explanation.action}
**Size:** {explanation.size:.2f} lots
**Price:** {explanation.price:.5f}
**Confidence:** {explanation.confidence_score:.1%}

**Explanation:**
{explanation.plain_language_explanation}

**Source Verification:** {explanation.verification_status}
**Hallucination Score:** {explanation.hallucination_score:.2f}

**Reasoning Steps:**
{chr(10).join(f"• {step}" for step in explanation.reasoning_steps)}

**Trade ID:** {explanation.trade_id}
**Generated:** {explanation.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
"""

        try:
            # In production, this would use actual Telegram API
            self.logger.info(f"Telegram message prepared for trade {explanation.trade_id}")
            # requests.post(f"https://api.telegram.org/bot/{self.config['telegram_bot_token']}/sendMessage",
            #               json={"chat_id": self.config['telegram_channel_id'], "text": message})
        except Exception as e:
            self.logger.error(f"Failed to send Telegram message: {e}")

    def save_to_audit_db(self, explanation: TradeExplanation):
        """Save explanation to audit database"""
        # In production, this would save to a proper database
        audit_path = Path("data/audit/trade_explanations.json")

        try:
            # Load existing explanations
            if audit_path.exists():
                with open(audit_path, 'r') as f:
                    all_explanations = json.load(f)
            else:
                all_explanations = []

            # Add new explanation
            explanation_dict = {
                "trade_id": explanation.trade_id,
                "timestamp": explanation.timestamp.isoformat(),
                "symbol": explanation.symbol,
                "action": explanation.action,
                "size": explanation.size,
                "price": explanation.price,
                "pnl": explanation.pnl,
                "explanation": explanation.plain_language_explanation,
                "confidence_score": explanation.confidence_score,
                "verification_status": explanation.verification_status,
                "hallucination_score": explanation.hallucination_score
            }

            all_explanations.append(explanation_dict)

            # Keep only last 1000 explanations
            if len(all_explanations) > 1000:
                all_explanations = all_explanations[-1000:]

            # Save to file
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            with open(audit_path, 'w') as f:
                json.dump(all_explanations, f, indent=2)

        except Exception as e:
            self.logger.error(f"Failed to save to audit DB: {e}")

    def get_explanation_stats(self) -> Dict[str, Any]:
        """Get statistics about explanation system"""
        if not self.explanation_history:
            return {"total_explanations": 0}

        verification_counts = {"verified": 0, "partial": 0, "unverified": 0}
        avg_confidence = 0
        avg_hallucination = 0

        for explanation in self.explanation_history:
            verification_counts[explanation.verification_status] += 1
            avg_confidence += explanation.confidence_score
            avg_hallucination += explanation.hallucination_score

        total = len(self.explanation_history)
        avg_confidence /= total
        avg_hallucination /= total

        return {
            "total_explanations": total,
            "verification_distribution": verification_counts,
            "average_confidence": avg_confidence,
            "average_hallucination_score": avg_hallucination,
            "last_explanation": self.explanation_history[-1].timestamp.isoformat() if self.explanation_history else None
        }

if __name__ == "__main__":
    # Demo usage
    explainability_system = RAGLLMExplainability()

    # Example trade data
    trade_data = {
        "trade_id": "demo_trade_001",
        "symbol": "EURUSD",
        "action": "BUY",
        "size": 0.1,
        "price": 1.12345,
        "ensemble_confidence": 0.85,
        "risk_score": 0.3,
        "regime_context": {
            "predicted_regime": "TREND",
            "confidence": 0.78
        },
        "feature_importance": {
            "rsi": 0.25,
            "macd": 0.20,
            "volume": 0.15,
            "momentum": 0.40
        },
        "ensemble_votes": {
            "xgboost": {"up": 0.7, "down": 0.2, "sideways": 0.1},
            "transformer": {"up": 0.6, "down": 0.25, "sideways": 0.15}
        }
    }

    # Generate explanation
    explanation = explainability_system.generate_trade_explanation(trade_data)

    print("Trade Explanation Generated:")
    print(f"Action: {explanation.action}")
    print(f"Explanation: {explanation.plain_language_explanation}")
    print(f"Verification Status: {explanation.verification_status}")
    print(f"Confidence: {explanation.confidence_score:.2%}")

    # Send to Telegram (demo)
    explainability_system.send_to_telegram(explanation)

    # Save to audit
    explainability_system.save_to_audit_db(explanation)

    # Show stats
    stats = explainability_system.get_explanation_stats()
    print(f"\nSystem Stats: {stats}")
