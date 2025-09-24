"""
TitanovaX RAG/LLM Integration Module
Retrieval-Augmented Generation and Large Language Model integration for explainability
"""

import asyncio
import json
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import sqlite3
import faiss
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VectorMemoryStore:
    """Vector memory store for RAG system"""
    
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.index = None
        self.documents = []
        self.metadata = []
        self.dimension = 384  # Dimension for all-MiniLM-L6-v2
        
    def initialize(self):
        """Initialize vector store"""
        try:
            # Create FAISS index
            self.index = faiss.IndexFlatL2(self.dimension)
            logger.info("✅ Vector memory store initialized")
            return True
        except Exception as e:
            logger.error(f"Vector store initialization failed: {e}")
            return False
    
    def add_documents(self, documents: List[str], metadata: List[Dict[str, Any]]):
        """Add documents to vector store"""
        try:
            # Generate embeddings
            embeddings = self.embedding_model.encode(documents)
            
            # Add to FAISS index
            if self.index is not None:
                self.index.add(np.array(embeddings).astype('float32'))
                self.documents.extend(documents)
                self.metadata.extend(metadata)
                
                logger.info(f"Added {len(documents)} documents to vector store")
                return True
            else:
                logger.error("Vector index not initialized")
                return False
                
        except Exception as e:
            logger.error(f"Document addition failed: {e}")
            return False
    
    def search(self, query: str, k: int = 5) -> List[Tuple[str, Dict[str, Any], float]]:
        """Search for relevant documents"""
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])
            
            # Search in FAISS index
            if self.index is not None and len(self.documents) > 0:
                distances, indices = self.index.search(np.array(query_embedding).astype('float32'), k)
                
                results = []
                for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                    if idx < len(self.documents):
                        similarity = 1 / (1 + distance)  # Convert distance to similarity
                        results.append((
                            self.documents[idx],
                            self.metadata[idx],
                            similarity
                        ))
                
                return results
            else:
                logger.warning("No documents in vector store")
                return []
                
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

class TradingKnowledgeBase:
    """Trading knowledge base for RAG system"""
    
    def __init__(self):
        self.vector_store = VectorMemoryStore()
        self.knowledge_documents = []
        self.knowledge_metadata = []
        
    def initialize(self):
        """Initialize knowledge base"""
        try:
            # Initialize vector store
            self.vector_store.initialize()
            
            # Load trading knowledge
            self.load_trading_knowledge()
            
            # Add documents to vector store
            self.vector_store.add_documents(self.knowledge_documents, self.knowledge_metadata)
            
            logger.info("✅ Trading knowledge base initialized")
            return True
            
        except Exception as e:
            logger.error(f"Knowledge base initialization failed: {e}")
            return False
    
    def load_trading_knowledge(self):
        """Load trading knowledge documents"""
        try:
            # Trading strategies
            strategies = [
                {
                    "text": "Trend following strategy involves identifying and trading in the direction of established market trends. Key indicators include moving averages, MACD, and ADX. Entry signals occur when price breaks above/below trend lines with volume confirmation.",
                    "metadata": {"type": "strategy", "category": "trend_following", "complexity": "medium"}
                },
                {
                    "text": "Mean reversion strategy capitalizes on price movements that deviate significantly from historical averages. Uses Bollinger Bands, RSI, and standard deviation. Best performed in ranging markets with clear support and resistance levels.",
                    "metadata": {"type": "strategy", "category": "mean_reversion", "complexity": "medium"}
                },
                {
                    "text": "Breakout trading focuses on identifying key support and resistance levels, entering positions when price breaks through these levels with increased volume. Requires confirmation through multiple timeframes and momentum indicators.",
                    "metadata": {"type": "strategy", "category": "breakout", "complexity": "high"}
                },
                {
                    "text": "Scalping strategy involves making multiple quick trades to capture small price movements. Requires tight spreads, fast execution, and liquid markets. Typically uses 1-5 minute timeframes with high leverage.",
                    "metadata": {"type": "strategy", "category": "scalping", "complexity": "high"}
                }
            ]
            
            # Risk management
            risk_docs = [
                {
                    "text": "Position sizing should never exceed 2% of total account balance per trade. Use the formula: Position Size = (Account Balance × Risk %) ÷ (Entry Price - Stop Loss Price). This ensures sustainable trading and capital preservation.",
                    "metadata": {"type": "risk_management", "category": "position_sizing", "complexity": "basic"}
                },
                {
                    "text": "Stop loss orders should be placed at logical technical levels, typically below support levels for long positions and above resistance for short positions. Consider market volatility using ATR to avoid premature stop-outs.",
                    "metadata": {"type": "risk_management", "category": "stop_loss", "complexity": "medium"}
                },
                {
                    "text": "Maximum daily drawdown should be limited to 3% of account balance. If this limit is reached, all trading should be halted for the day to prevent emotional decision-making and excessive losses.",
                    "metadata": {"type": "risk_management", "category": "drawdown", "complexity": "basic"}
                },
                {
                    "text": "Correlation risk occurs when multiple positions move in the same direction. Limit exposure to correlated pairs (like EUR/USD and GBP/USD) to maximum 5% total account risk. Use correlation matrices to monitor relationships.",
                    "metadata": {"type": "risk_management", "category": "correlation", "complexity": "high"}
                }
            ]
            
            # Market analysis
            analysis_docs = [
                {
                    "text": "Technical analysis uses historical price data to predict future movements. Key principles include: price discounts all information, prices move in trends, and history tends to repeat. Combine multiple timeframes for confirmation.",
                    "metadata": {"type": "analysis", "category": "technical", "complexity": "basic"}
                },
                {
                    "text": "Support and resistance levels are price zones where buying or selling pressure historically emerges. These levels become stronger with more touches and higher volume. Breaks above resistance or below support often signal trend changes.",
                    "metadata": {"type": "analysis", "category": "support_resistance", "complexity": "medium"}
                },
                {
                    "text": "Market regimes describe different market behaviors: trending (sustained directional movement), ranging (sideways movement), volatile (high volatility), and crisis (extreme conditions). Each regime requires different strategies and risk parameters.",
                    "metadata": {"type": "analysis", "category": "market_regimes", "complexity": "high"}
                },
                {
                    "text": "Volume analysis confirms price movements. High volume on breakouts indicates strong conviction, while low volume suggests weak moves that may reverse. Volume should increase in the direction of the trend for sustainability.",
                    "metadata": {"type": "analysis", "category": "volume", "complexity": "medium"}
                }
            ]
            
            # Combine all knowledge
            all_knowledge = strategies + risk_docs + analysis_docs
            
            for doc in all_knowledge:
                self.knowledge_documents.append(doc["text"])
                self.knowledge_metadata.append(doc["metadata"])
            
            logger.info(f"Loaded {len(all_knowledge)} knowledge documents")
            
        except Exception as e:
            logger.error(f"Trading knowledge loading failed: {e}")
    
    def get_relevant_knowledge(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Get relevant knowledge for a query"""
        try:
            # Search vector store
            results = self.vector_store.search(query, k)
            
            # Format results
            relevant_knowledge = []
            for text, metadata, similarity in results:
                relevant_knowledge.append({
                    "text": text,
                    "metadata": metadata,
                    "relevance_score": similarity
                })
            
            return relevant_knowledge
            
        except Exception as e:
            logger.error(f"Knowledge retrieval failed: {e}")
            return []

class LLMGenerator:
    """Large Language Model for generating explanations"""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.generator = None
        
    def initialize(self):
        """Initialize LLM"""
        try:
            # Initialize tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            
            # Create text generation pipeline
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_length=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            logger.info(f"✅ LLM initialized: {self.model_name}")
            return True
            
        except Exception as e:
            logger.error(f"LLM initialization failed: {e}")
            return False
    
    def generate_explanation(self, prompt: str, context: str = "") -> str:
        """Generate explanation using LLM"""
        try:
            # Create enhanced prompt with context
            enhanced_prompt = f"""
            You are TitanovaX, an AI trading assistant. Provide clear, professional explanations for trading decisions.
            
            Context: {context}
            
            User Query: {prompt}
            
            Provide a detailed, easy-to-understand explanation:
            """
            
            # Generate response
            if self.generator:
                response = self.generator(enhanced_prompt, max_length=512, num_return_sequences=1)
                explanation = response[0]['generated_text'].strip()
                
                # Clean up the response
                explanation = explanation.replace(enhanced_prompt, "").strip()
                
                return explanation
            else:
                return "AI explanation system is currently unavailable."
                
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return "Unable to generate explanation at this time."

class RAGLLMIntegration:
    """Main RAG/LLM integration system"""
    
    def __init__(self):
        self.knowledge_base = TradingKnowledgeBase()
        self.llm = LLMGenerator()
        self.trade_memory = []
        self.explanation_cache = {}
        
    def initialize(self):
        """Initialize RAG/LLM system"""
        try:
            # Initialize knowledge base
            kb_success = self.knowledge_base.initialize()
            
            # Initialize LLM
            llm_success = self.llm.initialize()
            
            if kb_success and llm_success:
                logger.info("✅ RAG/LLM integration system initialized")
                return True
            else:
                logger.error("RAG/LLM initialization incomplete")
                return False
                
        except Exception as e:
            logger.error(f"RAG/LLM integration initialization failed: {e}")
            return False
    
    def generate_trade_explanation(self, trade_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive trade explanation"""
        try:
            # Create cache key
            cache_key = f"{trade_data.get('symbol', '')}_{trade_data.get('direction', '')}_{trade_data.get('timestamp', '')}"
            
            # Check cache
            if cache_key in self.explanation_cache:
                return self.explanation_cache[cache_key]
            
            # Build context from trade data
            context = self.build_trade_context(trade_data)
            
            # Get relevant knowledge
            query = f"{trade_data.get('direction', '').lower()} {trade_data.get('symbol', '')} trading strategy risk management"
            relevant_knowledge = self.knowledge_base.get_relevant_knowledge(query, k=3)
            
            # Build comprehensive prompt
            prompt = self.build_explanation_prompt(trade_data, relevant_knowledge)
            
            # Generate explanation
            explanation = self.llm.generate_explanation(prompt, context)
            
            # Create structured explanation
            structured_explanation = {
                "trade_summary": self.generate_trade_summary(trade_data),
                "ai_reasoning": explanation,
                "risk_analysis": self.generate_risk_analysis(trade_data, relevant_knowledge),
                "market_context": self.generate_market_context(trade_data),
                "confidence_breakdown": self.generate_confidence_breakdown(trade_data),
                "recommended_actions": self.generate_recommended_actions(trade_data),
                "timestamp": datetime.now().isoformat(),
                "knowledge_sources": [k["metadata"] for k in relevant_knowledge]
            }
            
            # Cache explanation
            self.explanation_cache[cache_key] = structured_explanation
            
            return structured_explanation
            
        except Exception as e:
            logger.error(f"Trade explanation generation failed: {e}")
            return {
                "error": "Failed to generate trade explanation",
                "basic_info": self.generate_trade_summary(trade_data),
                "timestamp": datetime.now().isoformat()
            }
    
    def build_trade_context(self, trade_data: Dict[str, Any]) -> str:
        """Build context from trade data"""
        try:
            symbol = trade_data.get('symbol', 'Unknown')
            direction = trade_data.get('direction', 'Unknown')
            confidence = trade_data.get('confidence', 0)
            regime = trade_data.get('regime', 'Unknown')
            size = trade_data.get('size', 0)
            model_sources = trade_data.get('model_sources', [])
            
            context = f"""
            Trade Context:
            - Symbol: {symbol}
            - Direction: {direction}
            - Confidence: {confidence:.2f}
            - Market Regime: {regime}
            - Position Size: {size:.4f}
            - AI Models: {', '.join(model_sources)}
            - Timestamp: {trade_data.get('timestamp', 'Unknown')}
            """
            
            return context
            
        except Exception as e:
            logger.error(f"Trade context building failed: {e}")
            return "Trade context unavailable"
    
    def build_explanation_prompt(self, trade_data: Dict[str, Any], relevant_knowledge: List[Dict[str, Any]]) -> str:
        """Build explanation prompt"""
        try:
            # Extract key information
            symbol = trade_data.get('symbol', 'Unknown')
            direction = trade_data.get('direction', 'Unknown')
            confidence = trade_data.get('confidence', 0)
            regime = trade_data.get('regime', 'Unknown')
            
            # Build knowledge context
            knowledge_context = ""
            for i, knowledge in enumerate(relevant_knowledge, 1):
                knowledge_context += f"\nKnowledge Source {i}: {knowledge['text'][:200]}...\n"
            
            prompt = f"""
            Explain this {direction} trade decision for {symbol} in {regime} market conditions.
            
            Trade Details:
            - Confidence Level: {confidence:.2f}
            - Position Size: {trade_data.get('size', 0):.4f}
            - Model Sources: {', '.join(trade_data.get('model_sources', []))}
            
            Relevant Trading Knowledge:{knowledge_context}
            
            Provide a clear, professional explanation that covers:
            1. Why this trade was recommended
            2. Risk considerations and management
            3. Market context and timing
            4. Confidence level justification
            5. Recommended next steps
            
            Make the explanation educational and actionable for the trader.
            """
            
            return prompt
            
        except Exception as e:
            logger.error(f"Explanation prompt building failed: {e}")
            return f"Explain {direction} trade for {symbol}"
    
    def generate_trade_summary(self, trade_data: Dict[str, Any]) -> str:
        """Generate trade summary"""
        try:
            symbol = trade_data.get('symbol', 'Unknown')
            direction = trade_data.get('direction', 'Unknown')
            confidence = trade_data.get('confidence', 0)
            size = trade_data.get('size', 0)
            
            summary = f"""
            **Trade Summary:**
            - Symbol: {symbol}
            - Direction: {direction}
            - Confidence: {confidence:.2f} ({'High' if confidence > 0.8 else 'Medium' if confidence > 0.6 else 'Low'})
            - Position Size: {size:.4f}
            - Market Regime: {trade_data.get('regime', 'Unknown')}
            - Execution Time: {trade_data.get('timestamp', 'Unknown')}
            """
            
            return summary.strip()
            
        except Exception as e:
            logger.error(f"Trade summary generation failed: {e}")
            return "Trade summary unavailable"
    
    def generate_risk_analysis(self, trade_data: Dict[str, Any], relevant_knowledge: List[Dict[str, Any]]) -> str:
        """Generate risk analysis"""
        try:
            # Extract risk-related knowledge
            risk_knowledge = [k for k in relevant_knowledge if k['metadata'].get('type') == 'risk_management']
            
            risk_analysis = f"""
            **Risk Analysis:**
            - Position Size Risk: {trade_data.get('size', 0) * 100:.2f}% of account (within 2% limit)
            - Market Regime Risk: {self.assess_regime_risk(trade_data.get('regime', 'Unknown'))}
            - Confidence Risk: {'Low' if trade_data.get('confidence', 0) > 0.8 else 'Moderate' if trade_data.get('confidence', 0) > 0.6 else 'High'}
            """
            
            if risk_knowledge:
                risk_analysis += f"\n- Risk Management: {risk_knowledge[0]['text'][:150]}..."
            
            return risk_analysis.strip()
            
        except Exception as e:
            logger.error(f"Risk analysis generation failed: {e}")
            return "Risk analysis unavailable"
    
    def assess_regime_risk(self, regime: str) -> str:
        """Assess risk based on market regime"""
        regime_risks = {
            "TREND_UP": "Low - Favorable for long positions",
            "TREND_DOWN": "Low - Favorable for short positions", 
            "RANGE": "Moderate - Watch for false breakouts",
            "VOLATILE": "High - Increased uncertainty",
            "CRISIS": "Very High - Extreme caution advised"
        }
        return regime_risks.get(regime, "Unknown - Assess carefully")
    
    def generate_market_context(self, trade_data: Dict[str, Any]) -> str:
        """Generate market context"""
        try:
            symbol = trade_data.get('symbol', 'Unknown')
            regime = trade_data.get('regime', 'Unknown')
            
            context = f"""
            **Market Context:**
            - Symbol: {symbol}
            - Current Regime: {regime}
            - Regime Suitability: {self.get_regime_suitability(regime, trade_data.get('direction', 'Unknown'))}
            - Volatility Assessment: {self.assess_volatility_context(regime)}
            """
            
            return context.strip()
            
        except Exception as e:
            logger.error(f"Market context generation failed: {e}")
            return "Market context unavailable"
    
    def get_regime_suitability(self, regime: str, direction: str) -> str:
        """Get regime suitability for trade direction"""
        if direction.upper() == "BUY":
            if regime == "TREND_UP":
                return "Excellent - Strong uptrend support"
            elif regime == "RANGE":
                return "Good - Buy near support levels"
            else:
                return "Caution - Assess trend carefully"
        elif direction.upper() == "SELL":
            if regime == "TREND_DOWN":
                return "Excellent - Strong downtrend support"
            elif regime == "RANGE":
                return "Good - Sell near resistance levels"
            else:
                return "Caution - Assess trend carefully"
        else:
            return "Neutral - Market analysis required"
    
    def assess_volatility_context(self, regime: str) -> str:
        """Assess volatility context"""
        volatility_context = {
            "TREND_UP": "Normal volatility - Trend continuation likely",
            "TREND_DOWN": "Normal volatility - Trend continuation likely",
            "RANGE": "Low volatility - Breakout watch recommended",
            "VOLATILE": "High volatility - Wider stops advised",
            "CRISIS": "Extreme volatility - Consider position reduction"
        }
        return volatility_context.get(regime, "Monitor volatility closely")
    
    def generate_confidence_breakdown(self, trade_data: Dict[str, Any]) -> str:
        """Generate confidence breakdown"""
        try:
            confidence = trade_data.get('confidence', 0)
            model_sources = trade_data.get('model_sources', [])
            
            breakdown = f"""
            **Confidence Breakdown:**
            - Overall Confidence: {confidence:.2f}
            - Model Consensus: {len(model_sources)} models agree
            - Contributing Models: {', '.join(model_sources)}
            - Confidence Level: {'Very High' if confidence > 0.9 else 'High' if confidence > 0.8 else 'Moderate' if confidence > 0.6 else 'Low'}
            """
            
            return breakdown.strip()
            
        except Exception as e:
            logger.error(f"Confidence breakdown generation failed: {e}")
            return "Confidence breakdown unavailable"
    
    def generate_recommended_actions(self, trade_data: Dict[str, Any]) -> List[str]:
        """Generate recommended actions"""
        try:
            actions = []
            
            # Basic actions
            actions.append("Monitor position closely for first 30 minutes")
            actions.append("Set stop loss at technical support/resistance level")
            actions.append("Take partial profits if position moves 1.5% in favor")
            
            # Regime-specific actions
            regime = trade_data.get('regime', 'Unknown')
            if regime == "VOLATILE":
                actions.append("Consider reducing position size due to high volatility")
                actions.append("Use wider stop loss to avoid premature exit")
            elif regime == "CRISIS":
                actions.append("Exercise extreme caution - consider position closure")
                actions.append("Monitor news and market sentiment closely")
            
            # Confidence-specific actions
            confidence = trade_data.get('confidence', 0)
            if confidence < 0.7:
                actions.append("Lower confidence - consider smaller position size")
                actions.append("Wait for additional confirmation before adding to position")
            
            return actions
            
        except Exception as e:
            logger.error(f"Recommended actions generation failed: {e}")
            return ["Monitor position and follow risk management rules"]
    
    def clear_cache(self):
        """Clear explanation cache"""
        self.explanation_cache.clear()
        logger.info("Explanation cache cleared")

async def main():
    """Main function for testing RAG/LLM system"""
    logger.info("Testing TitanovaX RAG/LLM Integration...")
    
    # Create RAG/LLM system
    rag_system = RAGLLMIntegration()
    
    # Initialize system
    if await asyncio.to_thread(rag_system.initialize):
        logger.info("✅ RAG/LLM system initialized successfully")
        
        # Test trade explanation
        test_trade = {
            "symbol": "EURUSD",
            "direction": "BUY",
            "confidence": 0.85,
            "regime": "TREND_UP",
            "size": 0.02,
            "timestamp": datetime.now().isoformat(),
            "model_sources": ["xgboost", "transformer", "ensemble"],
            "explanation": "Strong upward trend detected with multiple confirmations"
        }
        
        # Generate explanation
        explanation = await asyncio.to_thread(rag_system.generate_trade_explanation, test_trade)
        
        logger.info("Generated Trade Explanation:")
        logger.info(json.dumps(explanation, indent=2))
        
        logger.info("✅ RAG/LLM system test completed successfully")
    else:
        logger.error("❌ RAG/LLM system initialization failed")

if __name__ == "__main__":
    asyncio.run(main())