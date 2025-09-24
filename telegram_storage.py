"""
TitanovaX Trading System - FAISS + Parquet Storage Engine
Memory-efficient storage for Telegram logs with embeddings compression
"""

import os
import json
import hashlib
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import gc
from concurrent.futures import ThreadPoolExecutor
import threading
import time

# FAISS imports
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("Warning: FAISS not available. Install with: pip install faiss-cpu")

# Sentence transformers
try:
    from sentence_transformers import SentenceTransformer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: Sentence Transformers not available. Install with: pip install sentence-transformers")

# Compression
try:
    import pyarrow.parquet as pq
    import pyarrow as pa
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False
    print("Warning: PyArrow not available. Install with: pip install pyarrow")


@dataclass
class TelegramMessage:
    """Telegram message data structure"""
    message_id: int
    chat_id: int
    user_id: Optional[int]
    username: Optional[str]
    text: str
    timestamp: datetime
    message_type: str  # 'text', 'photo', 'document', etc.
    reply_to_message_id: Optional[int] = None
    forward_from: Optional[str] = None
    entities: Optional[List[Dict]] = None
    hash_signature: Optional[str] = None
    embedding: Optional[np.ndarray] = None
    compressed: bool = False


@dataclass
class DailySummary:
    """Daily summary for old messages"""
    date: datetime
    message_count: int
    unique_users: int
    top_keywords: List[str]
    sentiment_summary: str
    trading_signals: List[str]
    summary_text: str
    embedding: Optional[np.ndarray] = None


class FAISSStorageEngine:
    """Memory-efficient FAISS storage with compression"""
    
    def __init__(self, config_manager, index_path: str = "storage/faiss_index", 
                 mmap_path: str = "storage/faiss_mmap"):
        self.config = config_manager
        self.logger = logging.getLogger(__name__)
        self.index_path = Path(index_path)
        self.mmap_path = Path(mmap_path)
        self.embedding_model = None
        self.faiss_index = None
        self.index_lock = threading.RLock()
        
        # Create directories
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.mmap_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._initialize_embedding_model()
        self._initialize_faiss_index()
        
    def _initialize_embedding_model(self):
        """Initialize sentence transformer model"""
        if not TRANSFORMERS_AVAILABLE:
            self.logger.error("Sentence Transformers not available")
            return
            
        try:
            model_name = self.config.storage.embeddings_model
            self.embedding_model = SentenceTransformer(model_name)
            self.logger.info(f"Embedding model loaded: {model_name}")
        except Exception as e:
            self.logger.error(f"Failed to load embedding model: {e}")
            
    def _initialize_faiss_index(self):
        """Initialize FAISS index with compression"""
        if not FAISS_AVAILABLE:
            self.logger.error("FAISS not available")
            return
            
        try:
            dimension = self.config.storage.embeddings_dimension
            index_type = self.config.storage.faiss_index_type
            
            if index_type == "IndexIVFPQ":
                # Create IVF-PQ index for memory efficiency
                nlist = self.config.storage.faiss_nlist
                m = self.config.storage.faiss_pq_m
                nbits = self.config.storage.faiss_pq_nbits
                
                quantizer = faiss.IndexFlatL2(dimension)
                self.faiss_index = faiss.IndexIVFPQ(quantizer, dimension, nlist, m, nbits)
                
                # Enable on-disk storage if configured
                if self.config.storage.use_disk_storage:
                    faiss.write_index(self.faiss_index, str(self.mmap_path / "index_mmap"))
                    self.faiss_index = faiss.read_index(str(self.mmap_path / "index_mmap"), 
                                                       faiss.IO_FLAG_MMAP)
                
            elif index_type == "IndexFlatIP":
                self.faiss_index = faiss.IndexFlatIP(dimension)
            else:
                # Default to L2 index
                self.faiss_index = faiss.IndexFlatL2(dimension)
            
            self.logger.info(f"FAISS index initialized: {index_type}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize FAISS index: {e}")
    
    def generate_embedding(self, text: str) -> Optional[np.ndarray]:
        """Generate embedding for text"""
        if not self.embedding_model:
            self.logger.error("Embedding model not available")
            return None
            
        try:
            # Preprocess text
            text = self._preprocess_text(text)
            
            # Generate embedding
            embedding = self.embedding_model.encode([text], convert_to_numpy=True)[0]
            
            # Quantize to int8 if configured
            if self.config.storage.embeddings_dimension == 384:
                embedding = self._quantize_embedding(embedding)
            
            return embedding.astype(np.float32)
            
        except Exception as e:
            self.logger.error(f"Failed to generate embedding: {e}")
            return None
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for embedding"""
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Limit length to prevent memory issues
        max_length = 512
        if len(text) > max_length:
            text = text[:max_length]
        
        return text.strip()
    
    def _quantize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Quantize embedding to reduce memory usage"""
        # Simple quantization to int8
        quantized = (embedding * 128).astype(np.int8)
        return quantized.astype(np.float32) / 128
    
    def add_message(self, message: TelegramMessage) -> bool:
        """Add message to FAISS index"""
        if not self.faiss_index or not message.embedding:
            return False
            
        with self.index_lock:
            try:
                # Add to FAISS index
                embedding = message.embedding.reshape(1, -1)
                self.faiss_index.add(embedding)
                
                # Save index periodically
                if self.faiss_index.ntotal % 1000 == 0:
                    self._save_index()
                
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to add message to index: {e}")
                return False
    
    def search_similar_messages(self, query_text: str, k: int = 5) -> List[Tuple[TelegramMessage, float]]:
        """Search for similar messages"""
        if not self.faiss_index or not self.embedding_model:
            return []
            
        try:
            # Generate query embedding
            query_embedding = self.generate_embedding(query_text)
            if query_embedding is None:
                return []
            
            # Search in FAISS index
            query_embedding = query_embedding.reshape(1, -1)
            distances, indices = self.faiss_index.search(query_embedding, k)
            
            # Return results (this would need message storage integration)
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx != -1:  # Valid index
                    # In a real implementation, you would retrieve the message from storage
                    similarity = 1.0 / (1.0 + distance)  # Convert distance to similarity
                    results.append((None, similarity))  # Placeholder
            
            return results
            
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return []
    
    def _save_index(self):
        """Save FAISS index to disk"""
        if not self.faiss_index:
            return
            
        try:
            faiss.write_index(self.faiss_index, str(self.index_path))
            self.logger.info(f"FAISS index saved: {self.faiss_index.ntotal} vectors")
        except Exception as e:
            self.logger.error(f"Failed to save FAISS index: {e}")


class ParquetStorageEngine:
    """Parquet storage for raw messages and summaries"""
    
    def __init__(self, config_manager, storage_path: str = "storage/parquet"):
        self.config = config_manager
        self.logger = logging.getLogger(__name__)
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.write_lock = threading.Lock()
        
    def store_message(self, message: TelegramMessage) -> bool:
        """Store single message in Parquet format"""
        try:
            # Generate hash for deduplication
            message.hash_signature = self._generate_message_hash(message)
            
            # Convert to DataFrame
            df = self._message_to_dataframe(message)
            
            # Store in daily Parquet file
            date_str = message.timestamp.strftime("%Y-%m-%d")
            parquet_file = self.storage_path / f"messages_{date_str}.parquet"
            
            with self.write_lock:
                if parquet_file.exists():
                    # Append to existing file
                    existing_df = pd.read_parquet(parquet_file)
                    
                    # Check for duplicates
                    if message.hash_signature in existing_df['hash_signature'].values:
                        self.logger.debug(f"Duplicate message skipped: {message.hash_signature}")
                        return False
                    
                    # Append new message
                    combined_df = pd.concat([existing_df, df], ignore_index=True)
                    combined_df.to_parquet(parquet_file, 
                                         compression=self.config.storage.parquet_compression,
                                         engine='pyarrow')
                else:
                    # Create new file
                    df.to_parquet(parquet_file, 
                                compression=self.config.storage.parquet_compression,
                                engine='pyarrow')
            
            self.logger.debug(f"Message stored: {message.message_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store message: {e}")
            return False
    
    def _generate_message_hash(self, message: TelegramMessage) -> str:
        """Generate SHA256 hash for message deduplication"""
        content = f"{message.chat_id}:{message.message_id}:{message.timestamp.isoformat()}:{message.text}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _message_to_dataframe(self, message: TelegramMessage) -> pd.DataFrame:
        """Convert TelegramMessage to DataFrame"""
        data = {
            'message_id': [message.message_id],
            'chat_id': [message.chat_id],
            'user_id': [message.user_id],
            'username': [message.username],
            'text': [message.text],
            'timestamp': [message.timestamp],
            'message_type': [message.message_type],
            'reply_to_message_id': [message.reply_to_message_id],
            'forward_from': [message.forward_from],
            'entities': [json.dumps(message.entities) if message.entities else None],
            'hash_signature': [message.hash_signature],
            'compressed': [message.compressed]
        }
        
        # Handle embedding storage
        if message.embedding is not None:
            data['embedding'] = [message.embedding.tobytes()]
            data['embedding_shape'] = [message.embedding.shape]
        else:
            data['embedding'] = [None]
            data['embedding_shape'] = [None]
        
        return pd.DataFrame(data)
    
    def store_daily_summary(self, summary: DailySummary) -> bool:
        """Store daily summary"""
        try:
            df = pd.DataFrame([{
                'date': summary.date,
                'message_count': summary.message_count,
                'unique_users': summary.unique_users,
                'top_keywords': json.dumps(summary.top_keywords),
                'sentiment_summary': summary.sentiment_summary,
                'trading_signals': json.dumps(summary.trading_signals),
                'summary_text': summary.summary_text
            }])
            
            summary_file = self.storage_path / f"daily_summaries.parquet"
            
            with self.write_lock:
                if summary_file.exists():
                    existing_df = pd.read_parquet(summary_file)
                    combined_df = pd.concat([existing_df, df], ignore_index=True)
                    combined_df.to_parquet(summary_file, 
                                         compression=self.config.storage.parquet_compression)
                else:
                    df.to_parquet(summary_file, 
                                compression=self.config.storage.parquet_compression)
            
            self.logger.info(f"Daily summary stored for {summary.date}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store daily summary: {e}")
            return False
    
    def cleanup_old_data(self):
        """Clean up old data based on retention policy"""
        try:
            current_time = datetime.now()
            
            # Clean up raw messages
            raw_retention = timedelta(days=self.config.storage.retention_days_raw)
            summary_retention = timedelta(days=self.config.storage.retention_days_summary)
            
            for parquet_file in self.storage_path.glob("messages_*.parquet"):
                # Extract date from filename
                date_str = parquet_file.stem.replace("messages_", "")
                file_date = datetime.strptime(date_str, "%Y-%m-%d")
                
                if current_time - file_date > raw_retention:
                    if current_time - file_date <= summary_retention:
                        # Generate summary before deletion
                        self._generate_summary_from_file(parquet_file)
                    
                    # Delete old file
                    parquet_file.unlink()
                    self.logger.info(f"Deleted old messages file: {parquet_file}")
            
            self.logger.info("Data cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Data cleanup failed: {e}")
    
    def _generate_summary_from_file(self, parquet_file: Path):
        """Generate summary from old message file before deletion"""
        try:
            df = pd.read_parquet(parquet_file)
            
            if len(df) == 0:
                return
            
            # Extract date from filename
            date_str = parquet_file.stem.replace("messages_", "")
            file_date = datetime.strptime(date_str, "%Y-%m-%d")
            
            # Generate simple summary
            summary = DailySummary(
                date=file_date,
                message_count=len(df),
                unique_users=df['user_id'].nunique(),
                top_keywords=[],  # Would need NLP processing
                sentiment_summary="neutral",  # Would need sentiment analysis
                trading_signals=[],  # Would need signal extraction
                summary_text=f"Summary for {date_str}: {len(df)} messages from {df['user_id'].nunique()} users"
            )
            
            self.store_daily_summary(summary)
            
        except Exception as e:
            self.logger.error(f"Failed to generate summary: {e}")


class TelegramStorageManager:
    """Main storage manager for Telegram logs"""
    
    def __init__(self, config_manager):
        self.config = config_manager
        self.logger = logging.getLogger(__name__)
        
        # Initialize storage engines
        self.faiss_engine = FAISSStorageEngine(config_manager)
        self.parquet_engine = ParquetStorageEngine(config_manager)
        
        # Start cleanup scheduler
        self._start_cleanup_scheduler()
        
    def store_telegram_message(self, message_data: Dict[str, Any]) -> bool:
        """Store incoming Telegram message"""
        try:
            # Create TelegramMessage object
            message = self._parse_telegram_message(message_data)
            
            # Generate embedding
            if self.config.storage.deduplication_enabled:
                embedding = self.faiss_engine.generate_embedding(message.text)
                message.embedding = embedding
            
            # Store in Parquet
            stored = self.parquet_engine.store_message(message)
            
            # Store embedding in FAISS if available
            if stored and message.embedding is not None:
                self.faiss_engine.add_message(message)
            
            self.logger.debug(f"Telegram message stored: {message.message_id}")
            return stored
            
        except Exception as e:
            self.logger.error(f"Failed to store Telegram message: {e}")
            return False
    
    def _parse_telegram_message(self, message_data: Dict[str, Any]) -> TelegramMessage:
        """Parse Telegram API message data"""
        message = message_data.get('message', message_data)
        
        return TelegramMessage(
            message_id=message.get('message_id', 0),
            chat_id=message['chat']['id'],
            user_id=message.get('from', {}).get('id') if message.get('from') else None,
            username=message.get('from', {}).get('username') if message.get('from') else None,
            text=message.get('text', ''),
            timestamp=datetime.fromtimestamp(message.get('date', time.time())),
            message_type='text' if 'text' in message else 'other',
            reply_to_message_id=message.get('reply_to_message', {}).get('message_id') if message.get('reply_to_message') else None,
            forward_from=message.get('forward_from', {}).get('username') if message.get('forward_from') else None,
            entities=message.get('entities', [])
        )
    
    def search_messages(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Search for similar messages"""
        results = self.faiss_engine.search_similar_messages(query, k)
        
        # Convert results to dict format
        search_results = []
        for message, similarity in results:
            if message:
                search_results.append({
                    'message': asdict(message),
                    'similarity': similarity
                })
        
        return search_results
    
    def _start_cleanup_scheduler(self):
        """Start background cleanup scheduler"""
        def cleanup_task():
            while True:
                try:
                    time.sleep(86400)  # Run daily
                    self.parquet_engine.cleanup_old_data()
                except Exception as e:
                    self.logger.error(f"Cleanup task error: {e}")
        
        cleanup_thread = threading.Thread(target=cleanup_task, daemon=True)
        cleanup_thread.start()
        self.logger.info("Cleanup scheduler started")
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        stats = {
            'faiss_index_size': self.faiss_engine.faiss_index.ntotal if self.faiss_engine.faiss_index else 0,
            'parquet_files': len(list(self.parquet_engine.storage_path.glob("*.parquet"))),
            'storage_path_size_mb': sum(f.stat().st_size for f in self.parquet_engine.storage_path.rglob("*") if f.is_file()) / (1024 * 1024)
        }
        
        return stats


# Usage example and testing
if __name__ == "__main__":
    # Test the storage system
    try:
        from config_manager import initialize_config
        
        # Initialize configuration
        config = initialize_config()
        
        # Create storage manager
        storage = TelegramStorageManager(config)
        
        # Test message storage
        test_message = {
            'message_id': 12345,
            'chat': {'id': -1002302007470},
            'from': {'id': 6389423283, 'username': 'test_user'},
            'text': 'Test trading signal: BUY EUR/USD at 1.0850',
            'date': int(time.time()),
            'entities': []
        }
        
        success = storage.store_telegram_message(test_message)
        print(f"Message storage test: {'SUCCESS' if success else 'FAILED'}")
        
        # Test search
        results = storage.search_messages("trading signal EUR/USD")
        print(f"Search results: {len(results)} matches")
        
        # Get stats
        stats = storage.get_storage_stats()
        print(f"Storage stats: {json.dumps(stats, indent=2)}")
        
    except Exception as e:
        print(f"Storage system test failed: {e}")
        import traceback
        traceback.print_exc()