#!/usr/bin/env python3
"""
TitanovaX Data Storage System - FAISS + Parquet
Memory-efficient storage for embeddings and raw text data
"""

import os
import json
import logging
import hashlib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import faiss
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor
import psutil
import gc

@dataclass
class TelegramMessage:
    """Telegram message structure"""
    message_id: str
    chat_id: str
    user_id: str
    username: str
    text: str
    timestamp: datetime
    message_type: str = "text"  # text, photo, document, etc.
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    hash: str = ""

@dataclass
class MessageSummary:
    """Daily message summary"""
    date: str  # YYYY-MM-DD
    message_count: int
    unique_users: int
    topics: List[str]
    sentiment_score: float
    key_messages: List[str]
    summary_text: str

class FAISSStorageManager:
    """FAISS-based vector storage with memory optimization"""

    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.storage_config = config_manager.storage
        self.logger = logging.getLogger(__name__)

        # Storage paths
        self.base_path = Path('data/storage')
        self.embeddings_path = self.base_path / 'embeddings'
        self.metadata_path = self.base_path / 'metadata'
        self.summaries_path = self.base_path / 'summaries'

        # Create directories
        for path in [self.base_path, self.embeddings_path, self.metadata_path, self.summaries_path]:
            path.mkdir(parents=True, exist_ok=True)

        # FAISS index configuration
        self.dimension = self.storage_config.embeddings_dimension
        self.index_type = self.storage_config.faiss_index_type
        self.nlist = self.storage_config.faiss_nlist  # For IVF indices
        self.m = self.storage_config.faiss_pq_m      # For PQ compression
        self.nbits = self.storage_config.faiss_pq_nbits

        # Memory management
        self.memory_map_enabled = self.storage_config.memory_map_enabled
        self.max_memory_mb = 8192  # 8GB default
        self.lock = threading.Lock()

        # Initialize or load FAISS index
        self.index, self.id_to_message = self._load_or_create_index()

        # Deduplication tracking
        self.message_hashes = set()
        self._load_existing_hashes()

    def _load_or_create_index(self) -> Tuple[faiss.Index, Dict]:
        """Load existing FAISS index or create new one"""
        index_file = self.embeddings_path / 'faiss_index.idx'
        metadata_file = self.metadata_path / 'index_metadata.pkl'

        if index_file.exists() and metadata_file.exists():
            # Load existing index
            self.logger.info("Loading existing FAISS index...")
            index = faiss.read_index(str(index_file))

            with open(metadata_file, 'rb') as f:
                id_to_message = pickle.load(f)

            self.logger.info(f"Loaded index with {index.ntotal} vectors")
            return index, id_to_message
        else:
            # Create new index
            self.logger.info("Creating new FAISS index...")
            if self.index_type == "IndexIVFPQ":
                quantizer = faiss.IndexFlatIP(self.dimension)
                index = faiss.IndexIVFPQ(quantizer, self.dimension, self.nlist, self.m, self.nbits)
            elif self.index_type == "IndexFlatIP":
                index = faiss.IndexFlatIP(self.dimension)
            else:
                raise ValueError(f"Unsupported index type: {self.index_type}")

            # Train index if needed
            if hasattr(index, 'is_trained') and not index.is_trained:
                # Need some training data to train the index
                dummy_data = np.random.random((1000, self.dimension)).astype('float32')
                index.train(dummy_data)

            return index, {}

    def _load_existing_hashes(self):
        """Load existing message hashes for deduplication"""
        hash_file = self.metadata_path / 'message_hashes.pkl'

        if hash_file.exists():
            try:
                with open(hash_file, 'rb') as f:
                    self.message_hashes = pickle.load(f)
                self.logger.info(f"Loaded {len(self.message_hashes)} existing message hashes")
            except Exception as e:
                self.logger.warning(f"Could not load message hashes: {e}")
                self.message_hashes = set()

    def _save_hashes(self):
        """Save message hashes to disk"""
        hash_file = self.metadata_path / 'message_hashes.pkl'

        try:
            with open(hash_file, 'wb') as f:
                pickle.dump(self.message_hashes, f)
        except Exception as e:
            self.logger.error(f"Could not save message hashes: {e}")

    def _check_memory_usage(self) -> bool:
        """Check if memory usage is within limits"""
        memory = psutil.virtual_memory()
        memory_mb = memory.used / 1024 / 1024

        if memory_mb > self.max_memory_mb:
            self.logger.warning(f"Memory usage high: {memory_mb:.1f}MB / {self.max_memory_mb}MB")
            return False

        return True

    def add_message(self, message: TelegramMessage) -> bool:
        """Add a message to the FAISS index"""
        with self.lock:
            # Check memory before processing
            if not self._check_memory_usage():
                self._cleanup_memory()
                if not self._check_memory_usage():
                    self.logger.error("Memory limit exceeded, skipping message")
                    return False

            # Deduplication check
            if self.storage_config.deduplication_enabled:
                if message.hash in self.message_hashes:
                    self.logger.debug(f"Duplicate message skipped: {message.message_id}")
                    return False
                self.message_hashes.add(message.hash)

            # Add to FAISS index
            if message.embedding is not None:
                embedding = message.embedding.reshape(1, -1).astype('float32')

                if not self.index.is_trained:
                    self.index.train(embedding)

                message_id = len(self.id_to_message)
                self.id_to_message[message_id] = message

                # Add to index
                if hasattr(self.index, 'add_with_ids'):
                    self.index.add_with_ids(embedding, np.array([message_id]))
                else:
                    self.index.add(embedding)

                self._save_index()
                self._save_metadata()
                self._save_hashes()

                return True

            return False

    def search_similar(self, query_embedding: np.ndarray, k: int = 10) -> List[Tuple[TelegramMessage, float]]:
        """Search for similar messages"""
        with self.lock:
            if self.index.ntotal == 0:
                return []

            query = query_embedding.reshape(1, -1).astype('float32')
            distances, indices = self.index.search(query, min(k, self.index.ntotal))

            results = []
            for i, idx in enumerate(indices[0]):
                if idx != -1 and idx in self.id_to_message:
                    message = self.id_to_message[idx]
                    distance = distances[0][i]
                    similarity = 1.0 / (1.0 + distance)  # Convert distance to similarity
                    results.append((message, similarity))

            return results

    def _save_index(self):
        """Save FAISS index to disk"""
        try:
            index_file = self.embeddings_path / 'faiss_index.idx'
            faiss.write_index(self.index, str(index_file))
        except Exception as e:
            self.logger.error(f"Could not save FAISS index: {e}")

    def _save_metadata(self):
        """Save metadata to disk"""
        try:
            metadata_file = self.metadata_path / 'index_metadata.pkl'
            with open(metadata_file, 'wb') as f:
                pickle.dump(self.id_to_message, f)
        except Exception as e:
            self.logger.error(f"Could not save metadata: {e}")

    def _cleanup_memory(self):
        """Clean up memory usage"""
        try:
            gc.collect()

            # Clear some caches if they exist
            if hasattr(self, 'message_hashes') and len(self.message_hashes) > 10000:
                # Keep only recent hashes in memory
                all_hashes = list(self.message_hashes)
                self.message_hashes = set(all_hashes[-5000:])

            self.logger.info("Memory cleanup completed")
        except Exception as e:
            self.logger.error(f"Memory cleanup failed: {e}")

class ParquetStorageManager:
    """Parquet-based storage for raw messages and summaries"""

    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.storage_config = config_manager.storage
        self.logger = logging.getLogger(__name__)

        # Storage paths
        self.raw_messages_path = Path('data/storage/messages')
        self.summaries_path = Path('data/storage/summaries')
        self.raw_messages_path.mkdir(parents=True, exist_ok=True)
        self.summaries_path.mkdir(parents=True, exist_ok=True)

        # Retention settings
        self.retention_days_raw = self.storage_config.retention_days_raw
        self.retention_days_summary = self.storage_config.retention_days_summary

    def store_message(self, message: TelegramMessage) -> bool:
        """Store raw message in Parquet format"""
        try:
            # Convert message to DataFrame row
            message_data = {
                'message_id': [message.message_id],
                'chat_id': [message.chat_id],
                'user_id': [message.user_id],
                'username': [message.username],
                'text': [message.text],
                'timestamp': [message.timestamp],
                'message_type': [message.message_type],
                'hash': [message.hash],
                'stored_at': [datetime.now()]
            }

            df = pd.DataFrame(message_data)

            # Append to daily file
            date_str = message.timestamp.strftime('%Y-%m-%d')
            file_path = self.raw_messages_path / f'messages_{date_str}.parquet'

            if file_path.exists():
                existing_df = pd.read_parquet(file_path)
                df = pd.concat([existing_df, df], ignore_index=True)

            # Save with compression
            df.to_parquet(
                file_path,
                compression=self.storage_config.parquet_compression,
                index=False
            )

            return True

        except Exception as e:
            self.logger.error(f"Failed to store message: {e}")
            return False

    def get_messages_by_date(self, date: str, chat_id: str = None) -> pd.DataFrame:
        """Get messages for a specific date"""
        try:
            file_path = self.raw_messages_path / f'messages_{date}.parquet'

            if not file_path.exists():
                return pd.DataFrame()

            df = pd.read_parquet(file_path)

            if chat_id:
                df = df[df['chat_id'] == chat_id]

            return df

        except Exception as e:
            self.logger.error(f"Failed to get messages for date {date}: {e}")
            return pd.DataFrame()

    def get_messages_by_timerange(self, start_date: str, end_date: str, chat_id: str = None) -> pd.DataFrame:
        """Get messages for a date range"""
        try:
            start = datetime.strptime(start_date, '%Y-%m-%d')
            end = datetime.strptime(end_date, '%Y-%m-%d')

            all_messages = []

            current = start
            while current <= end:
                date_str = current.strftime('%Y-%m-%d')
                df = self.get_messages_by_date(date_str, chat_id)
                if not df.empty:
                    all_messages.append(df)
                current += timedelta(days=1)

            if all_messages:
                return pd.concat(all_messages, ignore_index=True)
            else:
                return pd.DataFrame()

        except Exception as e:
            self.logger.error(f"Failed to get messages for range {start_date} to {end_date}: {e}")
            return pd.DataFrame()

    def store_daily_summary(self, summary: MessageSummary) -> bool:
        """Store daily summary"""
        try:
            summary_data = {
                'date': [summary.date],
                'message_count': [summary.message_count],
                'unique_users': [summary.unique_users],
                'topics': [json.dumps(summary.topics)],
                'sentiment_score': [summary.sentiment_score],
                'key_messages': [json.dumps(summary.key_messages)],
                'summary_text': [summary.summary_text],
                'created_at': [datetime.now()]
            }

            df = pd.DataFrame(summary_data)

            # Append to summaries file
            file_path = self.summaries_path / 'daily_summaries.parquet'

            if file_path.exists():
                existing_df = pd.read_parquet(file_path)
                df = pd.concat([existing_df, df], ignore_index=True)

            df.to_parquet(
                file_path,
                compression=self.storage_config.parquet_compression,
                index=False
            )

            return True

        except Exception as e:
            self.logger.error(f"Failed to store daily summary: {e}")
            return False

    def get_daily_summary(self, date: str) -> Optional[MessageSummary]:
        """Get daily summary for a specific date"""
        try:
            file_path = self.summaries_path / 'daily_summaries.parquet'

            if not file_path.exists():
                return None

            df = pd.read_parquet(file_path)
            df = df[df['date'] == date]

            if df.empty:
                return None

            row = df.iloc[0]
            return MessageSummary(
                date=row['date'],
                message_count=row['message_count'],
                unique_users=row['unique_users'],
                topics=json.loads(row['topics']),
                sentiment_score=row['sentiment_score'],
                key_messages=json.loads(row['key_messages']),
                summary_text=row['summary_text']
            )

        except Exception as e:
            self.logger.error(f"Failed to get daily summary for {date}: {e}")
            return None

    def cleanup_old_data(self):
        """Clean up old data according to retention policy"""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.retention_days_raw)

            # Clean up raw messages
            for file_path in self.raw_messages_path.glob('messages_*.parquet'):
                try:
                    # Extract date from filename
                    date_str = file_path.stem.split('_')[1]  # messages_YYYY-MM-DD.parquet
                    file_date = datetime.strptime(date_str, '%Y-%m-%d')

                    if file_date < cutoff_date:
                        file_path.unlink()
                        self.logger.info(f"Removed old message file: {file_path}")

                except Exception as e:
                    self.logger.warning(f"Could not process file {file_path}: {e}")

            # Clean up old summaries (keep longer retention)
            summary_cutoff = datetime.now() - timedelta(days=self.retention_days_summary)

            summary_file = self.summaries_path / 'daily_summaries.parquet'
            if summary_file.exists():
                df = pd.read_parquet(summary_file)

                # Convert date column to datetime
                df['date'] = pd.to_datetime(df['date'])
                df = df[df['date'] >= summary_cutoff]

                if not df.empty:
                    df.to_parquet(summary_file, compression=self.storage_config.parquet_compression, index=False)

            self.logger.info("Data cleanup completed")

        except Exception as e:
            self.logger.error(f"Data cleanup failed: {e}")

class EmbeddingManager:
    """Manage text embeddings with sentence transformers"""

    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.model_name = config_manager.storage.embeddings_model
        self.dimension = config_manager.storage.embeddings_dimension
        self.logger = logging.getLogger(__name__)

        # Lazy load model
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load sentence transformer model"""
        try:
            from sentence_transformers import SentenceTransformer

            self.logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)

            # Move to GPU if available
            if hasattr(self.model, 'to'):
                try:
                    import torch
                    if torch.cuda.is_available():
                        self.model = self.model.to('cuda')
                        self.logger.info("Model moved to GPU")
                except ImportError:
                    pass

        except ImportError:
            self.logger.error("sentence-transformers not installed. Install with: pip install sentence-transformers")
            self.model = None
        except Exception as e:
            self.logger.error(f"Failed to load embedding model: {e}")
            self.model = None

    def embed_text(self, text: str) -> Optional[np.ndarray]:
        """Generate embedding for text"""
        if self.model is None:
            self.logger.error("Embedding model not available")
            return None

        try:
            # Generate embedding
            embedding = self.model.encode(text, convert_to_numpy=True)

            # Quantize to int8 for memory efficiency
            embedding = embedding.astype(np.float32)

            return embedding

        except Exception as e:
            self.logger.error(f"Failed to generate embedding: {e}")
            return None

    def embed_batch(self, texts: List[str]) -> Optional[np.ndarray]:
        """Generate embeddings for multiple texts"""
        if self.model is None:
            self.logger.error("Embedding model not available")
            return None

        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True, batch_size=32)
            return embeddings.astype(np.float32)

        except Exception as e:
            self.logger.error(f"Failed to generate batch embeddings: {e}")
            return None

class TitanovaXStorageSystem:
    """Main storage system combining FAISS and Parquet"""

    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.faiss_manager = FAISSStorageManager(config_manager)
        self.parquet_manager = ParquetStorageManager(config_manager)
        self.embedding_manager = EmbeddingManager(config_manager)

        # Cleanup thread
        self.cleanup_thread = None
        self._start_cleanup_thread()

    def _start_cleanup_thread(self):
        """Start background cleanup thread"""
        def cleanup_worker():
            while True:
                try:
                    # Sleep for 24 hours
                    import time
                    time.sleep(24 * 60 * 60)

                    self.logger.info("Running scheduled data cleanup")
                    self.parquet_manager.cleanup_old_data()

                    # Clean up FAISS memory
                    self.faiss_manager._cleanup_memory()

                except Exception as e:
                    self.logger.error(f"Cleanup thread error: {e}")

        self.cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        self.cleanup_thread.start()

    def process_telegram_message(self, message_data: Dict) -> bool:
        """Process a Telegram message for storage"""
        try:
            # Create message object
            message = TelegramMessage(
                message_id=str(message_data.get('message_id', '')),
                chat_id=str(message_data.get('chat_id', '')),
                user_id=str(message_data.get('user_id', '')),
                username=message_data.get('username', ''),
                text=message_data.get('text', ''),
                timestamp=message_data.get('timestamp', datetime.now()),
                message_type=message_data.get('type', 'text'),
                metadata=message_data.get('metadata', {})
            )

            # Generate hash for deduplication
            message.hash = hashlib.sha256(
                f"{message.chat_id}:{message.message_id}:{message.text}".encode()
            ).hexdigest()

            # Generate embedding
            if message.text:
                message.embedding = self.embedding_manager.embed_text(message.text)

            # Store in Parquet
            parquet_success = self.parquet_manager.store_message(message)

            # Add to FAISS index
            faiss_success = False
            if message.embedding is not None:
                faiss_success = self.faiss_manager.add_message(message)

            return parquet_success or faiss_success

        except Exception as e:
            self.logger.error(f"Failed to process message: {e}")
            return False

    def search_messages(self, query: str, k: int = 10) -> List[TelegramMessage]:
        """Search for similar messages"""
        try:
            # Generate embedding for query
            query_embedding = self.embedding_manager.embed_text(query)

            if query_embedding is None:
                return []

            # Search FAISS index
            results = self.faiss_manager.search_similar(query_embedding, k)

            # Extract messages
            messages = [message for message, similarity in results]

            return messages

        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return []

    def get_messages_by_date(self, date: str, chat_id: str = None) -> pd.DataFrame:
        """Get messages for a specific date"""
        return self.parquet_manager.get_messages_by_date(date, chat_id)

    def get_daily_summary(self, date: str) -> Optional[MessageSummary]:
        """Get daily summary"""
        return self.parquet_manager.get_daily_summary(date)

    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage system statistics"""
        try:
            # FAISS stats
            faiss_stats = {
                'index_type': self.faiss_manager.index_type,
                'total_vectors': self.faiss_manager.index.ntotal,
                'dimension': self.faiss_manager.dimension,
                'is_trained': getattr(self.faiss_manager.index, 'is_trained', True)
            }

            # Parquet stats
            total_messages = 0
            total_size = 0

            for file_path in self.parquet_manager.raw_messages_path.glob('*.parquet'):
                total_messages += len(pd.read_parquet(file_path, columns=['message_id']))
                total_size += file_path.stat().st_size

            parquet_stats = {
                'total_messages': total_messages,
                'total_files': len(list(self.parquet_manager.raw_messages_path.glob('*.parquet'))),
                'total_size_mb': total_size / 1024 / 1024
            }

            # Memory stats
            memory = psutil.virtual_memory()
            memory_stats = {
                'used_mb': memory.used / 1024 / 1024,
                'available_mb': memory.available / 1024 / 1024,
                'percent_used': memory.percent
            }

            return {
                'faiss': faiss_stats,
                'parquet': parquet_stats,
                'memory': memory_stats,
                'config': {
                    'retention_days_raw': self.parquet_manager.retention_days_raw,
                    'retention_days_summary': self.parquet_manager.retention_days_summary,
                    'deduplication_enabled': self.faiss_manager.storage_config.deduplication_enabled
                }
            }

        except Exception as e:
            self.logger.error(f"Failed to get storage stats: {e}")
            return {}

if __name__ == "__main__":
    # Demo usage
    from config_manager import get_config_manager

    try:
        config = get_config_manager()
        storage = TitanovaXStorageSystem(config)

        # Process a sample message
        message_data = {
            'message_id': '12345',
            'chat_id': '-1001234567890',
            'user_id': '987654321',
            'username': 'testuser',
            'text': 'Hello, this is a test message for the TitanovaX storage system!',
            'timestamp': datetime.now(),
            'type': 'text'
        }

        success = storage.process_telegram_message(message_data)
        print(f"Message processed: {success}")

        # Search for similar messages
        results = storage.search_messages("Hello world", k=5)
        print(f"Found {len(results)} similar messages")

        # Get storage stats
        stats = storage.get_storage_stats()
        print(f"Storage stats: {json.dumps(stats, indent=2, default=str)}")

    except Exception as e:
        print(f"Demo failed: {e}")
