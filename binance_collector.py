#!/usr/bin/env python3
"""
Binance WebSocket Data Acquisition for TitanovaX Trading System
Collects live tick data from Binance and saves to Parquet format
"""

import asyncio
import json
import websockets
import pandas as pd
import datetime
import os
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/logs/binance_collector.log'),
        logging.StreamHandler()
    ]
)

class BinanceDataCollector:
    def __init__(self, symbols=['btcusdt', 'ethusdt', 'bnbusdt'], data_dir='data/raw/crypto'):
        self.symbols = symbols
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.running = False
        self.connections = {}
        self.data_buffers = {symbol: [] for symbol in symbols}

    async def collect_symbol_data(self, symbol):
        """Collect data for a single symbol"""
        uri = f"wss://stream.binance.com:9443/ws/{symbol}@trade"

        try:
            async with websockets.connect(uri) as websocket:
                logging.info(f"Connected to Binance WebSocket for {symbol}")

                while self.running:
                    try:
                        message = await websocket.recv()
                        data = json.loads(message)

                        # Parse trade data
                        trade_data = {
                            'timestamp': pd.to_datetime(data['T'], unit='ms'),
                            'symbol': data['s'].lower(),
                            'price': float(data['p']),
                            'quantity': float(data['q']),
                            'trade_id': data['t'],
                            'is_buyer_maker': data['m'],
                            'best_match': data['M']
                        }

                        self.data_buffers[symbol].append(trade_data)

                        # Write to file every 1000 trades
                        if len(self.data_buffers[symbol]) >= 1000:
                            self.flush_buffer(symbol)

                        # Log progress every 100 trades
                        if len(self.data_buffers[symbol]) % 100 == 0:
                            logging.info(f"{symbol}: Collected {len(self.data_buffers[symbol])} trades")

                    except Exception as e:
                        logging.error(f"Error processing message for {symbol}: {e}")
                        await asyncio.sleep(1)

        except Exception as e:
            logging.error(f"Connection error for {symbol}: {e}")

    def flush_buffer(self, symbol):
        """Write buffered data to Parquet file"""
        if not self.data_buffers[symbol]:
            return

        try:
            df = pd.DataFrame(self.data_buffers[symbol])

            # Create filename with date
            date_str = df['timestamp'].iloc[0].strftime('%Y%m%d')
            filename = self.data_dir / f"{symbol}_trades_{date_str}.parquet"

            # Append to existing file or create new one
            if filename.exists():
                existing_df = pd.read_parquet(filename)
                df = pd.concat([existing_df, df], ignore_index=True)

            df.to_parquet(filename, index=False)
            logging.info(f"Flushed {len(self.data_buffers[symbol])} trades for {symbol} to {filename}")

            self.data_buffers[symbol] = []

        except Exception as e:
            logging.error(f"Error flushing buffer for {symbol}: {e}")

    async def start_collection(self):
        """Start collecting data for all symbols"""
        self.running = True
        logging.info("Starting Binance data collection...")

        tasks = []
        for symbol in self.symbols:
            task = asyncio.create_task(self.collect_symbol_data(symbol))
            tasks.append(task)

        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            logging.info("Stopping data collection...")
            self.running = False
            # Flush remaining data
            for symbol in self.symbols:
                self.flush_buffer(symbol)

    def get_collection_stats(self):
        """Get statistics about data collection"""
        stats = {}
        for symbol in self.symbols:
            symbol_path = self.data_dir / f"{symbol}_trades_*.parquet"
            files = list(symbol_path.parent.glob(symbol_path.name.replace('*', '*')))
            total_trades = 0

            for file in files:
                try:
                    df = pd.read_parquet(file)
                    total_trades += len(df)
                except Exception as e:
                    logging.error(f"Error reading {file}: {e}")

            stats[symbol] = {
                'files': len(files),
                'total_trades': total_trades,
                'latest_file': max(files, key=lambda x: x.stat().st_mtime) if files else None
            }

        return stats

# Historical data fetcher for backfilling
class BinanceHistoricalData:
    def __init__(self, api_key=None, api_secret=None):
        self.api_key = api_key
        self.api_secret = api_secret

    async def fetch_historical_klines(self, symbol, interval='1m', start_str='1 day ago UTC'):
        """Fetch historical klines data"""
        try:
            import requests

            base_url = 'https://api.binance.com/api/v3/klines'

            params = {
                'symbol': symbol.upper(),
                'interval': interval,
                'startTime': int(pd.to_datetime(start_str).timestamp() * 1000),
                'limit': 1000
            }

            response = requests.get(base_url, params=params)
            data = response.json()

            if isinstance(data, list) and len(data) > 0:
                df = pd.DataFrame(data, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_asset_volume', 'number_of_trades',
                    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                ])

                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')

                # Convert numeric columns
                numeric_cols = ['open', 'high', 'low', 'close', 'volume']
                df[numeric_cols] = df[numeric_cols].astype(float)

                return df
            else:
                logging.error(f"No historical data received for {symbol}")
                return None

        except Exception as e:
            logging.error(f"Error fetching historical data for {symbol}: {e}")
            return None

    async def fetch_multiple_symbols(self, symbols, interval='1m', start_str='30 days ago UTC'):
        """Fetch historical data for multiple symbols"""
        tasks = []
        for symbol in symbols:
            task = self.fetch_historical_klines(symbol, interval, start_str)
            tasks.append((symbol, task))

        results = {}
        for symbol, task in tasks:
            results[symbol] = await task

        return results

if __name__ == "__main__":
    # Example usage
    collector = BinanceDataCollector(
        symbols=['btcusdt', 'ethusdt', 'bnbusdt', 'adausdt', 'solusdt'],
        data_dir='data/raw/crypto'
    )

    try:
        asyncio.run(collector.start_collection())
    except KeyboardInterrupt:
        print("\nData collection stopped by user")
    finally:
        # Show final statistics
        stats = collector.get_collection_stats()
        print("\n=== Collection Statistics ===")
        for symbol, stat in stats.items():
            print(f"{symbol}: {stat['total_trades']} trades in {stat['files']} files")
