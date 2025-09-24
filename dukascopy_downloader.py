#!/usr/bin/env python3
"""
Dukascopy Historical Data Downloader for TitanovaX Trading System
Downloads historical tick data from Dukascopy and converts to Parquet format
"""

import requests
import pandas as pd
import numpy as np
from pathlib import Path
import zipfile
import io
import time
from datetime import datetime, timedelta
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/logs/dukascopy_downloader.log'),
        logging.StreamHandler()
    ]
)

class DukascopyDownloader:
    def __init__(self, data_dir='data/raw/fx'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.base_url = "https://datafeed.dukascopy.com/datafeed"
        self.symbols = {
            'EURUSD': 'EURUSD',
            'GBPUSD': 'GBPUSD',
            'USDJPY': 'USDJPY',
            'USDCHF': 'USDCHF',
            'AUDUSD': 'AUDUSD',
            'USDCAD': 'USDCAD',
            'NZDUSD': 'NZDUSD',
            'XAUUSD': 'XAUUSD',  # Gold
            'BTCUSD': 'BTCUSD',  # Bitcoin
            'ETHUSD': 'ETHUSD'   # Ethereum
        }

    def download_tick_data(self, symbol, year, month):
        """Download tick data for a specific symbol, year, and month"""

        # Create directory structure: data/raw/fx/{symbol}/{year}
        symbol_dir = self.data_dir / symbol
        year_dir = symbol_dir / str(year)
        year_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{symbol}_{year}_{month:02d}.parquet"
        filepath = year_dir / filename

        # Skip if file already exists
        if filepath.exists():
            logging.info(f"File already exists: {filepath}")
            return filepath

        try:
            # Dukascopy tick data URL format
            url = f"{self.base_url}/{symbol.lower()}/{year}/{month:02d}/{symbol.lower()}_{year}_{month:02d}.csv.gz"

            logging.info(f"Downloading: {url}")
            response = requests.get(url, timeout=30)

            if response.status_code == 200:
                # Read the compressed CSV
                with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
                    csv_filename = f"{symbol.lower()}_{year}_{month:02d}.csv"
                    with zf.open(csv_filename) as f:
                        df = pd.read_csv(f, header=None)

                        # Dukascopy CSV format: timestamp,ask,bid,ask_volume,bid_volume
                        df.columns = ['timestamp', 'ask', 'bid', 'ask_volume', 'bid_volume']

                        # Convert timestamp to datetime
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

                        # Calculate spread and mid price
                        df['mid_price'] = (df['ask'] + df['bid']) / 2
                        df['spread'] = df['ask'] - df['bid']

                        # Add symbol column
                        df['symbol'] = symbol

                        # Save to Parquet
                        df.to_parquet(filepath, index=False)
                        logging.info(f"Downloaded {len(df)} ticks to {filepath}")
                        return filepath

            else:
                logging.warning(f"Failed to download {url}: HTTP {response.status_code}")
                return None

        except Exception as e:
            logging.error(f"Error downloading {symbol} {year}-{month}: {e}")
            return None

    def download_year_range(self, symbol, start_year=2000, end_year=None):
        """Download all months for a symbol within year range"""

        if end_year is None:
            end_year = datetime.now().year

        successful_downloads = []
        failed_downloads = []

        for year in range(start_year, end_year + 1):
            logging.info(f"Downloading {symbol} for year {year}")

            for month in range(1, 13):
                filepath = self.download_tick_data(symbol, year, month)

                if filepath:
                    successful_downloads.append(filepath)
                else:
                    failed_downloads.append(f"{symbol}_{year}_{month}")

                # Small delay to be respectful to the server
                time.sleep(0.5)

        return successful_downloads, failed_downloads

    def download_all_symbols(self, start_year=2000, end_year=None, max_workers=4):
        """Download data for all symbols in parallel"""

        if end_year is None:
            end_year = datetime.now().year

        all_successful = []
        all_failed = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all download tasks
            future_to_symbol = {}

            for symbol in self.symbols.keys():
                logging.info(f"Starting download for {symbol}")
                future = executor.submit(self.download_year_range, symbol, start_year, end_year)
                future_to_symbol[future] = symbol

            # Process results as they complete
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    successful, failed = future.result()
                    all_successful.extend(successful)
                    all_failed.extend(failed)

                    logging.info(f"Completed {symbol}: {len(successful)} successful, {len(failed)} failed")

                except Exception as e:
                    logging.error(f"Error processing {symbol}: {e}")
                    all_failed.append(symbol)

        return all_successful, all_failed

    def validate_download(self, symbol, year, month):
        """Validate downloaded data quality"""

        filepath = self.data_dir / symbol / str(year) / f"{symbol}_{year}_{month:02d}.parquet"

        if not filepath.exists():
            return False, "File does not exist"

        try:
            df = pd.read_parquet(filepath)

            # Basic validation checks
            if len(df) == 0:
                return False, "Empty file"

            if df['timestamp'].is_monotonic_increasing == False:
                return False, "Timestamps not in order"

            if df['ask'].min() <= 0 or df['bid'].min() <= 0:
                return False, "Invalid prices found"

            if df['spread'].min() < 0:
                return False, "Negative spread found"

            # Check for reasonable data density
            expected_ticks = 24 * 60 * 60  # One tick per second (theoretical max)
            actual_ticks = len(df)
            density_ratio = actual_ticks / expected_ticks

            if density_ratio < 0.1:  # Less than 10% density
                return False, f"Low data density: {density_ratio:.2%}"

            return True, f"Valid: {len(df)} ticks, density: {density_ratio:.2%}"

        except Exception as e:
            return False, f"Validation error: {e}"

    def create_data_summary(self):
        """Create a summary of all downloaded data"""

        summary = {
            'total_symbols': 0,
            'total_files': 0,
            'total_ticks': 0,
            'date_range': {'start': None, 'end': None},
            'symbols': {}
        }

        for symbol in self.symbols.keys():
            symbol_dir = self.data_dir / symbol
            if not symbol_dir.exists():
                continue

            symbol_info = {
                'files': 0,
                'total_ticks': 0,
                'years': [],
                'date_range': {'start': None, 'end': None}
            }

            for year_dir in symbol_dir.iterdir():
                if year_dir.is_dir():
                    year = int(year_dir.name)
                    symbol_info['years'].append(year)

                    for file in year_dir.glob('*.parquet'):
                        symbol_info['files'] += 1
                        summary['total_files'] += 1

                        try:
                            df = pd.read_parquet(file)
                            symbol_info['total_ticks'] += len(df)

                            # Update date ranges
                            if symbol_info['date_range']['start'] is None:
                                symbol_info['date_range']['start'] = df['timestamp'].min()
                            else:
                                symbol_info['date_range']['start'] = min(
                                    symbol_info['date_range']['start'], df['timestamp'].min()
                                )

                            if symbol_info['date_range']['end'] is None:
                                symbol_info['date_range']['end'] = df['timestamp'].max()
                            else:
                                symbol_info['date_range']['end'] = max(
                                    symbol_info['date_range']['end'], df['timestamp'].max()
                                )

                        except Exception as e:
                            logging.error(f"Error reading {file}: {e}")

            if symbol_info['total_ticks'] > 0:
                summary['symbols'][symbol] = symbol_info
                summary['total_ticks'] += symbol_info['total_ticks']
                summary['total_symbols'] += 1

                # Update global date range
                if summary['date_range']['start'] is None:
                    summary['date_range']['start'] = symbol_info['date_range']['start']
                else:
                    summary['date_range']['start'] = min(
                        summary['date_range']['start'], symbol_info['date_range']['start']
                    )

                if summary['date_range']['end'] is None:
                    summary['date_range']['end'] = symbol_info['date_range']['end']
                else:
                    summary['date_range']['end'] = max(
                        summary['date_range']['end'], symbol_info['date_range']['end']
                    )

        # Save summary to JSON
        summary_file = self.data_dir / 'data_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        return summary

def main():
    """Main function to download FX data"""

    downloader = DukascopyDownloader()

    # Download data for all symbols from 2000 to current year
    logging.info("Starting Dukascopy data download...")
    successful, failed = downloader.download_all_symbols(start_year=2020, end_year=2025, max_workers=2)

    print(f"\n=== Download Summary ===")
    print(f"Successful downloads: {len(successful)}")
    print(f"Failed downloads: {len(failed)}")

    if failed:
        print(f"Failed: {failed}")

    # Create summary
    print("\nCreating data summary...")
    summary = downloader.create_data_summary()

    print(f"\n=== Data Summary ===")
    print(f"Total symbols: {summary['total_symbols']}")
    print(f"Total files: {summary['total_files']}")
    print(f"Total ticks: {summary['total_ticks']:,}")

    if summary['date_range']['start'] and summary['date_range']['end']:
        print(f"Date range: {summary['date_range']['start']} to {summary['date_range']['end']}")

    # Show per-symbol breakdown
    for symbol, info in summary['symbols'].items():
        years_str = ', '.join(map(str, sorted(info['years'])))
        print(f"{symbol}: {info['total_ticks']:,} ticks, {info['files']} files, years: {years_str}")

if __name__ == "__main__":
    main()
