"""
TitanovaX Trading System - OANDA REST API Integration
Enterprise-grade trading interface with comprehensive error handling and monitoring
"""

import json
import logging
import time
import asyncio
import aiohttp
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from decimal import Decimal
import hmac
import hashlib
import base64
from urllib.parse import urlencode
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import threading


@dataclass
class OANDAOrder:
    """OANDA order data structure"""
    instrument: str
    units: Decimal
    side: str  # 'buy' or 'sell'
    type: str  # 'market', 'limit', 'stop', 'marketIfTouched'
    price: Optional[Decimal] = None
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    trailing_stop_loss: Optional[Dict] = None
    time_in_force: str = 'FOK'  # 'FOK', 'IOC', 'GTD'
    position_fill: str = 'DEFAULT'  # 'DEFAULT', 'REDUCE_ONLY', 'OPEN_ONLY'
    client_extensions: Optional[Dict] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to OANDA API format"""
        order_dict = {
            'instrument': self.instrument,
            'units': str(abs(self.units)),
            'side': self.side.upper(),
            'type': self.type.upper(),
            'timeInForce': self.time_in_force,
            'positionFill': self.position_fill
        }
        
        if self.price:
            order_dict['price'] = str(self.price)
        
        if self.stop_loss:
            order_dict['stopLossOnFill'] = {
                'price': str(self.stop_loss),
                'timeInForce': 'GTC'
            }
        
        if self.take_profit:
            order_dict['takeProfitOnFill'] = {
                'price': str(self.take_profit),
                'timeInForce': 'GTC'
            }
        
        if self.trailing_stop_loss:
            order_dict['trailingStopLossOnFill'] = self.trailing_stop_loss
        
        if self.client_extensions:
            order_dict['clientExtensions'] = self.client_extensions
            
        return order_dict


@dataclass
class OANDAPosition:
    """OANDA position data structure"""
    instrument: str
    long_units: Decimal
    short_units: Decimal
    long_avg_price: Optional[Decimal] = None
    short_avg_price: Optional[Decimal] = None
    pl: Decimal = Decimal('0')
    unrealized_pl: Decimal = Decimal('0')
    margin_used: Decimal = Decimal('0')
    
    @property
    def net_units(self) -> Decimal:
        return self.long_units - self.short_units


@dataclass
class OANDAAccount:
    """OANDA account information"""
    id: str
    alias: str
    currency: str
    balance: Decimal
    pl: Decimal
    unrealized_pl: Decimal
    margin_used: Decimal
    margin_available: Decimal
    margin_closeout_percent: Decimal
    open_trade_count: int
    open_position_count: int
    pending_order_count: int
    hedging_enabled: bool
    
    @property
    def margin_level(self) -> Decimal:
        if self.margin_used == 0:
            return Decimal('inf')
        return (self.balance + self.unrealized_pl) / self.margin_used * 100


class OANDAError(Exception):
    """Custom OANDA API exception"""
    def __init__(self, message: str, error_code: int = None, response: Dict = None):
        super().__init__(message)
        self.error_code = error_code
        self.response = response


class OANDARateLimiter:
    """Rate limiter for OANDA API calls"""
    
    def __init__(self, requests_per_second: int = 10):
        self.requests_per_second = requests_per_second
        self.min_interval = 1.0 / requests_per_second
        self.last_request_time = 0
        self.lock = threading.Lock()
    
    def wait_if_needed(self):
        """Wait if necessary to respect rate limits"""
        with self.lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            
            if time_since_last < self.min_interval:
                sleep_time = self.min_interval - time_since_last
                time.sleep(sleep_time)
            
            self.last_request_time = time.time()


class OANDAClient:
    """Enterprise-grade OANDA REST API client"""
    
    def __init__(self, config_manager):
        self.config = config_manager
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.api_key = config_manager.oanda.api_key
        self.account_id = config_manager.oanda.account_id
        self.base_url = config_manager.oanda.base_url
        self.streaming_url = config_manager.oanda.streaming_url
        self.timeout = config_manager.oanda.timeout
        self.max_retries = config_manager.oanda.max_retries
        self.retry_delay = config_manager.oanda.retry_delay
        
        # Rate limiting
        self.rate_limiter = OANDARateLimiter(requests_per_second=10)
        
        # Session management
        self.session = None
        self.executor = ThreadPoolExecutor(max_workers=5)
        
        # Cache for prices and account info
        self.price_cache = {}
        self.account_cache = {}
        self.cache_timeout = 30  # seconds
        
        self._setup_logging()
        self._validate_credentials()
    
    def _setup_logging(self):
        """Setup logging configuration"""
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def _validate_credentials(self):
        """Validate OANDA credentials"""
        if not self.api_key:
            raise OANDAError("OANDA API key is required")
        
        if not self.account_id:
            raise OANDAError("OANDA account ID is required")
        
        self.logger.info(f"OANDA client initialized for account: {self.account_id}")
    
    async def _make_request(self, method: str, endpoint: str, 
                           params: Dict = None, data: Dict = None, 
                           headers: Dict = None) -> Dict[str, Any]:
        """Make authenticated HTTP request to OANDA API"""
        
        # Rate limiting
        self.rate_limiter.wait_if_needed()
        
        # Prepare headers
        request_headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
            'Accept-Datetime-Format': 'RFC3339'
        }
        
        if headers:
            request_headers.update(headers)
        
        # Build URL
        url = f"{self.base_url}/v3/{endpoint}"
        
        # Retry logic
        for attempt in range(self.max_retries + 1):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.request(
                        method=method,
                        url=url,
                        params=params,
                        json=data,
                        headers=request_headers,
                        timeout=aiohttp.ClientTimeout(total=self.timeout)
                    ) as response:
                        
                        response_text = await response.text()
                        
                        if response.status == 200:
                            return json.loads(response_text)
                        elif response.status == 204:
                            return {}
                        elif response.status in [429, 503]:  # Rate limit or service unavailable
                            if attempt < self.max_retries:
                                wait_time = self.retry_delay * (2 ** attempt)
                                self.logger.warning(f"Rate limited, waiting {wait_time}s")
                                await asyncio.sleep(wait_time)
                                continue
                            else:
                                raise OANDAError(f"Rate limit exceeded after {self.max_retries} retries")
                        else:
                            error_data = json.loads(response_text) if response_text else {}
                            error_message = error_data.get('errorMessage', f'HTTP {response.status}')
                            raise OANDAError(f"API Error: {error_message}", response.status, error_data)
            
            except asyncio.TimeoutError:
                if attempt < self.max_retries:
                    self.logger.warning(f"Request timeout, retrying (attempt {attempt + 1})")
                    await asyncio.sleep(self.retry_delay)
                    continue
                else:
                    raise OANDAError("Request timeout after all retries")
            
            except Exception as e:
                if attempt < self.max_retries:
                    self.logger.warning(f"Request failed: {e}, retrying (attempt {attempt + 1})")
                    await asyncio.sleep(self.retry_delay)
                    continue
                else:
                    raise OANDAError(f"Request failed after all retries: {e}")
    
    async def get_account_info(self, use_cache: bool = True) -> OANDAAccount:
        """Get account information"""
        
        # Check cache
        cache_key = f"account_{self.account_id}"
        if use_cache and cache_key in self.account_cache:
            cached_data, cache_time = self.account_cache[cache_key]
            if time.time() - cache_time < self.cache_timeout:
                return cached_data
        
        endpoint = f"accounts/{self.account_id}"
        response = await self._make_request('GET', endpoint)
        
        account_data = response.get('account', {})
        account = OANDAAccount(
            id=account_data.get('id', ''),
            alias=account_data.get('alias', ''),
            currency=account_data.get('currency', ''),
            balance=Decimal(str(account_data.get('balance', '0'))),
            pl=Decimal(str(account_data.get('pl', '0'))),
            unrealized_pl=Decimal(str(account_data.get('unrealizedPL', '0'))),
            margin_used=Decimal(str(account_data.get('marginUsed', '0'))),
            margin_available=Decimal(str(account_data.get('marginAvailable', '0'))),
            margin_closeout_percent=Decimal(str(account_data.get('marginCloseoutPercent', '0'))),
            open_trade_count=account_data.get('openTradeCount', 0),
            open_position_count=account_data.get('openPositionCount', 0),
            pending_order_count=account_data.get('pendingOrderCount', 0),
            hedging_enabled=account_data.get('hedgingEnabled', False)
        )
        
        # Update cache
        self.account_cache[cache_key] = (account, time.time())
        
        self.logger.info(f"Account info retrieved: Balance={account.balance} {account.currency}")
        return account
    
    async def get_instruments(self) -> List[Dict[str, Any]]:
        """Get available trading instruments"""
        endpoint = f"accounts/{self.account_id}/instruments"
        response = await self._make_request('GET', endpoint)
        
        instruments = response.get('instruments', [])
        self.logger.info(f"Retrieved {len(instruments)} instruments")
        
        return instruments
    
    async def get_prices(self, instruments: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get current prices for instruments"""
        
        # Check cache first
        prices = {}
        uncached_instruments = []
        
        for instrument in instruments:
            cache_key = f"price_{instrument}"
            if cache_key in self.price_cache:
                cached_price, cache_time = self.price_cache[cache_key]
                if time.time() - cache_time < self.cache_timeout:
                    prices[instrument] = cached_price
                    continue
            uncached_instruments.append(instrument)
        
        if uncached_instruments:
            endpoint = "accounts/{}/pricing".format(self.account_id)
            params = {
                'instruments': ','.join(uncached_instruments)
            }
            
            response = await self._make_request('GET', endpoint, params=params)
            
            for price_data in response.get('prices', []):
                instrument = price_data['instrument']
                prices[instrument] = price_data
                
                # Update cache
                cache_key = f"price_{instrument}"
                self.price_cache[cache_key] = (price_data, time.time())
        
        return prices
    
    async def get_positions(self) -> List[OANDAPosition]:
        """Get open positions"""
        endpoint = f"accounts/{self.account_id}/openPositions"
        response = await self._make_request('GET', endpoint)
        
        positions = []
        for position_data in response.get('positions', []):
            instrument = position_data['instrument']
            long_data = position_data.get('long', {})
            short_data = position_data.get('short', {})
            
            position = OANDAPosition(
                instrument=instrument,
                long_units=Decimal(str(long_data.get('units', '0'))),
                short_units=Decimal(str(short_data.get('units', '0'))),
                long_avg_price=Decimal(str(long_data.get('averagePrice', '0'))),
                short_avg_price=Decimal(str(short_data.get('averagePrice', '0'))),
                pl=Decimal(str(long_data.get('pl', '0'))) + Decimal(str(short_data.get('pl', '0'))),
                unrealized_pl=Decimal(str(long_data.get('unrealizedPL', '0'))) + Decimal(str(short_data.get('unrealizedPL', '0'))),
                margin_used=Decimal(str(long_data.get('marginUsed', '0'))) + Decimal(str(short_data.get('marginUsed', '0')))
            )
            
            positions.append(position)
        
        self.logger.info(f"Retrieved {len(positions)} open positions")
        return positions
    
    async def place_order(self, order: OANDAOrder) -> Dict[str, Any]:
        """Place a trading order"""
        
        # Validate order
        if order.side not in ['buy', 'sell']:
            raise OANDAError("Order side must be 'buy' or 'sell'")
        
        if order.type not in ['market', 'limit', 'stop', 'marketIfTouched']:
            raise OANDAError("Invalid order type")
        
        # Get current prices for validation
        prices = await self.get_prices([order.instrument])
        if order.instrument not in prices:
            raise OANDAError(f"Cannot get price for instrument {order.instrument}")
        
        current_price = Decimal(prices[order.instrument]['bids'][0]['price'])
        
        # Risk management checks
        account = await self.get_account_info()
        
        # Check margin availability
        estimated_margin = self._calculate_margin_requirement(order, current_price)
        if account.margin_available < estimated_margin:
            raise OANDAError(f"Insufficient margin. Available: {account.margin_available}, Required: {estimated_margin}")
        
        # Check position limits
        if self._would_exceed_position_limits(order, account):
            raise OANDAError("Order would exceed position limits")
        
        # Prepare order data
        endpoint = f"accounts/{self.account_id}/orders"
        order_data = {
            'order': order.to_dict()
        }
        
        # Add client extensions for tracking
        order_data['order']['clientExtensions'] = {
            'id': f"titanovax_{int(time.time())}",
            'tag': 'titanovax_trading_system',
            'comment': f"Placed at {datetime.now().isoformat()}"
        }
        
        # Place order
        response = await self._make_request('POST', endpoint, data=order_data)
        
        order_fill_data = response.get('orderFillTransaction', {})
        order_cancel_data = response.get('orderCancelTransaction', {})
        
        if order_fill_data:
            self.logger.info(f"Order filled: {order_fill_data.get('id')} for {order.units} {order.instrument}")
            return {
                'status': 'filled',
                'transaction_id': order_fill_data.get('id'),
                'price': order_fill_data.get('price'),
                'pl': order_fill_data.get('pl'),
                'data': order_fill_data
            }
        elif order_cancel_data:
            self.logger.warning(f"Order cancelled: {order_cancel_data.get('reason')}")
            return {
                'status': 'cancelled',
                'reason': order_cancel_data.get('reason'),
                'data': order_cancel_data
            }
        else:
            self.logger.info("Order created successfully")
            return {
                'status': 'created',
                'data': response.get('orderCreateTransaction', {})
            }
    
    def _calculate_margin_requirement(self, order: OANDAOrder, current_price: Decimal) -> Decimal:
        """Calculate margin requirement for order"""
        # Simplified margin calculation
        # In practice, this would use OANDA's margin rates
        notional_value = abs(order.units) * current_price
        margin_rate = Decimal('0.02')  # 2% margin rate (example)
        return notional_value * margin_rate
    
    def _would_exceed_position_limits(self, order: OANDAOrder, account: OANDAAccount) -> bool:
        """Check if order would exceed position limits"""
        # Implement position limit checks based on configuration
        max_risk_percent = self.config.trading.max_risk_percent
        max_positions = self.config.trading.max_positions
        
        # Check account-level risk
        total_exposure = abs(order.units) * Decimal('0.0001')  # Simplified exposure calculation
        account_risk = (total_exposure / account.balance) * 100
        
        if account_risk > max_risk_percent:
            return True
        
        # Check position count
        if account.open_position_count >= max_positions:
            return True
        
        return False
    
    async def close_position(self, instrument: str, units: Optional[Decimal] = None) -> Dict[str, Any]:
        """Close position for instrument"""
        
        # Get current position
        positions = await self.get_positions()
        position = next((p for p in positions if p.instrument == instrument), None)
        
        if not position:
            raise OANDAError(f"No open position for {instrument}")
        
        # Determine closing units
        if units is None:
            # Close entire position
            closing_units = position.net_units
        else:
            closing_units = units
        
        # Create closing order (opposite side)
        closing_side = 'sell' if position.net_units > 0 else 'buy'
        
        close_order = OANDAOrder(
            instrument=instrument,
            units=abs(closing_units),
            side=closing_side,
            type='market',
            client_extensions={
                'id': f"close_{int(time.time())}",
                'tag': 'titanovax_position_close',
                'comment': f"Closing position at {datetime.now().isoformat()}"
            }
        )
        
        return await self.place_order(close_order)
    
    async def get_order_history(self, count: int = 50) -> List[Dict[str, Any]]:
        """Get order history"""
        endpoint = f"accounts/{self.account_id}/orders"
        params = {
            'state': 'ALL',
            'count': min(count, 500)  # OANDA limit
        }
        
        response = await self._make_request('GET', endpoint, params=params)
        orders = response.get('orders', [])
        
        self.logger.info(f"Retrieved {len(orders)} orders from history")
        return orders
    
    async def get_trades(self) -> List[Dict[str, Any]]:
        """Get open trades"""
        endpoint = f"accounts/{self.account_id}/openTrades"
        response = await self._make_request('GET', endpoint)
        
        trades = response.get('trades', [])
        self.logger.info(f"Retrieved {len(trades)} open trades")
        
        return trades
    
    async def get_candles(self, instrument: str, granularity: str = 'M1', 
                         count: int = 500, from_time: Optional[str] = None) -> pd.DataFrame:
        """Get historical candle data"""
        
        endpoint = f"instruments/{instrument}/candles"
        params = {
            'granularity': granularity,
            'count': min(count, 5000)  # OANDA limit
        }
        
        if from_time:
            params['from'] = from_time
        
        response = await self._make_request('GET', endpoint, params=params)
        
        candles_data = response.get('candles', [])
        
        # Convert to DataFrame
        df_data = []
        for candle in candles_data:
            mid = candle['mid']
            df_data.append({
                'time': candle['time'],
                'open': float(mid['o']),
                'high': float(mid['h']),
                'low': float(mid['l']),
                'close': float(mid['c']),
                'volume': int(candle['volume']),
                'complete': candle['complete']
            })
        
        df = pd.DataFrame(df_data)
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        
        self.logger.info(f"Retrieved {len(df)} candles for {instrument}")
        return df
    
    def get_account_health(self) -> Dict[str, Any]:
        """Get account health metrics"""
        return {
            'api_key_configured': bool(self.api_key),
            'account_id_configured': bool(self.account_id),
            'environment': self.config.oanda.environment,
            'rate_limiter_status': 'active',
            'cache_size': len(self.price_cache) + len(self.account_cache),
            'last_request_time': getattr(self.rate_limiter, 'last_request_time', 0)
        }


# Synchronous wrapper for async methods
class OANDAClientSync:
    """Synchronous wrapper for OANDA client"""
    
    def __init__(self, config_manager):
        self.async_client = OANDAClient(config_manager)
        self.loop = None
    
    def _get_event_loop(self):
        """Get or create event loop"""
        try:
            self.loop = asyncio.get_event_loop()
        except RuntimeError:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
        return self.loop
    
    def get_account_info(self, use_cache: bool = True) -> OANDAAccount:
        """Get account information synchronously"""
        loop = self._get_event_loop()
        return loop.run_until_complete(self.async_client.get_account_info(use_cache))
    
    def place_order(self, order: OANDAOrder) -> Dict[str, Any]:
        """Place order synchronously"""
        loop = self._get_event_loop()
        return loop.run_until_complete(self.async_client.place_order(order))
    
    def get_prices(self, instruments: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get prices synchronously"""
        loop = self._get_event_loop()
        return loop.run_until_complete(self.async_client.get_prices(instruments))
    
    def get_positions(self) -> List[OANDAPosition]:
        """Get positions synchronously"""
        loop = self._get_event_loop()
        return loop.run_until_complete(self.async_client.get_positions())


# Usage example and testing
if __name__ == "__main__":
    # Test OANDA client
    try:
        from config_manager import initialize_config
        
        # Initialize configuration
        config = initialize_config()
        
        # Create OANDA client
        client = OANDAClientSync(config)
        
        # Test account info
        account = client.get_account_info()
        print(f"Account Balance: {account.balance} {account.currency}")
        print(f"Open Positions: {account.open_position_count}")
        print(f"Margin Level: {account.margin_level:.2f}%")
        
        # Test instruments
        instruments = client.async_client.get_instruments()
        print(f"Available Instruments: {len(instruments)}")
        
        # Test prices
        prices = client.get_prices(['EUR_USD', 'GBP_USD'])
        for instrument, price_data in prices.items():
            bid = price_data['bids'][0]['price']
            ask = price_data['asks'][0]['price']
            print(f"{instrument}: Bid={bid}, Ask={ask}")
        
        # Test account health
        health = client.async_client.get_account_health()
        print(f"Account Health: {json.dumps(health, indent=2)}")
        
        print("OANDA client test completed successfully!")
        
    except Exception as e:
        print(f"OANDA client test failed: {e}")
        import traceback
        traceback.print_exc()