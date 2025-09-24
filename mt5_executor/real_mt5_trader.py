"""
Real MT5 Broker Integration for TitanovaX Trading System
Implements actual MT5 API connections and trading operations
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import json
import time
from decimal import Decimal

@dataclass
class MT5Config:
    """MT5 connection configuration"""
    login: int
    password: str
    server: str
    path: str = "C:\\Program Files\\MetaTrader 5\\terminal64.exe"
    timeout: int = 60
    max_retries: int = 3
    retry_delay: int = 5

@dataclass
class TradeResult:
    """Trade execution result"""
    success: bool
    order_id: Optional[int] = None
    ticket: Optional[int] = None
    price: Optional[float] = None
    volume: Optional[float] = None
    sl: Optional[float] = None
    tp: Optional[float] = None
    comment: Optional[str] = None
    error_code: Optional[int] = None
    error_description: Optional[str] = None
    execution_time: Optional[float] = None

@dataclass
class MarketData:
    """Market data structure"""
    symbol: str
    bid: float
    ask: float
    spread: float
    volume: int
    time: datetime
    tick_value: float
    tick_size: float
    point: float

class RealMT5Trader:
    """Real MT5 broker integration"""
    
    def __init__(self, config: MT5Config, symbol: str = "EURUSD", 
                 lot_size: float = 0.1, max_risk_percent: float = 2.0):
        self.config = config
        self.symbol = symbol
        self.lot_size = lot_size
        self.max_risk_percent = max_risk_percent
        self.logger = self._setup_logging()
        self.connected = False
        self.account_info = None
        self.symbol_info = None
        
        # Trading statistics
        self.trade_stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0
        }
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        # Create logs directory
        Path("logs").mkdir(exist_ok=True)
        
        # File handler
        handler = logging.FileHandler(f'logs/mt5_trader_{datetime.now().strftime("%Y%m%d")}.log')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        # Console handler
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(formatter)
        logger.addHandler(console)
        
        return logger
    
    def connect(self) -> bool:
        """Connect to MT5 terminal"""
        self.logger.info(f"Connecting to MT5: {self.config.server}")
        
        try:
            # Initialize MT5
            if not mt5.initialize(
                path=self.config.path,
                login=self.config.login,
                password=self.config.password,
                server=self.config.server,
                timeout=self.config.timeout * 1000  # Convert to milliseconds
            ):
                error_code = mt5.last_error()
                self.logger.error(f"MT5 initialization failed: {error_code}")
                return False
            
            # Check connection
            if not mt5.terminal_info():
                self.logger.error("MT5 terminal info unavailable")
                mt5.shutdown()
                return False
            
            # Get account info
            self.account_info = mt5.account_info()
            if self.account_info is None:
                self.logger.error("Failed to get account info")
                mt5.shutdown()
                return False
            
            # Get symbol info
            self.symbol_info = mt5.symbol_info(self.symbol)
            if self.symbol_info is None:
                self.logger.error(f"Failed to get symbol info for {self.symbol}")
                mt5.shutdown()
                return False
            
            # Enable symbol
            if not self.symbol_info.visible:
                if not mt5.symbol_select(self.symbol, True):
                    self.logger.error(f"Failed to select symbol {self.symbol}")
                    mt5.shutdown()
                    return False
            
            self.connected = True
            self.logger.info(f"Connected to MT5. Account: {self.account_info.login}, Balance: {self.account_info.balance}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Connection error: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from MT5"""
        if self.connected:
            mt5.shutdown()
            self.connected = False
            self.logger.info("Disconnected from MT5")
    
    def get_account_info(self) -> Optional[Dict]:
        """Get current account information"""
        if not self.connected:
            return None
        
        account = mt5.account_info()
        if account is None:
            return None
        
        return {
            'login': account.login,
            'balance': account.balance,
            'equity': account.equity,
            'profit': account.profit,
            'margin': account.margin,
            'margin_free': account.margin_free,
            'margin_level': account.margin_level,
            'leverage': account.leverage,
            'currency': account.currency
        }
    
    def get_market_data(self, timeframe: str = "M1", count: int = 100) -> Optional[pd.DataFrame]:
        """Get historical market data"""
        if not self.connected:
            self.logger.error("Not connected to MT5")
            return None
        
        # Map timeframe strings to MT5 constants
        timeframe_map = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1
        }
        
        tf = timeframe_map.get(timeframe, mt5.TIMEFRAME_M1)
        
        # Get rates
        rates = mt5.copy_rates_from_pos(self.symbol, tf, 0, count)
        if rates is None:
            self.logger.error(f"Failed to get rates for {self.symbol}")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        # Rename columns to standard format
        df.rename(columns={
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'tick_volume': 'volume'
        }, inplace=True)
        
        return df[['open', 'high', 'low', 'close', 'volume']]
    
    def get_current_price(self) -> Optional[MarketData]:
        """Get current market price"""
        if not self.connected:
            return None
        
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            return None
        
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            return None
        
        return MarketData(
            symbol=self.symbol,
            bid=tick.bid,
            ask=tick.ask,
            spread=(tick.ask - tick.bid) / symbol_info.point,
            volume=tick.volume,
            time=datetime.fromtimestamp(tick.time),
            tick_value=symbol_info.trade_tick_value,
            tick_size=symbol_info.trade_tick_size,
            point=symbol_info.point
        )
    
    def calculate_position_size(self, stop_loss_pips: float, 
                               risk_percent: Optional[float] = None) -> float:
        """Calculate position size based on risk management"""
        if not self.connected:
            return 0.0
        
        account = self.get_account_info()
        if account is None:
            return 0.0
        
        risk_percent = risk_percent or self.max_risk_percent
        risk_amount = account['balance'] * (risk_percent / 100)
        
        # Get symbol info
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            return 0.0
        
        # Calculate pip value
        pip_value = symbol_info.trade_tick_value * stop_loss_pips
        
        # Calculate position size
        if pip_value > 0:
            position_size = risk_amount / pip_value
            
            # Apply limits
            position_size = min(position_size, symbol_info.volume_max)
            position_size = max(position_size, symbol_info.volume_min)
            
            # Round to volume step
            position_size = round(position_size / symbol_info.volume_step) * symbol_info.volume_step
            
            return position_size
        
        return 0.0
    
    def open_position(self, action: str, volume: float, 
                     sl_pips: Optional[float] = None, 
                     tp_pips: Optional[float] = None,
                     comment: str = "TitanovaX") -> TradeResult:
        """Open a new position"""
        
        start_time = time.time()
        
        if not self.connected:
            return TradeResult(
                success=False,
                error_code=-1,
                error_description="Not connected to MT5",
                execution_time=time.time() - start_time
            )
        
        # Get current price
        market_data = self.get_current_price()
        if market_data is None:
            return TradeResult(
                success=False,
                error_code=-2,
                error_description="Failed to get market data",
                execution_time=time.time() - start_time
            )
        
        # Determine order type
        if action.lower() == "buy":
            order_type = mt5.ORDER_TYPE_BUY
            price = market_data.ask
            sl_price = price - (sl_pips * market_data.point) if sl_pips else 0
            tp_price = price + (tp_pips * market_data.point) if tp_pips else 0
        elif action.lower() == "sell":
            order_type = mt5.ORDER_TYPE_SELL
            price = market_data.bid
            sl_price = price + (sl_pips * market_data.point) if sl_pips else 0
            tp_price = price - (tp_pips * market_data.point) if tp_pips else 0
        else:
            return TradeResult(
                success=False,
                error_code=-3,
                error_description="Invalid action. Use 'buy' or 'sell'",
                execution_time=time.time() - start_time
            )
        
        # Create order request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "sl": sl_price if sl_pips else 0,
            "tp": tp_price if tp_pips else 0,
            "deviation": 20,
            "magic": 234000,  # EA magic number
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Send order
        result = mt5.order_send(request)
        
        execution_time = time.time() - start_time
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return TradeResult(
                success=False,
                error_code=result.retcode,
                error_description=result.comment,
                execution_time=execution_time
            )
        
        # Update trade statistics
        self.trade_stats['total_trades'] += 1
        
        return TradeResult(
            success=True,
            order_id=result.order,
            ticket=result.position,
            price=result.price,
            volume=result.volume,
            sl=result.sl,
            tp=result.tp,
            comment=result.comment,
            execution_time=execution_time
        )
    
    def close_position(self, ticket: int, volume: Optional[float] = None,
                      comment: str = "TitanovaX Close") -> TradeResult:
        """Close an existing position"""
        
        start_time = time.time()
        
        if not self.connected:
            return TradeResult(
                success=False,
                error_code=-1,
                error_description="Not connected to MT5",
                execution_time=time.time() - start_time
            )
        
        # Get position info
        position = mt5.positions_get(ticket=ticket)
        if position is None or len(position) == 0:
            return TradeResult(
                success=False,
                error_code=-4,
                error_description="Position not found",
                execution_time=time.time() - start_time
            )
        
        position = position[0]
        
        # Get current price
        market_data = self.get_current_price()
        if market_data is None:
            return TradeResult(
                success=False,
                error_code=-2,
                error_description="Failed to get market data",
                execution_time=time.time() - start_time
            )
        
        # Determine close price and type
        if position.type == mt5.POSITION_TYPE_BUY:
            close_price = market_data.bid
            close_type = mt5.ORDER_TYPE_SELL
        else:
            close_price = market_data.ask
            close_type = mt5.ORDER_TYPE_BUY
        
        # Use full volume if not specified
        close_volume = volume if volume else position.volume
        
        # Create close request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": close_volume,
            "type": close_type,
            "position": ticket,
            "price": close_price,
            "deviation": 20,
            "magic": 234000,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Send close order
        result = mt5.order_send(request)
        
        execution_time = time.time() - start_time
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return TradeResult(
                success=False,
                error_code=result.retcode,
                error_description=result.comment,
                execution_time=execution_time
            )
        
        return TradeResult(
            success=True,
            order_id=result.order,
            ticket=result.position,
            price=result.price,
            volume=result.volume,
            comment=result.comment,
            execution_time=execution_time
        )
    
    def get_positions(self) -> List[Dict]:
        """Get all open positions"""
        if not self.connected:
            return []
        
        positions = mt5.positions_get(symbol=self.symbol)
        if positions is None:
            return []
        
        result = []
        for pos in positions:
            result.append({
                'ticket': pos.ticket,
                'symbol': pos.symbol,
                'type': 'buy' if pos.type == mt5.POSITION_TYPE_BUY else 'sell',
                'volume': pos.volume,
                'price_open': pos.price_open,
                'price_current': pos.price_current,
                'sl': pos.sl,
                'tp': pos.tp,
                'profit': pos.profit,
                'comment': pos.comment,
                'magic': pos.magic,
                'time': datetime.fromtimestamp(pos.time)
            })
        
        return result
    
    def get_orders(self) -> List[Dict]:
        """Get all pending orders"""
        if not self.connected:
            return []
        
        orders = mt5.orders_get(symbol=self.symbol)
        if orders is None:
            return []
        
        result = []
        for order in orders:
            result.append({
                'ticket': order.ticket,
                'symbol': order.symbol,
                'type': order.type,
                'volume': order.volume_current,
                'price_open': order.price_open,
                'sl': order.sl,
                'tp': order.tp,
                'comment': order.comment,
                'magic': order.magic,
                'time_setup': datetime.fromtimestamp(order.time_setup)
            })
        
        return result
    
    def modify_position(self, ticket: int, sl: Optional[float] = None,
                       tp: Optional[float] = None) -> TradeResult:
        """Modify existing position (SL/TP)"""
        
        start_time = time.time()
        
        if not self.connected:
            return TradeResult(
                success=False,
                error_code=-1,
                error_description="Not connected to MT5",
                execution_time=time.time() - start_time
            )
        
        # Get position info
        position = mt5.positions_get(ticket=ticket)
        if position is None or len(position) == 0:
            return TradeResult(
                success=False,
                error_code=-4,
                error_description="Position not found",
                execution_time=time.time() - start_time
            )
        
        position = position[0]
        
        # Create modification request
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": self.symbol,
            "position": ticket,
            "sl": sl if sl is not None else position.sl,
            "tp": tp if tp is not None else position.tp,
            "magic": position.magic,
        }
        
        # Send modification
        result = mt5.order_send(request)
        
        execution_time = time.time() - start_time
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return TradeResult(
                success=False,
                error_code=result.retcode,
                error_description=result.comment,
                execution_time=execution_time
            )
        
        return TradeResult(
            success=True,
            order_id=result.order,
            ticket=result.position,
            sl=result.sl,
            tp=result.tp,
            execution_time=execution_time
        )
    
    def get_trading_history(self, days: int = 30) -> pd.DataFrame:
        """Get trading history for specified number of days"""
        if not self.connected:
            return pd.DataFrame()
        
        from_date = datetime.now() - timedelta(days=days)
        to_date = datetime.now()
        
        # Get deals history
        deals = mt5.history_deals_get(from_date, to_date)
        if deals is None:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(list(deals), columns=deals[0]._asdict().keys())
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        return df
    
    def get_trade_statistics(self) -> Dict:
        """Get comprehensive trade statistics"""
        history = self.get_trading_history(30)
        
        if history.empty:
            return self.trade_stats
        
        # Calculate statistics
        closed_deals = history[history['entry'] == 1]  # Entry deals only
        
        if len(closed_deals) > 0:
            total_trades = len(closed_deals)
            winning_trades = len(closed_deals[closed_deals['profit'] > 0])
            losing_trades = len(closed_deals[closed_deals['profit'] < 0])
            total_pnl = closed_deals['profit'].sum()
            
            avg_win = closed_deals[closed_deals['profit'] > 0]['profit'].mean() if winning_trades > 0 else 0
            avg_loss = closed_deals[closed_deals['profit'] < 0]['profit'].mean() if losing_trades > 0 else 0
            
            # Calculate max drawdown
            cumulative_pnl = closed_deals['profit'].cumsum()
            running_max = cumulative_pnl.expanding().max()
            drawdown = cumulative_pnl - running_max
            max_drawdown = drawdown.min()
            
            self.trade_stats.update({
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'total_pnl': total_pnl,
                'max_drawdown': abs(max_drawdown),
                'avg_win': avg_win,
                'avg_loss': abs(avg_loss),
                'win_rate': winning_trades / total_trades if total_trades > 0 else 0
            })
        
        return self.trade_stats
    
    async def monitor_positions_async(self, callback=None):
        """Async position monitoring"""
        while self.connected:
            try:
                positions = self.get_positions()
                account_info = self.get_account_info()
                
                if callback:
                    await callback(positions, account_info)
                
                await asyncio.sleep(1)  # Check every second
                
            except Exception as e:
                self.logger.error(f"Position monitoring error: {e}")
                await asyncio.sleep(5)  # Wait longer on error
    
    def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        health = {
            'connected': self.connected,
            'timestamp': datetime.now().isoformat(),
            'symbol': self.symbol,
            'status': 'healthy' if self.connected else 'disconnected'
        }
        
        if self.connected:
            # Check account
            account = self.get_account_info()
            if account:
                health['account'] = {
                    'balance': account['balance'],
                    'equity': account['equity'],
                    'margin_level': account['margin_level']
                }
            
            # Check symbol
            market_data = self.get_current_price()
            if market_data:
                health['market'] = {
                    'bid': market_data.bid,
                    'ask': market_data.ask,
                    'spread': market_data.spread
                }
            
            # Check positions
            positions = self.get_positions()
            health['positions'] = {
                'count': len(positions),
                'total_pnl': sum(pos['profit'] for pos in positions)
            }
        
        return health

def main():
    """Example usage and testing"""
    
    # Load configuration
    config_path = Path("config/mt5_config.json")
    if config_path.exists():
        with open(config_path, 'r') as f:
            config_data = json.load(f)
    else:
        # Default configuration
        config_data = {
            "login": 12345,
            "password": "your_password",
            "server": "YourBroker-Server",
            "path": "C:\\Program Files\\MetaTrader 5\\terminal64.exe"
        }
    
    config = MT5Config(**config_data)
    
    # Initialize trader
    trader = RealMT5Trader(
        config=config,
        symbol="EURUSD",
        lot_size=0.1,
        max_risk_percent=2.0
    )
    
    # Connect to MT5
    if trader.connect():
        print("✅ Connected to MT5")
        
        # Get account info
        account = trader.get_account_info()
        if account:
            print(f"Account: {account['login']}, Balance: {account['balance']} {account['currency']}")
        
        # Get market data
        market_data = trader.get_current_price()
        if market_data:
            print(f"EURUSD: Bid={market_data.bid}, Ask={market_data.ask}, Spread={market_data.spread}")
        
        # Get positions
        positions = trader.get_positions()
        print(f"Open positions: {len(positions)}")
        
        # Get trade statistics
        stats = trader.get_trade_statistics()
        print(f"Trade stats: {stats}")
        
        # Perform health check
        health = trader.health_check()
        print(f"Health: {health}")
        
        # Disconnect
        trader.disconnect()
        print("✅ Disconnected from MT5")
    else:
        print("❌ Failed to connect to MT5")

if __name__ == "__main__":
    main()