# TitanovaX MT5 Executor

A safe, latency-conscious MetaTrader 5 Expert Advisor that receives validated trading signals and executes them with comprehensive risk management.

## Overview

The MT5 Executor is the execution layer of the TitanovaX trading system. It provides:

- **Signal Validation**: HMAC-SHA256 signature verification
- **Risk Management**: Per-trade loss limits, max open trades, daily drawdown protection
- **Safe Execution**: Retry logic with exponential backoff, slippage monitoring
- **Comprehensive Logging**: Detailed execution logs with latency and outcome tracking
- **Health Monitoring**: Real-time heartbeat and status reporting
- **Kill Switch**: Automatic trading disable on risk limit breach

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Signal Source │    │   MT5 Executor   │    │   Risk Engine   │
│                 │───▶│                  │───▶│                 │
│  latest.json    │    │  SignalExecutorEA│    │  Risk Limits    │
│  latest.json.hmac│    │                  │    │  Kill Switch    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   HMAC Key      │    │   Execution Log  │    │   Heartbeat     │
│   hmac.key      │    │   exec_log.csv   │    │   hb.json       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## File Structure

```
mt5-executor/
├── SignalExecutorEA.mq5      # Main Expert Advisor
├── signal_schema.json        # Signal validation schema
├── deploy.ps1               # Deployment script
├── simulate_signals.ps1       # Signal simulation for testing
└── README.md                # This file
```

## Installation

### Prerequisites

- MetaTrader 5 installed on Windows
- MetaEditor (comes with MT5)
- PowerShell 5.0 or higher
- Demo trading account

### Quick Start

1. **Deploy the system:**
   ```powershell
   .\deploy.ps1
   ```

2. **Compile the EA:**
   - Open MetaEditor
   - File → Open → Navigate to `MQL5\Experts\TitanovaX\SignalExecutorEA.mq5`
   - Press F7 to compile (should show 0 errors, 0 warnings)

3. **Attach to chart:**
   - Open MetaTrader 5
   - Open EURUSD M1 chart
   - Drag SignalExecutorEA from Navigator to chart
   - Enable "Allow automated trading"
   - Click OK

4. **Test with simulated signals:**
   ```powershell
   .\simulate_signals.ps1 -SignalCount 3 -DelaySeconds 5
   ```

## Configuration

The EA uses the following input parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| InpSignalPath | `C:\titanovax\signals\latest.json` | Path to signal file |
| InpHMACKeyPath | `C:\titanovax\secrets\hmac.key` | Path to HMAC key |
| InpLogPath | `C:\titanovax\logs\exec_log.csv` | Execution log path |
| InpStatePath | `C:\titanovax\state\hb.json` | Heartbeat file path |
| InpDisabledLockPath | `C:\titanovax\state\disabled.lock` | Kill switch file |
| InpMaxRiskPerTrade | 50.0 | Max loss per trade (account currency) |
| InpMaxOpenTrades | 5 | Maximum open trades |
| InpDailyDrawdownLimit | 200.0 | Daily PnL drawdown limit |
| InpMaxSlippagePips | 10 | Maximum slippage in pips |
| InpOrderRetryCount | 3 | Maximum order retry attempts |
| InpTimerInterval | 2000 | Timer interval in milliseconds |

## Signal Format

Trading signals must follow this JSON schema:

```json
{
  "timestamp": 1703123456,
  "symbol": "EURUSD",
  "side": "BUY",
  "volume": 0.01,
  "price": 1.0850,
  "sl": 1.0800,
  "tp": 1.0900,
  "model_id": "ensemble_v1",
  "model_version": "v1.2.3",
  "features_hash": "sha256:abcd...",
  "hmac_signature": "sha256:hmac...",
  "meta": {
    "signal_id": "sig_001",
    "prediction_window": 60,
    "market_condition": "trending",
    "confidence": 0.78
  }
}
```

Files required:
- `latest.json` - The signal JSON
- `latest.json.hmac` - HMAC-SHA256 signature

## Safety Features

### Risk Management

1. **Per-Trade Risk Limit**: Maximum loss per trade in account currency
2. **Max Open Trades**: Limits total concurrent positions
3. **Daily Drawdown**: Stops trading if daily losses exceed limit
4. **Slippage Protection**: Maximum allowed slippage in pips

### Kill Switch

Trading is automatically disabled when:
- Daily drawdown limit exceeded
- Risk validation fails
- Manual `disabled.lock` file created

To re-enable trading, delete `C:\titanovax\state\disabled.lock`.

### Signal Validation

- HMAC signature verification
- Timestamp validation (max 5 minutes old)
- Symbol matching current chart
- Side validation (BUY/SELL only)
- Price and volume validation

## Monitoring

### Heartbeat

Real-time status in `C:\titanovax\state\hb.json`:
```json
{
  "timestamp": 1703123456,
  "cpu_time": 123.45,
  "last_signal": 1703123400,
  "status": "enabled",
  "error": "",
  "open_trades": 2
}
```

### Execution Log

Detailed trade history in `C:\titanovax\logs\exec_log.csv`:
```csv
timestamp,ticket,symbol,side,volume,price,slippage,sl,tp,outcome,model_id,model_version,features_hash
```

## Testing

### Unit Tests

The system includes comprehensive testing:

1. **Signal Simulation**:
   ```powershell
   .\simulate_signals.ps1 -SignalCount 5 -RandomSignals
   ```

2. **Invalid Signal Test**:
   ```powershell
   .\simulate_signals.ps1 -InvalidHMAC -SignalCount 1
   ```

3. **Monitor Status**:
   ```powershell
   Get-Content C:\titanovax\state\hb.json
   Get-Content C:\titanovax\logs\exec_log.csv
   ```

### Manual Testing

1. Create signal files manually
2. Monitor MT5 terminal for trade execution
3. Check logs for validation results
4. Test kill switch by creating disabled.lock

## Troubleshooting

### Common Issues

1. **EA not compiling**: Check MetaEditor for syntax errors
2. **No trades executed**: Check signal files and logs
3. **HMAC validation fails**: Verify key file and signature
4. **Trading disabled**: Check disabled.lock and risk limits

### Debug Steps

1. Check MT5 Experts tab for error messages
2. Monitor `C:\titanovax\logs\exec_log.csv`
3. Check heartbeat file for status
4. Use MetaEditor debugger if needed

## Security

- HMAC keys stored with restricted file permissions
- No network calls from EA (local files only)
- Signal validation before execution
- Encrypted key storage recommended for production

## Performance

- Timer interval: 2 seconds (configurable)
- Signal processing: <100ms typical
- Order execution: Market orders with FOK/IOC
- Memory usage: <10MB typical

## Acceptance Criteria

✅ EA compiles without warnings  
✅ Reads and validates signals with HMAC  
✅ Executes demo trades with risk limits  
✅ Writes detailed execution logs  
✅ Provides real-time heartbeat  
✅ Implements kill switch functionality  
✅ Handles order retry and slippage  
✅ Includes comprehensive testing tools  

## Next Steps

1. Test with demo account
2. Integrate with ML signal generation (Part 2)
3. Set up monitoring and alerting
4. Deploy to production environment