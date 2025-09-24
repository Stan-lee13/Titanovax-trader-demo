# TitanovaX Signal Processor
# Processes real trading signals for the MT5 Executor EA

param(
    [string]$TitanovaXPath = "C:\titanovax",
    [string]$SignalSource = "file",  # file, api, websocket
    [string]$SignalFile = "C:\titanovax\signals\latest.json",
    [string]$ApiEndpoint = "http://localhost:8080/signals",
    [switch]$Continuous = $false,
    [switch]$Help = $false
)

if ($Help) {
    Write-Host "TitanovaX Signal Processor" -ForegroundColor Green
    Write-Host "==========================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Usage: .\signal_processor.ps1 [options]" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Options:" -ForegroundColor Yellow
    Write-Host "  -TitanovaXPath <path>    Base TitanovaX directory (default: C:\titanovax)" -ForegroundColor White
    Write-Host "  -SignalSource <source>   Signal source: file, api, websocket (default: file)" -ForegroundColor White
    Write-Host "  -SignalFile <path>       Path to signal file (for file source)" -ForegroundColor White
    Write-Host "  -ApiEndpoint <url>       API endpoint for signals (for api source)" -ForegroundColor White
    Write-Host "  -Continuous              Run continuously monitoring for signals" -ForegroundColor White
    Write-Host "  -Help                    Show this help message" -ForegroundColor White
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Yellow
    Write-Host "  .\signal_processor.ps1" -ForegroundColor Gray
    Write-Host "  .\signal_processor.ps1 -SignalSource api -Continuous" -ForegroundColor Gray
    exit 0
}

# Function to validate signal structure
function Test-SignalStructure {
    param([string]$SignalJson)

    try {
        $signal = $SignalJson | ConvertFrom-Json

        # Required fields validation
        $requiredFields = @("timestamp", "symbol", "side", "volume", "model_id", "model_version", "features_hash", "meta")
        foreach ($field in $requiredFields) {
            if (-not (Get-Member -InputObject $signal -Name $field -MemberType Properties)) {
                throw "Missing required field: $field"
            }
        }

        # Validate timestamp
        if ($signal.timestamp -lt 1609459200) {  # 2021-01-01
            throw "Invalid timestamp: $($signal.timestamp)"
        }

        # Validate symbol format
        if ($signal.symbol.Length -lt 6 -or $signal.symbol.Length -gt 10) {
            throw "Invalid symbol format: $($signal.symbol)"
        }

        # Validate side
        if ($signal.side -notin @("BUY", "SELL")) {
            throw "Invalid side: $($signal.side)"
        }

        # Validate volume
        if ($signal.volume -le 0) {
            throw "Invalid volume: $($signal.volume)"
        }

        return $true
    } catch {
        Write-Host "Signal validation failed: $_" -ForegroundColor Red
        return $false
    }
}

# Function to read signal from file
function Read-SignalFromFile {
    param([string]$FilePath)

    if (-not (Test-Path $FilePath)) {
        return $null
    }

    try {
        $signalJson = Get-Content $FilePath -Raw
        if (Test-SignalStructure -SignalJson $signalJson) {
            return $signalJson
        }
    } catch {
        Write-Host "Failed to read signal from file: $_" -ForegroundColor Red
    }

    return $null
}

# Function to fetch signal from API
function Get-SignalFromAPI {
    param([string]$Endpoint)

    try {
        $response = Invoke-WebRequest -Uri $Endpoint -Method GET -UseBasicParsing
        $signalJson = $response.Content

        if (Test-SignalStructure -SignalJson $signalJson) {
            return $signalJson
        }
    } catch {
        Write-Host "Failed to fetch signal from API: $_" -ForegroundColor Red
    }

    return $null
}

# Function to process signal
function Invoke-Signal {
    param(
        [string]$SignalJson,
        [string]$SignalsPath
    )

    Write-Host "Processing signal..." -ForegroundColor Yellow

    # Parse signal
    try {
        $signal = $SignalJson | ConvertFrom-Json

        # Display signal details
        Write-Host "Signal Details:" -ForegroundColor Cyan
        Write-Host "  Timestamp: $(Get-Date -UnixTimeSeconds $signal.timestamp)" -ForegroundColor White
        Write-Host "  Symbol: $($signal.symbol)" -ForegroundColor White
        Write-Host "  Side: $($signal.side)" -ForegroundColor White
        Write-Host "  Volume: $($signal.volume)" -ForegroundColor White
        Write-Host "  Price: $($signal.price)" -ForegroundColor White
        Write-Host "  Model: $($signal.model_id) v$($signal.model_version)" -ForegroundColor White
        Write-Host "  Confidence: $($signal.meta.confidence)" -ForegroundColor White

        # Check for duplicate signal
        $signalHash = "$($signal.timestamp)_$($signal.symbol)_$($signal.side)_$($signal.volume)"
        $hashFile = Join-Path $SignalsPath "last_signal.hash"

        if (Test-Path $hashFile) {
            $lastHash = Get-Content $hashFile -Raw
            if ($lastHash -eq $signalHash) {
                Write-Host "Duplicate signal detected, skipping..." -ForegroundColor Yellow
                return
            }
        }

        # Save signal hash
        $signalHash | Out-File $hashFile -Force

        # Write signal files
        $signalFile = Join-Path $SignalsPath "latest.json"
        $hmacFile = Join-Path $SignalsPath "latest.json.hmac"

        # Generate HMAC (in production, this would be done by the signal source)
        $hmacKeyPath = Join-Path $TitanovaXPath "secrets\hmac.key"
        if (Test-Path $hmacKeyPath) {
            $hmacKey = Get-Content $hmacKeyPath -Raw
            $hmacSignature = Compute-HMAC -Content $SignalJson -Key $hmacKey.Trim()
        } else {
            Write-Host "HMAC key not found, using placeholder" -ForegroundColor Yellow
            $hmacSignature = "placeholder_hmac_signature"
        }

        # Write signal and HMAC
        $SignalJson | Out-File $signalFile -Force -Encoding UTF8
        $hmacSignature | Out-File $hmacFile -Force -Encoding ASCII

        Write-Host "Signal written to: $signalFile" -ForegroundColor Green
        Write-Host "HMAC written to: $hmacFile" -ForegroundColor Green

        # Wait for EA to process
        Write-Host "Waiting for EA to process signal..." -ForegroundColor Yellow
        Start-Sleep -Seconds 5

    } catch {
        Write-Host "Failed to process signal: $_" -ForegroundColor Red
    }
}

# Function to compute HMAC (simplified for demo)
function Compute-HMAC {
    param([string]$Content, [string]$Key)

    # Simple hash-based HMAC for demo purposes
    $combined = $Key + $Content
    $bytes = [System.Text.Encoding]::UTF8.GetBytes($combined)
    $hash = [System.Security.Cryptography.SHA256]::Create().ComputeHash($bytes)
    return [System.BitConverter]::ToString($hash) -replace "-", ""
}

# Function to monitor EA status
function Get-EAStatus {
    param([string]$TitanovaXPath)

    $heartbeatPath = Join-Path $TitanovaXPath "state\hb.json"
    $logPath = Join-Path $TitanovaXPath "logs\exec_log.csv"

    if (Test-Path $heartbeatPath) {
        try {
            $heartbeat = Get-Content $heartbeatPath -Raw | ConvertFrom-Json
            Write-Host "EA Status: $($heartbeat.status)" -ForegroundColor Green
            Write-Host "Last Signal: $(Get-Date -UnixTimeSeconds $heartbeat.last_signal)" -ForegroundColor Gray
            Write-Host "Open Trades: $($heartbeat.open_trades)" -ForegroundColor Gray
        } catch {
            Write-Host "Could not read heartbeat" -ForegroundColor Yellow
        }
    }

    if (Test-Path $logPath) {
        try {
            $logs = Get-Content $logPath -Tail 3
            Write-Host "Recent log entries:" -ForegroundColor Yellow
            $logs | ForEach-Object { Write-Host "  $_" -ForegroundColor Gray }
        } catch {
            Write-Host "Could not read logs" -ForegroundColor Yellow
        }
    }
}

# Main processing function
try {
    Write-Host "TitanovaX Signal Processor" -ForegroundColor Green
    Write-Host "==========================" -ForegroundColor Green
    Write-Host ""

    # Check if TitanovaX directory exists
    if (-not (Test-Path $TitanovaXPath)) {
        Write-Host "TitanovaX directory not found: $TitanovaXPath" -ForegroundColor Red
        Write-Host "Please run deploy.ps1 first to set up the environment." -ForegroundColor Yellow
        exit 1
    }

    Write-Host "Configuration:" -ForegroundColor Yellow
    Write-Host "  TitanovaX Path: $TitanovaXPath" -ForegroundColor White
    Write-Host "  Signal Source: $SignalSource" -ForegroundColor White
    Write-Host ""

    if ($Continuous) {
        Write-Host "Running in continuous mode..." -ForegroundColor Green
        Write-Host "Press Ctrl+C to stop" -ForegroundColor Yellow
        Write-Host ""

        while ($true) {
            $signalJson = $null

            switch ($SignalSource) {
                "file" {
                    $signalJson = Read-SignalFromFile -FilePath $SignalFile
                }
                "api" {
                    $signalJson = Get-SignalFromAPI -Endpoint $ApiEndpoint
                }
                "websocket" {
                    Write-Host "WebSocket source not implemented yet" -ForegroundColor Red
                    $signalJson = $null
                }
                default {
                    Write-Host "Unknown signal source: $SignalSource" -ForegroundColor Red
                    exit 1
                }
            }

            if ($signalJson) {
                Invoke-Signal -SignalJson $signalJson -SignalsPath (Join-Path $TitanovaXPath "signals")
                Get-EAStatus -TitanovaXPath $TitanovaXPath
            } else {
                Write-Host "No signal available, waiting..." -ForegroundColor Gray
            }

            Start-Sleep -Seconds 2
        }
    } else {
        # Single signal processing
        $signalJson = $null

        switch ($SignalSource) {
            "file" {
                $signalJson = Read-SignalFromFile -FilePath $SignalFile
            }
            "api" {
                $signalJson = Get-SignalFromAPI -Endpoint $ApiEndpoint
            }
            "websocket" {
                Write-Host "WebSocket source not implemented yet" -ForegroundColor Red
                $signalJson = $null
            }
            default {
                Write-Host "Unknown signal source: $SignalSource" -ForegroundColor Red
                exit 1
            }
        }

        if ($signalJson) {
            Invoke-Signal -SignalJson $signalJson -SignalsPath (Join-Path $TitanovaXPath "signals")
            Get-EAStatus -TitanovaXPath $TitanovaXPath
        } else {
            Write-Host "No signal found" -ForegroundColor Yellow
        }
    }

    Write-Host "Signal processing completed!" -ForegroundColor Green

} catch {
    Write-Host "Signal processing failed: $_" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Check MT5 terminal for trade execution." -ForegroundColor Green