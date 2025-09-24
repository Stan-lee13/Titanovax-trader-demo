# SignalExecutorEA Deployment Script
# This script sets up the MT5 Expert Advisor environment

param(
    [string]$MT5Path = "C:\Program Files\MetaTrader 5",
    [switch]$CreateDemo,
    [switch]$InstallEA,
    [switch]$SetupDirectories,
    [string]$EAName = "SignalExecutorEA",
    [string]$ProfileName = "TitanovaxDemo"
)

# Configuration
$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$EAFiles = @(
    "SignalExecutorEA.mq5",
    "SignalExecutorEA_REST.mq5",
    "signal_schema.json",
    "latency_analysis.md"
)

$Directories = @(
    "C:\titanovax",
    "C:\titanovax\signals",
    "C:\titanovax\secrets",
    "C:\titanovax\state",
    "C:\titanovax\logs",
    "C:\titanovax\screenshots"
)

# Colors for output
$Green = "Green"
$Yellow = "Yellow"
$Red = "Red"

function Write-Success {
    param([string]$Message)
    Write-Host $Message -ForegroundColor $Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host $Message -ForegroundColor $Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host $Message -ForegroundColor $Red
}

function Test-MT5Installation {
    if (Test-Path $MT5Path) {
        Write-Success "MetaTrader 5 found at: $MT5Path"
        return $true
    } else {
        Write-Error "MetaTrader 5 not found at: $MT5Path"
        Write-Warning "Please install MetaTrader 5 or specify correct path with -MT5Path parameter"
        return $false
    }
}

function Create-Directories {
    Write-Host "Creating directories..." -ForegroundColor $Yellow

    foreach ($dir in $Directories) {
        if (Test-Path $dir) {
            Write-Warning "Directory already exists: $dir"
        } else {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
            Write-Success "Created directory: $dir"
        }
    }
}

function Create-DemoKey {
    Write-Host "Creating demo HMAC key..." -ForegroundColor $Yellow

    $KeyFile = "C:\titanovax\secrets\hmac.key"
    $KeySize = 32  # 256-bit key

    # Generate random key
    $Random = [System.Security.Cryptography.RandomNumberGenerator]::Create()
    $KeyBytes = New-Object byte[] $KeySize
    $Random.GetBytes($KeyBytes)

    # Convert to hex string
    $KeyHex = -join ($KeyBytes | ForEach-Object { $_.ToString("x2") })

    # Write key file
    $KeyHex | Out-File -FilePath $KeyFile -Encoding ASCII
    Write-Success "HMAC key created: $KeyFile"

    # Set restrictive permissions
    $Acl = Get-Acl $KeyFile
    $Acl.SetAccessRuleProtection($true, $false)
    $AccessRule = New-Object System.Security.AccessControl.FileSystemAccessRule("Administrators", "FullControl", "Allow")
    $Acl.SetAccessRule($AccessRule)
    $AccessRule = New-Object System.Security.AccessControl.FileSystemAccessRule("SYSTEM", "FullControl", "Allow")
    $Acl.SetAccessRule($AccessRule)
    Set-Acl -Path $KeyFile -AclObject $Acl

    Write-Success "HMAC key permissions set (Administrators and SYSTEM only)"
}

function Create-DemoSignal {
    Write-Host "Creating demo signal..." -ForegroundColor $Yellow

    $SignalFile = "C:\titanovax\signals\latest.json"
    $HmacFile = "C:\titanovax\signals\latest.json.hmac"

    # Demo signal JSON
    $DemoSignal = @"
{
    "timestamp": $([int](Get-Date -UFormat %s)),
    "symbol": "EURUSD",
    "side": "BUY",
    "volume": 0.01,
    "price": 1.12345,
    "model_id": "demo_ensemble_v1",
    "model_version": "2025-01-20",
    "features_hash": "sha256:abcd1234efgh5678",
    "meta": {
        "reason": "momentum_signal_demo",
        "confidence": 0.72
    }
}
"@

    $DemoSignal | Out-File -FilePath $SignalFile -Encoding UTF8
    Write-Success "Demo signal created: $SignalFile"

    # For demo purposes, create a simple HMAC (in production, use proper HMAC)
    $DemoHmac = Get-FileHash -Path $SignalFile -Algorithm SHA256 | Select-Object -ExpandProperty Hash
    $DemoHmac | Out-File -FilePath $HmacFile -Encoding ASCII
    Write-Success "Demo HMAC created: $HmacFile"
}

function Install-ExpertAdvisor {
    Write-Host "Installing Expert Advisor..." -ForegroundColor $Yellow

    if (-not (Test-MT5Installation)) {
        return
    }

    $MQL5Path = Join-Path $MT5Path "MQL5"
    $ExpertsPath = Join-Path $MQL5Path "Experts"
    $IncludePath = Join-Path $MQL5Path "Include"

    if (-not (Test-Path $ExpertsPath)) {
        Write-Error "MQL5 Experts directory not found: $ExpertsPath"
        return
    }

    # Copy EA files
    foreach ($file in $EAFiles) {
        $SourceFile = Join-Path $ScriptRoot $file
        $DestFile = Join-Path $ExpertsPath $file

        if (Test-Path $SourceFile) {
            Copy-Item -Path $SourceFile -Destination $DestFile -Force
            Write-Success "Copied: $file -> $DestFile"
        } else {
            Write-Warning "Source file not found: $SourceFile"
        }
    }
}

function Create-MT5Profile {
    Write-Host "Creating MT5 profile..." -ForegroundColor $Yellow

    if (-not (Test-MT5Installation)) {
        return
    }

    $ProfilesPath = Join-Path $MT5Path "profiles"

    if (-not (Test-Path $ProfilesPath)) {
        New-Item -ItemType Directory -Path $ProfilesPath -Force | Out-Null
    }

    $ProfilePath = Join-Path $ProfilesPath "$ProfileName.ini"

    $ProfileContent = @"
[Common]
ProfileType=2

[Chart]
ChartShift=1
ChartShiftSize=50
ChartMode=1

[Objects]
ShowObjects=1
ShowObjectDescription=1

[Experts]
Enabled=1
"@

    $ProfileContent | Out-File -FilePath $ProfilePath -Encoding ASCII
    Write-Success "MT5 profile created: $ProfilePath"
}

function Show-SetupInstructions {
    Write-Host "`n" + "="*60 -ForegroundColor $Green
    Write-Host "TITANOVAX SIGNAL EXECUTOR EA - SETUP COMPLETE" -ForegroundColor $Green
    Write-Host "="*60 -ForegroundColor $Green

    Write-Host "`nNEXT STEPS:" -ForegroundColor $Yellow
    Write-Host "1. Open MetaTrader 5"
    Write-Host "2. Go to File -> Open Data Folder"
    Write-Host "3. Navigate to MQL5/Experts/"
    Write-Host "4. Compile the EA files (F7 or right-click -> Compile)"
    Write-Host "5. Attach EA to EURUSD M1 chart"
    Write-Host "6. Enable AlgoTrading (Ctrl+E)"

    Write-Host "`nDEMO SIGNAL TESTING:" -ForegroundColor $Yellow
    Write-Host "1. Run the signal processor:"
    Write-Host "   .\signal_processor.ps1"
    Write-Host "2. Or run continuous monitoring:"
    Write-Host "   .\signal_processor.ps1 -Continuous"
    Write-Host "3. Check logs in: C:\titanovax\logs\exec_log.csv"
    Write-Host "4. Check heartbeat in: C:\titanovax\state\hb.json"

    Write-Host "`nRISK SETTINGS:" -ForegroundColor $Yellow
    Write-Host "- Max risk per trade: $InpRiskMaxPerTrade USD"
    Write-Host "- Max open trades: $InpMaxOpenTrades"
    Write-Host "- Daily drawdown cap: $InpDailyDrawdownCap USD"

    Write-Host "`nSUPPORT:" -ForegroundColor $Yellow
    Write-Host "For issues, check MT5 Experts tab logs"
    Write-Host "Review latency analysis: .\latency_analysis.md"
}

# Main execution
if (-not $CreateDemo -and -not $InstallEA -and -not $SetupDirectories) {
    Write-Host "Titanovax SignalExecutorEA Deployment Script" -ForegroundColor $Green
    Write-Host "="*50 -ForegroundColor $Green
    Write-Host "`nUsage: $($MyInvocation.MyCommand.Name) [options]" -ForegroundColor $Yellow
    Write-Host "`nOptions:" -ForegroundColor $Yellow
    Write-Host "  -CreateDemo      Create demo environment with sample files"
    Write-Host "  -InstallEA       Install EA files to MT5 directory"
    Write-Host "  -SetupDirectories Create required directories"
    Write-Host "  -MT5Path <path>  Specify MT5 installation path"
    Write-Host "  -EAName <name>   EA name (default: SignalExecutorEA)"
    Write-Host "  -ProfileName <name> MT5 profile name"
    Write-Host "`nExamples:" -ForegroundColor $Yellow
    Write-Host "  $($MyInvocation.MyCommand.Name) -CreateDemo"
    Write-Host "  $($MyInvocation.MyCommand.Name) -InstallEA -MT5Path 'C:\MT5'"
    Write-Host "  $($MyInvocation.MyCommand.Name) -SetupDirectories -CreateDemo -InstallEA"
    exit
}

# Execute requested actions
if ($SetupDirectories) {
    Create-Directories
}

if ($CreateDemo) {
    Create-Directories
    Create-DemoKey
    Create-DemoSignal
    Create-MT5Profile
}

if ($InstallEA) {
    Install-ExpertAdvisor
}

if ($CreateDemo -or $InstallEA -or $SetupDirectories) {
    Show-SetupInstructions
}

Write-Host "`nDeployment script completed!" -ForegroundColor $Green