# PostgreSQL Setup Script
Write-Host "Setting up PostgreSQL for TitanovaX..." -ForegroundColor Green

# Check if PostgreSQL is already installed
$pgPaths = @(
    "C:\Program Files\PostgreSQL\15\bin",
    "C:\Program Files\PostgreSQL\14\bin", 
    "C:\Program Files\PostgreSQL\13\bin",
    "C:\Program Files\PostgreSQL\12\bin"
)

$pgBin = $null
foreach ($path in $pgPaths) {
    if (Test-Path $path) {
        $pgBin = $path
        break
    }
}

if ($pgBin) {
    Write-Host "Found PostgreSQL at: $pgBin" -ForegroundColor Green
    
    # Create data directory if it doesn't exist
    $dataDir = "$pwd\postgres_data"
    if (!(Test-Path $dataDir)) {
        New-Item -ItemType Directory -Path $dataDir | Out-Null
        
        # Initialize database cluster
        Write-Host "Initializing database cluster..." -ForegroundColor Yellow
        & "$pgBin\initdb.exe" -D $dataDir -U postgres --encoding=UTF8 --locale=en_US.UTF-8
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Database cluster initialized" -ForegroundColor Green
        } else {
            Write-Host "Database initialization failed" -ForegroundColor Red
        }
    }
    
    # Create startup script
    $startupScript = @"
@echo off
echo Starting PostgreSQL for TitanovaX...
cd /d "$pwd"
"$pgBin\postgres.exe" -D "postgres_data" -p 5432
echo PostgreSQL started on port 5432
pause
"@
    
    Set-Content -Path "start_postgres.bat" -Value $startupScript
    Write-Host "Startup script created: start_postgres.bat" -ForegroundColor Green
    
    # Create TitanovaX database
    Write-Host "Creating TitanovaX database..." -ForegroundColor Yellow
    & "$pgBin\createdb.exe" -U postgres -h localhost -p 5432 titanovax
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "TitanovaX database created" -ForegroundColor Green
    } else {
        Write-Host "Database may already exist" -ForegroundColor Yellow
    }
    
    return $true
} else {
    Write-Host "PostgreSQL not found. Please install PostgreSQL first." -ForegroundColor Red
    return $false
}
