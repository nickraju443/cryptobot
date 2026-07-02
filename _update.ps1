# CryptoBot auto-update — runs before each start (called by START_EVERYTHING.bat).
# Downloads the newest code from GitHub and copies it over this folder.
# NEVER touches your personal data: andx_credentials.json, portfolios, stats,
# learning files, the Chrome login profile and .venv are not in the download,
# so they survive every update. No internet? The bot just starts as-is.
$ErrorActionPreference = 'Stop'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$root   = Split-Path -Parent $MyInvocation.MyCommand.Path
$zipUrl = 'https://github.com/nickraju443/cryptobot/archive/refs/heads/master.zip'
$tmp    = Join-Path $env:TEMP ('cryptobot_update_' + [guid]::NewGuid().ToString('N'))

try {
    New-Item -ItemType Directory -Force -Path $tmp | Out-Null
    $zip = Join-Path $tmp 'update.zip'
    Invoke-WebRequest -Uri $zipUrl -OutFile $zip -UseBasicParsing -TimeoutSec 30
    Expand-Archive -Path $zip -DestinationPath $tmp -Force
    $src = Get-ChildItem -Path $tmp -Directory | Where-Object { $_.Name -like 'cryptobot-*' } | Select-Object -First 1
    if (-not $src) { throw 'unexpected zip layout' }

    $reqPath = Join-Path $root 'requirements.txt'
    $before = ''
    if (Test-Path $reqPath) { $before = (Get-FileHash $reqPath).Hash }

    # START_EVERYTHING.bat is excluded: a .bat must never be overwritten
    # while it is running (cmd reads it line by line from disk).
    robocopy $src.FullName $root /E /XF START_EVERYTHING.bat /XD .git /NFL /NDL /NJH /NJS /NP | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "robocopy failed with code $LASTEXITCODE" }

    $after = ''
    if (Test-Path $reqPath) { $after = (Get-FileHash $reqPath).Hash }
    $venvPy = Join-Path $root '.venv\Scripts\python.exe'
    if ($before -and $after -and ($before -ne $after) -and (Test-Path $venvPy)) {
        Write-Host '        new packages required - installing (1-2 min)...'
        & $venvPy -m pip install -r $reqPath --quiet
    }
    Write-Host '        bot is up to date.'
    exit 0
} catch {
    Write-Host "        update check skipped ($($_.Exception.Message))"
    Write-Host '        starting with the current version.'
    exit 0
} finally {
    if (Test-Path $tmp) { Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue }
}
