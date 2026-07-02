@echo off
REM CryptoBot startup
setlocal

cd /d "%~dp0"

REM Use the same venv as SRI MATA if present, otherwise fall back to system python.
if exist "..\Bot\.venv\Scripts\python.exe" (
    set PY=..\Bot\.venv\Scripts\python.exe
) else if exist ".venv\Scripts\python.exe" (
    set PY=.venv\Scripts\python.exe
) else (
    set PY=python
)

echo Using: %PY%
%PY% app.py
