@echo off
REM Stop only the CryptoBot python process. Find the one bound to PORT 5002.
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":5002" ^| findstr "LISTENING"') do (
    echo Killing PID %%a (CryptoBot on :5002)
    taskkill /F /PID %%a
)
