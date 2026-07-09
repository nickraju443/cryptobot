@echo off
title CryptoBot One-Click Start
setlocal

REM ============================================================
REM   CryptoBot -- One-Click Start
REM ============================================================
REM   1. Opens the andX Chrome window (log in there -- one time)
REM   2. First run only: installs Python packages automatically
REM   3. Starts the bot and opens the dashboard in your browser
REM
REM   Leave the Chrome window OPEN while the bot runs.
REM   Close THIS window to stop the bot.
REM ============================================================

REM Anchor everything to this .bat file's folder (portable)
cd /d "%~dp0"

REM ---- Guard: was this run from INSIDE the zip? ----
if exist "%~dp0app.py" goto files_ok
echo.
echo  ============================================================
echo   STOP -- the bot's files are not next to this launcher.
echo   You are probably running this from INSIDE the zip file.
echo.
echo   FIX: right-click the zip -^> "Extract All..." -^> Extract,
echo   open the NEW folder, and run START_EVERYTHING.bat in THERE.
echo  ============================================================
echo.
pause
exit /b 1
:files_ok

REM ---- Find Chrome (system, 32-bit, or per-user install) ----
set "CHROME_EXE=C:\Program Files\Google\Chrome\Application\chrome.exe"
if not exist "%CHROME_EXE%" set "CHROME_EXE=C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"
if not exist "%CHROME_EXE%" set "CHROME_EXE=%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"
if exist "%CHROME_EXE%" goto chrome_ok
echo.
echo  ERROR: Google Chrome not found.
echo  Install it from https://www.google.com/chrome/ and run this again.
echo.
pause
exit /b 1
:chrome_ok

echo.
echo  [1/3] Opening the andX Chrome window...
echo        FIRST RUN: log into andX with YOUR account in that window.
echo        It remembers you after that. LEAVE THE WINDOW OPEN.
echo.

start "" "%CHROME_EXE%" --remote-debugging-port=9222 --user-data-dir="%~dp0_andx_chrome_profile" --no-first-run --no-default-browser-check https://platform.andx.one/

REM ---- Use an existing environment if there is one ----
set "PYEXE="
if exist "%~dp0.venv\Scripts\python.exe" set "PYEXE=%~dp0.venv\Scripts\python.exe"
if defined PYEXE goto env_ready
if exist "%~dp0..\Bot\.venv\Scripts\python.exe" set "PYEXE=%~dp0..\Bot\.venv\Scripts\python.exe"
if defined PYEXE goto env_ready

REM ---- First run: find a REAL Python install ----
set "PYBOOT="
py -3 --version >nul 2>nul
if not errorlevel 1 set "PYBOOT=py -3"
if defined PYBOOT goto bootstrap
python --version >nul 2>nul
if not errorlevel 1 set "PYBOOT=python"
if defined PYBOOT goto bootstrap
echo.
echo  ============================================================
echo   ERROR: Python is not installed on this computer.
echo   Do NOT install Python from the Microsoft Store.
echo.
echo   1. Go to  https://www.python.org/downloads/
echo   2. Run the installer.
echo   3. CHECK "Add python.exe to PATH" on the first screen,
echo      then click Install Now.
echo   4. Run this file again.
echo  ============================================================
echo.
pause
exit /b 1

:bootstrap
echo  [2/3] FIRST RUN -- installing the bot's packages.
echo        This takes 5-15 minutes and only happens once.
echo        Let it finish. Do not close this window.
echo.
%PYBOOT% -m venv "%~dp0.venv"
if not errorlevel 1 goto venv_made
echo.
echo  ERROR: Could not create the Python environment.
echo  Install the real Python from https://www.python.org/downloads/
echo  (check "Add python.exe to PATH"), then run this again.
echo.
pause
exit /b 1
:venv_made
set "PYEXE=%~dp0.venv\Scripts\python.exe"
"%PYEXE%" -m pip install --upgrade pip
"%PYEXE%" -m pip install -r "%~dp0requirements.txt"
if not errorlevel 1 goto pkgs_ok
echo.
echo  ERROR: Package install failed. Check your internet and run again.
echo.
pause
exit /b 1
:pkgs_ok
"%PYEXE%" -m playwright install chromium

:env_ready
echo.
echo  [3/3] Starting the bot...
echo        Dashboard: http://localhost:5002  (opens automatically)
echo.
echo    - Keep the andX Chrome window OPEN. The bot trades through it.
echo    - Close THIS window to stop the bot.
echo.

start "" /min cmd /c "timeout /t 12 /nobreak >nul & start http://localhost:5002"

REM ---- Self-healing run loop: if the bot ever exits (crash, price-feed
REM hiccup, etc.) it relaunches automatically after 5s, so it keeps trading
REM 24/7. To STOP the bot on purpose, close this window (the X) or press
REM Ctrl+C twice — that ends the loop.
:runloop
"%PYEXE%" app.py
echo.
echo  ============================================================
echo   Bot stopped at %date% %time%.  Auto-restarting in 5s...
echo   (To stop for good: close this window now.)
echo  ============================================================
timeout /t 5 /nobreak >nul
goto runloop
