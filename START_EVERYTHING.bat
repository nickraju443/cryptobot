@echo off
title CryptoBot One-Click Start
setlocal

REM ============================================================
REM   CryptoBot -- One-Click Start
REM ============================================================
REM   1. Opens the andX Chrome window (log in there -- one time)
REM   2. First run only: installs Python packages automatically
REM   3. Starts the bot and opens the dashboard in your browser
REM   4. Puts a "CryptoBot" icon on your Desktop for next time
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
echo.
echo   You are probably running this from INSIDE the zip file.
echo   Windows only pulls out this one file, so nothing works.
echo.
echo   FIX:
echo     1. Right-click the zip file  -^>  "Extract All..."
echo     2. Click Extract.
echo     3. Open the NEW folder it created.
echo     4. Double-click START_EVERYTHING.bat in THERE.
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
REM (dev machine only -- shared venv next to this folder)
if exist "%~dp0..\Bot\.venv\Scripts\python.exe" set "PYEXE=%~dp0..\Bot\.venv\Scripts\python.exe"
if defined PYEXE goto env_ready

REM ---- First run: find a REAL Python install ----
REM The "py" launcher works even if "Add to PATH" was not checked.
set "PYBOOT="
py -3 --version >nul 2>nul
if not errorlevel 1 set "PYBOOT=py -3"
if defined PYBOOT goto bootstrap
REM Fall back to python on PATH. The fake Microsoft Store "python"
REM stub fails this version check, so it cannot fool us.
python --version >nul 2>nul
if not errorlevel 1 set "PYBOOT=python"
if defined PYBOOT goto bootstrap
echo.
echo  ============================================================
echo   ERROR: Python is not installed on this computer.
echo.
echo   NOTE: Do NOT install Python from the Microsoft Store.
echo.
echo   1. Go to  https://www.python.org/downloads/
echo   2. Download and run the installer.
echo   3. CHECK THE BOX "Add python.exe to PATH" on the first
echo      screen, then click Install Now.
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
echo  If a Microsoft Store window just opened: that is a fake Python.
echo  Install the real one from https://www.python.org/downloads/
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
echo  ERROR: Package install failed.
echo  Check your internet connection and run this file again.
echo  Still failing? Take a photo of this window and send it over.
echo.
pause
exit /b 1
:pkgs_ok
"%PYEXE%" -m playwright install chromium

:env_ready
REM ---- Desktop icon (created once; harmless if it already exists) ----
powershell -NoProfile -ExecutionPolicy Bypass -Command "$d=[Environment]::GetFolderPath('Desktop');$p=Join-Path $d 'CryptoBot.lnk';if(!(Test-Path $p)){$w=New-Object -ComObject WScript.Shell;$s=$w.CreateShortcut($p);$s.TargetPath='%~dp0START_EVERYTHING.bat';$s.WorkingDirectory='%~dp0';if(Test-Path '%~dp0bot.ico'){$s.IconLocation='%~dp0bot.ico,0'};$s.Description='Start CryptoBot + andX';$s.Save();Write-Output 'ICON_CREATED'}" 2>nul | findstr ICON_CREATED >nul
if not errorlevel 1 echo  A "CryptoBot" icon was added to your Desktop -- use that from now on.

echo.
echo  [3/3] Starting the bot...
echo        Dashboard: http://localhost:5002  (opens automatically)
echo.
echo    - Keep the andX Chrome window OPEN. The bot trades through it.
echo    - Close THIS window to stop the bot.
echo.

REM Open the dashboard 12s from now (detached, while the bot boots)
start "" /min cmd /c "timeout /t 12 /nobreak >nul & start http://localhost:5002"

"%PYEXE%" app.py

echo.
echo  Bot has stopped. Press any key to close.
pause >nul
