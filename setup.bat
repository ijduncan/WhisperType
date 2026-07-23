@echo off
REM One-time setup: create a virtual environment and install dependencies.
setlocal
set PY="%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if not exist %PY% set PY=python

echo Creating virtual environment...
%PY% -m venv "%~dp0.venv"
if errorlevel 1 goto :err

echo Installing dependencies (this can take a few minutes)...
"%~dp0.venv\Scripts\python.exe" -m pip install --upgrade pip
"%~dp0.venv\Scripts\python.exe" -m pip install -r "%~dp0requirements.txt"
if errorlevel 1 goto :err

echo.
echo Setup complete. Double-click run.bat (or run-hidden.vbs) to start WhisperType.
pause
exit /b 0

:err
echo.
echo Setup failed. See the messages above.
pause
exit /b 1
