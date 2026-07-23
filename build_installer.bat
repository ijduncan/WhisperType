@echo off
REM Rebuild the WhisperType installer from source.
REM   1. PyInstaller freezes the app (with GPU DLLs + bundled model)
REM   2. Inno Setup wraps it into WhisperType-Setup.exe
REM Output is written OUTSIDE OneDrive to avoid file-sync locks.
setlocal
cd /d "%~dp0"

set VENV_PY=%~dp0.venv\Scripts\python.exe
set ISCC=%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe
set BUILD=C:\WhisperTypeBuild

echo === [1/2] Building app with PyInstaller ===
"%VENV_PY%" -m PyInstaller --noconfirm --clean ^
    --distpath "%BUILD%\dist" --workpath "%BUILD%\work" whispertype.spec
if errorlevel 1 goto :err

echo === [2/2] Building installer with Inno Setup ===
"%ISCC%" installer.iss
if errorlevel 1 goto :err

echo.
echo Done. Installer: %BUILD%\Output\WhisperType-Setup.exe
pause
exit /b 0

:err
echo.
echo Build failed. See messages above.
pause
exit /b 1
