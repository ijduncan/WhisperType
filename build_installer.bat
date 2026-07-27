@echo off
REM Rebuild the WhisperType installer from source.
REM   1. PyInstaller freezes the app
REM   2. Inno Setup wraps it into WhisperType-Setup.exe
REM
REM By default this builds the SLIM installer (~75 MB): the speech model and
REM the CUDA libraries are downloaded during installation instead of bundled.
REM Pass "bundled" to build the fully offline installer (~2.3 GB) instead.
REM Output is written OUTSIDE OneDrive to avoid file-sync locks.
setlocal
cd /d "%~dp0"

set VENV_PY=%~dp0.venv\Scripts\python.exe
set ISCC=%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe
set BUILD=C:\WhisperTypeBuild
set DIST=%BUILD%\slimdist

if /I "%~1"=="bundled" (
    echo *** BUNDLED build: including model + CUDA libraries ***
    set WHISPERTYPE_BUNDLE=1
    set DIST=%BUILD%\dist
    if not exist "%~dp0models\medium.en\model.bin" (
        echo Fetching model to bundle...
        "%VENV_PY%" -c "from faster_whisper import download_model; download_model('medium.en', output_dir=r'models/medium.en')"
    )
)

echo === [1/2] Building app with PyInstaller ===
"%VENV_PY%" -m PyInstaller --noconfirm --clean ^
    --distpath "%DIST%" --workpath "%BUILD%\work" whispertype.spec
if errorlevel 1 goto :err

echo === [2/2] Building installer with Inno Setup ===
"%ISCC%" /DDistDir="%DIST%\WhisperType" installer.iss
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
