@echo off
echo ==========================================
echo    ArtNet Pixel Console - Windows Build
echo ==========================================

echo [1/3] Checking Python...
python --version
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH.
    pause
    exit /b
)

echo [2/3] Installing Dependencies...
pip install -r requirements.txt
pip install pyinstaller

echo [3/3] Building Executable...
pyinstaller build.spec --clean --noconfirm

echo.
echo ==========================================
echo    Build Complete!
echo    Executable is located in: dist\ArtNetPixelConsole.exe
echo ==========================================
pause
