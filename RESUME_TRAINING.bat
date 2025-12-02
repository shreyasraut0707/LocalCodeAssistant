@echo off
echo ================================================================
echo Resume CodeAlpaca Training
echo ================================================================
echo.
echo This will resume training from the last saved checkpoint
echo.
echo Press Ctrl+C anytime to safely stop training
echo It will auto-save every 500 steps
echo.
pause

python scripts\train_codealpaca.py

if %errorlevel% neq 0 (
    echo.
    echo Training stopped or failed
    echo Don't worry! Progress is saved.
    echo Run this script again to resume.
    pause
    exit /b 1
)

echo.
echo ================================================================
echo Training complete!
echo ================================================================
pause
