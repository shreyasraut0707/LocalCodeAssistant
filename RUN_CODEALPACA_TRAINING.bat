@echo off
echo ================================================================
echo CodeAlpaca-20k Fine-Tuning - Complete Pipeline
echo ================================================================
echo.

echo Step 1: Downloading CodeAlpaca-20k Dataset...
echo ----------------------------------------------------------------
python scripts\download_codealpaca.py
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Failed to download dataset
    pause
    exit /b 1
)

echo.
echo ================================================================
echo Step 2: Preparing Dataset for Training...
echo ================================================================
python scripts\prepare_codealpaca.py
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Failed to prepare dataset
    pause
    exit /b 1
)

echo.
echo ================================================================
echo Step 3: Starting Fine-Tuning...
echo ================================================================
echo.
echo This will take approximately 2-4 hours on GTX 1650
echo You can safely minimize this window, but don't close it!
echo.
pause

python scripts\train_codealpaca.py

if %errorlevel% neq 0 (
    echo.
    echo ERROR: Training failed
    pause
    exit /b 1
)

echo.
echo ================================================================
echo SUCCESS! Fine-tuning complete!
echo ================================================================
echo.
echo Your model is ready in: models\codealpaca-finetuned\final
echo.
pause
