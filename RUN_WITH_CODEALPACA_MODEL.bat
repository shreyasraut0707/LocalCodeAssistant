@echo off
echo ================================================================
echo Starting AI Code Assistant with CodeAlpaca Model
echo ================================================================
echo.
echo Using newly fine-tuned CodeAlpaca-20k model!
echo Model location: models/codealpaca-finetuned/final/
echo.
echo The app will open in your browser at: http://localhost:8501
echo.
pause

streamlit run app_with_openrouter.py

pause
