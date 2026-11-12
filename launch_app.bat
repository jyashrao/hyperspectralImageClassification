@echo off
title Hyperspectral AI Research Platform
cd /d "C:\Users\jyash\Desktop\hyperspectralImageClassification"
echo ==============================================================
echo Launching Hyperspectral AI Band Selection & Classification App
echo ==============================================================
echo.
echo Starting Streamlit web server on http://localhost:8501 ...
python -m streamlit run app.py --server.port 8501
pause
