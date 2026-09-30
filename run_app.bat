@echo off
title Customer Churn Prediction App
echo Starting Customer Churn Prediction System...
cd /d "%~dp0"
python -m streamlit run app.py
pause