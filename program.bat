@echo off
chcp 65001 >nul
title AI-Talib Inspector Suite
cd /d "%~dp0"
set "PY=.venv\Scripts\python.exe"

if not exist "%PY%" (
  echo [1/3] Creating virtual environment...
  py -3.12 -m venv .venv 2>nul || py -3.11 -m venv .venv 2>nul || py -3 -m venv .venv 2>nul || python -m venv .venv
  if not exist "%PY%" ( echo Python 3.10+ is required. Install it from python.org & pause & exit /b 1 )
)

"%PY%" -c "import PySide6, pyqtgraph, pandas, scipy, sklearn, openpyxl, requests" 2>nul
if errorlevel 1 (
  echo [2/3] Installing libraries ^(first run only^)...
  "%PY%" -m pip install --upgrade pip -q
  "%PY%" -m pip install -r requirements.txt || ( echo Install failed & pause & exit /b 1 )
)

"%PY%" -c "import keras" 2>nul
if errorlevel 1 if not exist ".venv\.no_tf" (
  echo [2/3] Installing TensorFlow/Keras ^(optional, may take a while^)...
  "%PY%" -m pip install tensorflow -q || ( echo TensorFlow unavailable - scikit-learn fallback will be used & echo.>.venv\.no_tf )
)

echo [3/3] Starting...
if exist ".venv\Scripts\pythonw.exe" ( start "" ".venv\Scripts\pythonw.exe" main_app.py ) else ( "%PY%" main_app.py )
exit /b 0
