@echo off
cd /d "%~dp0"
C:\Python311\python.exe -m streamlit run app.py --server.port 8501 --server.headless false
pause
