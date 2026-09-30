@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Ambiente virtual nao encontrado. Consulte COMO_USAR.md.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m streamlit run app/streamlit_app.py --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false
pause
