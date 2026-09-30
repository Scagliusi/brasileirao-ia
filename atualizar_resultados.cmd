@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Ambiente virtual nao encontrado. Consulte COMO_USAR.md.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -u pipeline.py completo --temporada 2026 --simulacoes 1000 --seed 42
if errorlevel 1 (
  echo A atualizacao nao foi concluida. Confira a mensagem acima. Resultados anteriores foram preservados.
) else (
  echo Atualizacao publicada. No painel, clique em Recarregar resultados.
)
pause
