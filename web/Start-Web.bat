@echo off
start "DI server" /min powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0server.ps1"
timeout /t 1 /nobreak >nul
start "" "http://localhost:8787/"
