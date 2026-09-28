@echo off
rem Kiln - starts the web server (and the engine it supervises).
rem Open http://localhost:8090/ here, or the LAN URL it prints from your phone.
title Kiln
cd /d "%~dp0"
if not exist "%~dp0engine\build\kiln-engine.exe" (
  echo Kiln isn't set up yet. Run setup.bat first.
  pause
  exit /b 1
)
set "NODE=%~dp0third_party\node\node.exe"
if not exist "%NODE%" set "NODE=node"
if "%NODE%"=="node" where node >nul 2>nul || (
  echo Node.js was not found. Run setup.bat first.
  pause
  exit /b 1
)
rem optional local additions (config\autostart.bat, not part of Kiln)
if exist "%~dp0config\autostart.bat" call "%~dp0config\autostart.bat"
"%NODE%" "%~dp0server\server.js"
if errorlevel 1 pause
