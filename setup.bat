@echo off
rem Kiln setup - gets everything Kiln needs into this folder:
rem   Node.js (a portable copy, only if a recent one isn't installed), the engine (prebuilt, or
rem   built from source with --build), the CUDA runtime, and the Anima models + turbo LoRA,
rem   plus the 4x upscaler and the face detector (skip those with --no-extras).
rem Safe to re-run: finished steps are skipped and interrupted downloads resume.
rem With no arguments it asks what to get. Or: setup.bat all ^| prereqs ^| models ^| anima te vae turbo
rem upscaler face engine cublas, setup.bat --list (links + folders, to download by hand),
rem setup.bat --from "D:\ComfyUI\models" (reuse files you already have).
setlocal
title Kiln setup
cd /d "%~dp0"

where curl >nul 2>nul || (echo curl.exe was not found. Kiln needs Windows 10 version 1803 or newer. & goto fail)
where tar >nul 2>nul || (echo tar.exe was not found. Kiln needs Windows 10 version 1803 or newer. & goto fail)

set "NODE="
if exist "third_party\node\node.exe" set "NODE=%~dp0third_party\node\node.exe"
if not defined NODE for /f "tokens=1 delims=v." %%v in ('node -p process.versions.node 2^>nul') do if %%v GEQ 20 set "NODE=node"
if not defined NODE call :get_node || goto fail

"%NODE%" "%~dp0tools\setup.js" %*
if errorlevel 1 goto fail
echo.
pause
exit /b 0

:fail
echo.
echo Setup did not finish. Fix the problem above and run setup.bat again.
pause
exit /b 1

:get_node
set NODEVER=v24.21.0
set NODEPKG=node-%NODEVER%-win-x64
echo Node.js 20 or newer was not found; downloading a portable Node.js %NODEVER% into third_party\node
if not exist third_party mkdir third_party
curl -fL --retry 3 -o "third_party\%NODEPKG%.zip" "https://nodejs.org/dist/%NODEVER%/%NODEPKG%.zip" || exit /b 1
curl -fsSL --retry 3 -o "third_party\SHASUMS256.txt" "https://nodejs.org/dist/%NODEVER%/SHASUMS256.txt" || exit /b 1
set "WANT="
set "GOT="
for /f "tokens=1" %%h in ('findstr /c:" %NODEPKG%.zip" "third_party\SHASUMS256.txt"') do set "WANT=%%h"
for /f "skip=1 tokens=*" %%h in ('certutil -hashfile "third_party\%NODEPKG%.zip" SHA256') do if not defined GOT set "GOT=%%h"
set "GOT=%GOT: =%"
del "third_party\SHASUMS256.txt"
if /i not "%GOT%"=="%WANT%" (
  echo The Node.js download failed its checksum check.
  del "third_party\%NODEPKG%.zip"
  exit /b 1
)
tar -xf "third_party\%NODEPKG%.zip" -C third_party || exit /b 1
del "third_party\%NODEPKG%.zip"
if exist "third_party\node" rmdir /s /q "third_party\node"
ren "third_party\%NODEPKG%" node || exit /b 1
set "NODE=%~dp0third_party\node\node.exe"
exit /b 0
