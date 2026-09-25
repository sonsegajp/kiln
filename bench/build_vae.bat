@echo off
rem Builds bench\vae_test.exe (VAE harness, loads only the VAE). Run it with ..\engine\build on PATH
rem for the CUDA DLLs. Optional %1: a directory holding alternative vae.cu / vaeconv.cuh / models.h.
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set CUDA=%~dp0..\third_party\cuda
set PATH=%CUDA%\bin;%PATH%
cd /d %~dp0
set SRC=%~1
if "%SRC%"=="" set SRC=..\engine\src
nvcc -O3 -std=c++17 -arch=sm_75 -Xcompiler "/EHsc /O2 /utf-8" -I"%SRC%" -I..\engine\src -o vae_test.exe ^
  vae_test.cu "%SRC%\vae.cu" ..\engine\src\kernels.cu ..\engine\src\safetensors.cpp -lcublas -lwindowscodecs -lole32 || exit /b 1
echo BUILD OK
