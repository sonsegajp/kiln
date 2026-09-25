@echo off
rem Builds the upscaler harness (bench\upscale_test.exe). Run it with ..\engine\build on PATH (CUDA DLLs).
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set CUDA=%~dp0..\third_party\cuda
set PATH=%CUDA%\bin;%PATH%
cd /d %~dp0
nvcc -O3 -std=c++17 -arch=sm_75 -Xcompiler "/EHsc /O2 /utf-8" -o upscale_test.exe ^
  upscale_test.cu ..\engine\src\upscale.cu ..\engine\src\kernels.cu ..\engine\src\safetensors.cpp -lcublas || exit /b 1
echo BUILD OK
