@echo off
rem usage: build_i8.bat [test|real|all]   (default all)
rem   test -> i8_test.exe  : dp4a GEMM speed + exactness vs hgemm (standalone)
rem   real -> i8_real.exe  : accuracy on real Anima activations (links engine sources read-only)
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set CUDA=%~dp0..\third_party\cuda
set PATH=%CUDA%\bin;%CUDA%\bin\x64;%PATH%
cd /d %~dp0
set WHAT=%1
if "%WHAT%"=="" set WHAT=all
if "%WHAT%"=="test" goto test
if "%WHAT%"=="real" goto real
:test
nvcc -O3 -std=c++17 -arch=sm_75 -Xptxas -v -Xcompiler "/EHsc /O2 /utf-8" -o i8_test.exe i8_test.cu || exit /b 1
if "%WHAT%"=="test" goto done
:real
set E=..\engine\src
nvcc -O3 -std=c++17 -arch=sm_75 -Xcompiler "/EHsc /O2 /utf-8" -o i8_real.exe i8_real.cu %E%\safetensors.cpp %E%\kernels.cu %E%\vae.cu -lcublas || exit /b 1
:done
echo BUILD OK
