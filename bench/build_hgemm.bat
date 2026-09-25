@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set CUDA=%~dp0..\third_party\cuda
set PATH=%CUDA%\bin;%CUDA%\bin\x64;%PATH%
cd /d %~dp0
nvcc -O3 -arch=sm_75 -Xptxas -v -o hgemm_test.exe hgemm_test.cu -lcublas || exit /b 1
hgemm_test.exe
