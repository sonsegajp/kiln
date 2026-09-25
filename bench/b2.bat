@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set CUDA=C:\Users\hyper\Kiln\third_party\cuda
set PATH=%CUDA%\bin;%CUDA%\bin\x64;%PATH%
cd /d C:\Users\hyper\Kiln\bench
where cl
nvcc -O3 -arch=sm_75 -o ceiling.exe ceiling.cu -lcublas
echo nvcc exit %errorlevel%
dir /b
