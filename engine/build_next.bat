@echo off
rem Builds to build\next\ so the running server's engine is untouched; swap in with the server stopped.
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set CUDA=%~dp0..\third_party\cuda
set PATH=%CUDA%\bin;%PATH%
cd /d %~dp0
rem optional modules: the face detector and the upscaler are compiled in when their sources exist
set EXTRA=
if exist src\detect.cu set EXTRA=%EXTRA% -DKILN_FACE src\detect.cu
if exist src\upscale.cu set EXTRA=%EXTRA% -DKILN_UPSCALE src\upscale.cu
if not exist build\next mkdir build\next
rem GPU code: Turing (sm_75), Ampere (sm_80, also runs on RTX 30), Ada (sm_89, RTX 40), Blackwell (sm_120, RTX 50),
rem plus PTX that newer GPUs compile on first start. The engine picks tensor-core or CUDA-core kernels at startup.
set ARCH=-gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_120,code=sm_120 -gencode arch=compute_80,code=compute_80
nvcc -O3 -std=c++17 %ARCH% --threads 0 -Xcompiler "/EHsc /O2 /utf-8" -o build\next\kiln-engine.exe ^
  src\main.cpp src\safetensors.cpp src\kernels.cu src\te.cu src\dit.cu src\train.cu src\tagger.cu src\image_io.cpp src\vae.cu src\pipeline.cu src\graph.cpp src\sdops.cu src\sdxl.cu src\clip.cu src\sdvae.cu src\sdpipe.cu %EXTRA% -lcublas || exit /b 1
echo BUILD OK
