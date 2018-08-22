call "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat"

setlocal
cd /d %~dp0

rmdir /S /Q build

mkdir build
cd build


cmake .. -A x64 -DCUDA_TOOLKIT_ROOT_DIR="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v9.0"

MSBuild cuDigraph.sln /t:"Build" /p:Platform="x64" /p:Configuration="Debug"

MSBuild cuDigraph.sln /t:"Build" /p:Platform="x64" /p:Configuration="Release"