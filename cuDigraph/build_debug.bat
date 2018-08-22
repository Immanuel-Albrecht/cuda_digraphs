call "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat"

setlocal
cd /d %~dp0
cd build
MSBuild cuDigraph.sln /t:"Build" /p:Platform="x64" /p:Configuration="Debug"
