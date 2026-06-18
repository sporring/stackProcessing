This directory holds platform-native helper libraries used by StackProcessing.

Build the current low-level helper from the repository root:

```bash
bash lowlevel/build.sh
```

On Windows, use:

```powershell
powershell -ExecutionPolicy Bypass -File .\lowlevel\build.ps1
```

Expected helper names are:

[LINUX]
liblowlevel.so

[MACOSX]
liblowlevel.dylib

[WINDOWS]
lowlevel.dll

The helper depends on single-precision FFTW. On Linux and macOS, install FFTW
through the system package manager before building. On Windows, install FFTW
with vcpkg or place the matching FFTW runtime DLL beside lowlevel.dll.
