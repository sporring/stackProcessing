Download and extract the SimpleITK CSharp Library from the release section of https://github.com/SimpleITK/SimpleITK?tab=readme-ov-file for your platform here, e.g., download and unzip SimpleITK-2.5.3-CSharp-macosx-10.9-anycpu.zip and move the dll and dylib files here. 
Do NOT change any file names. We expect following the filenames:

[SHARED]
SimpleITKCSharpManaged.dll

[LINUX]
libSimpleITKCSharpNative.so

[MACOSX]
libSimpleITKCSharpNative.dylib

[WINDOWS]
SimpleITKCSharpNative.dll

On MACOS (e.g., https://github.com/SimpleITK/SimpleITK/releases/tag/v2.5.3), you probably have to move the library out of quarantine by:
xattr -dr com.apple.quarantine libSimpleITKCSharpNative.dylib

StackProcessing also uses a small native median helper for C++ std::nth_element slab kernels.
Build it from the repository root with:

native/StackProcessing.NativeMedian/build.sh

The build writes the platform-specific library into this directory. Expected names are:

[LINUX]
libspnth.so

[MACOSX]
libspnth.dylib

[WINDOWS]
spnth.dll
