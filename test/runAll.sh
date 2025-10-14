#!/bin/zsh
# remember to turn on debugging in Image
dirs=(**/*.fsproj(:h))
for i in $dirs; do
  echo $i
  pushd $i
  dotnet build
  /usr/bin/time env DYLD_LIBRARY_PATH="$(pwd)/lib" dotnet run debug 30 --verbosity q > ../$i.out 2>&1; \
  popd
done;
