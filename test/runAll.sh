#!/bin/zsh
dirs=(**/*.fsproj(:h))
for i in $dirs; do
  echo $i
  pushd $i
  /usr/bin/time env DYLD_LIBRARY_PATH="$(pwd)/lib" dotnet run debug 18 --verbosity q > ../$i.out 2>&1; \
  popd
done;
