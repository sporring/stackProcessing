#!/bin/zsh
for i in *.fsproj; do \
  echo $i && \
  dotnet clean --verbosity q $i && \
  dotnet build --verbosity q $i && \
  /usr/bin/time -l env DYLD_LIBRARY_PATH="$PWD/lib" \
    dotnet run debug 30 --verbosity q --project $i > $i.out 2>&1; \
  done;