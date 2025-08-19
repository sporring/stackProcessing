#!/bin/zsh
echo "" > gather.csv
for i in *.fsproj.out; do \
  line=$(tail -n 1 $i)
  words=(${=line})
  peakMemory=$words[1]
  line="$(tail -n 18 -- "$i" | head -n 1)"
  words=(${=line})
  userTime=$words[3]
  echo $i, $peakMemory, $userTime >> gather.csv
  done;