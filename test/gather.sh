#!/bin/zsh
echo "name, peakMemory, estimatedPeak, userTime" > gather.csv
for i in *.fsproj.out; do \
  line=$(tail -n 1 $i)
  words=(${=line})
  peakMemory=$words[1]
  line="$(tail -n 18 -- "$i" | head -n 1)"
  words=(${=line})
  userTime=$words[3]
  line=$(grep "Running pipeline" $i)
  words=(${=line})
  estimate=$words[7]
  echo ${i%.fsproj.out}, $peakMemory, $estimate, $userTime >> gather.csv
  done;