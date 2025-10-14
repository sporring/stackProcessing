#!/bin/zsh
echo "name, estimatedPeakMemory, peakMemory, peakImages, userTime" > gather.csv
for i in *.out; do \
  line=$(grep sink $i)
  words=(${=line})
  estimatedPeakMemory=$words[7]

  line=$(grep " KB / " $i | tail -n 1)
  words=(${=line})
  peakMemory=$words[4]
  peakImages=$words[8]

  line="$(tail -n 1 -- "$i")"
  words=(${=line})
  userTime=$words[3]
  line=$(grep "Running pipeline" $i)
  words=(${=line})
  estimate=$words[7]
  echo ${i%.fsproj.out}, $estimatedPeakMemory, $peakMemory, $peakImages, $userTime >> gather.csv
  done;