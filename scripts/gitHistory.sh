#!/usr/bin/env bash

repo_dir="your-repo"
cd "$repo_dir"

echo "date,lines" > fsharp_loc.csv

# iterate over commits (one per day)
git log --reverse --pretty=format:'%ad %H' --date=short | \
awk '!seen[$1]++' | \
while read date commit; do
    git checkout -q $commit

    loc=$(cloc --include-lang=F# --csv --quiet . | awk -F, '/F#/ {print $5}')
    loc=${loc:-0}

    echo "$date,$loc" >> fsharp_loc.csv
done

git checkout -q main