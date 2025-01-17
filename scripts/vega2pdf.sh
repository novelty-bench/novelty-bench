#!/bin/bash

set -Eeuo pipefail

for f in $(find plots/ -name "*.json")
do
    dir=$(dirname $f)
    fname=$(basename $f .json)
    vl-convert vl2pdf -i $f -o $dir/$fname.pdf &
done