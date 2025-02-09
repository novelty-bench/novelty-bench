#!/bin/bash

set -Eeuo pipefail

for f in $(find plots/ -name "*.json")
do
    dir=$(dirname $f)
    fname=$(basename $f .json)
    vl-convert vl2jpeg -i $f -o $dir/$fname.jpg --scale 10.0 &
done