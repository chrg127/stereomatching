#!/bin/bash
mkdir -p serial;   cd serial;   ../out/stereomatch ../img/a.png ../img/b.png; cd ..
mkdir -p parallel; cd parallel; ../out/stereopar   ../img/a.png ../img/b.png; cd ..
for f in serial/*; do
    name=$(basename $f)
    echo "testing $name"
    diff "serial/$name" "parallel/$name" > /dev/null
    if [[ $? != 0 ]]; then
        echo "problem with image $name"
    fi
done
