#!/bin/bash
mkdir -p ser;   ../debug/stereomatch       a.png b.png
mkdir -p par;   ../out/stereopar         a.png b.png
mkdir -p sergh; ../debug/stereomatch-ghost a.png b.png
mkdir -p pargh; ../out/stereopar-ghost   a.png b.png

test_images() {
    diff "$1/$3" "$2/$3" > /dev/null
    if [[ $? != 0 ]]; then
        echo "problem with image $3 ($1 - $2)"
    fi
}

for f in ser/*; do
    name=$(basename $f)
    echo "testing $name"
    test_images ser par $name
    test_images sergh pargh $name
done
