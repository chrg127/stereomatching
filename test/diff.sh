#!/bin/bash
mkdir -p ser;   ../debug/stereomatch       a.png b.png
mkdir -p par;   ../debug/stereopar         a.png b.png
mkdir -p sergh; ../debug/stereomatch-ghost a.png b.png
mkdir -p pargh; ../debug/stereopar-ghost   a.png b.png

test_images() {
    diff "$1/$3" "$2/$3" > /dev/null
    if [[ $? != 0 ]]; then
        echo "problem with image $3 ($1 - $2)"
    fi
}

for f in ser/*; do
    name=$(basename $f)
    echo "testing $name (ser - par)"
    test_images ser par $name
    echo "testing $name (sergh - pargh)"
    test_images sergh pargh $name
done
