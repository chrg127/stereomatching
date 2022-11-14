#!/bin/bash

test_program() {
    time_sum=0
    num_tests=$(ls imgs | wc -l)
    for dir in imgs/*; do
        time=$($1       $dir/a.png $dir/b.png | awk '{ print $15}')
        time_sum=$(echo "$time_sum + $time" | bc)
        echo "$dir: $time"
    done
    echo "scale=4; $time_sum / $num_tests" | bc -l
}

# test_program ../timing/stereomatch
test_program ../timing/stereomatch-ghost
# test_program ../timing/stereopar
# test_program ../timing/stereopar-ghost
