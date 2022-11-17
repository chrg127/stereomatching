#!/bin/bash

test_program() {
    time_sum=0
    num_tests=$(ls test/imgs | wc -l)
    for dir in test/imgs/*; do
        time=$($1       $dir/a.png $dir/b.png | awk '{ print $15}')
        time_sum=$(echo "$time_sum + $time" | bc)
        echo -n "$time,"
    done
    echo "scale=6; $time_sum / $num_tests" | bc -l
}

times=$(test_program $1)
echo $times
