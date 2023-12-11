#!/bin/bash

if [ ! -d "results" ]; then
    mkdir results
fi

for i in {0..9}
do
    if [ ! -d "results/ex0$i" ]; then
        mkdir "results/ex0$i"
    fi
done

if [ ! -d "results/ex10" ]; then
    mkdir results/ex10
fi

for i in {0..9}
do
    if [[ $i -ne 6 ]] && [[ $i -ne 8 ]]; then
        python3 ex0$i/test.py > results/ex0$i/result.txt
    fi
done

python3 ex06/multivariate_linear_model.py > results/ex06/result.txt
python3 ex08/polynomial_train.py > results/ex08/result.txt
python3 ex10/space_avocado.py > results/ex10/result.txt
