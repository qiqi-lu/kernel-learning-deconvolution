#!/bin/bash

echo job start
module load anaconda/2022.10
module load cuda/11.8

source activate pytorchx

echo evaluate on each model
python -u evaluate.py

echo evaluate on RL network
# python -u evaluate_rl.py

echo done