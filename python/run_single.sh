#!/bin/bash

echo job start
module load anaconda/2022.10
module load cuda/11.8

source activate pytorchx

python -u main_v3.py