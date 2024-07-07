#!/bin/bash

echo job start
module load anaconda/2022.10
module load cuda/11.8

source activate pytorchx

python -u generate_synthetic_data_3d.py