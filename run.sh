#!/bin/bash
# Example run sequence (make executable: chmod +x run.sh)
# 1) install requirements: pip install -r requirements.txt
# 2) prepare data: place HR training images in data/HR_train and HR test images in data/HR_test
# 3) train and infer for the scales you want

SCALE=2

python train.py --scale $SCALE
python sr_infer.py --scale $SCALE
python evaluate.py
