#!/bin/bash

python examples/train_elic.py -m elic \
-d /hdd/zyw/ImageDataset --epochs 80 \
--quality-level 3 \
--num-workers 8 \
--lambda 75e-4 \
-lr 1e-4 --aux-learning-rate 1e-3 \
--batch-size 16 --cuda --save \
--checkpoint ./pretrained/elic/4/lambda0.015.pth.tar \
# --optimizer