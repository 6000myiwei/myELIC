#!/bin/bash

python examples/train_elic.py -m elic_ada \
-d /hdd/zyw/ImageDataset --epochs 140 \
--quality-level 6 \
--num-workers 8 \
--lambda 0.045 \
-lr 1e-4 --aux-learning-rate 1e-3 \
--batch-size 16 --cuda --save \
--checkpoint pretrained/elic_ada/6/sigma_judge_half.pth.tar \
# --optimizer