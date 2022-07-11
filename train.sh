#!/bin/bash

python examples/train_elic.py -m elic \
-d /hdd/zyw/ImageDataset --epochs 40 \
--quality-level 8 \
--num-workers 8 \
--lambda 0.045 \
-lr 2e-4 --aux-learning-rate 2e-3 \
--batch-size 32 --cuda --save \
--checkpoint ./pretrained/elic/8/checkpoint.pth.tar