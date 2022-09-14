#!/bin/bash

# python examples/train_elic.py -m elic_ada \
# -d /hdd/zyw/ImageDataset --epochs 140 \
# --quality-level 6 \
# --num-workers 8 \
# --lambda 0.045 \
# -lr 1e-4 --aux-learning-rate 1e-3 \
# --batch-size 16 --cuda --save \
# --checkpoint pretrained/elic_ada/6/mu_pooling_2.pth.tar \
# # --optimizer


python examples/train_elic.py -m minen18-ada \
-d /hdd/zyw/ImageDataset --epochs 500 \
--quality-level 6 \
--num-workers 8 \
--lambda 0.045 \
-lr 1e-4 --aux-learning-rate 1e-3 \
--batch-size 8 --cuda --save \
--checkpoint pretrained/minen18-ada/6/checkpoint.pth.tar \
--optimizer