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

data_path="/home/zhangyiwei/hdd/dataset"

python examples/train_elic.py -m elic_ada \
-d $data_path --epochs 80 \
--quality-level 7 \
--lambda 0.074 \
--num-workers 8 \
-lr 1e-4 --aux-learning-rate 1e-3 \
--batch-size 16 --cuda --save \
--checkpoint pretrained/elic_ada/7/checkpoint.pth.tar \
# --optimizer