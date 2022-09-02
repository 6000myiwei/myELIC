#!/bin/bash

python examples/train_vq.py -m elic_vq \
-d /hdd/zyw/ImageDataset --epochs 320 \
--quality-level 6 \
--num-workers 8 \
--lambda 0.045 \
-lr 4e-3 \
--batch-size 8 --cuda --save \
# --checkpoint pretrained/elic/best/lambda0.045.pth.tar \
# --optimizer