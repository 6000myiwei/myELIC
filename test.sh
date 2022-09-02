#!/bin/bash

python -m compressai.utils.eval_model checkpoint \
/hdd/zyw/ImageDataset/test -a elic_ada \
-p pretrained/elic_ada/6/sigma_judge_half.pth.tar \
--cuda


# python -m compressai.utils.eval_model checkpoint \
# /hdd/zyw/ImageDataset/test -a tinylic \
# -p ./pretrained/tinylic/3/checkpoint_best_loss.pth.tar \
# --cuda