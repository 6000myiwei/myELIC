#!/bin/bash

# python -m compressai.utils.eval_model checkpoint \
# /hdd/zyw/ImageDataset/test -a elic_ada \
# -p pretrained/elic_ada/6/checkpoint_best_loss.pth.tar \
# --cuda

python -m compressai.utils.eval_model checkpoint \
/hdd/zyw/ImageDataset/test -a minen18-checkerboard \
-p pretrained/minen18-checkerboard/6/checkpoint_best_loss.pth.tar \
--cuda

# python -m compressai.utils.eval_model checkpoint \
# /hdd/zyw/ImageDataset/test -a tinylic \
# -p ./pretrained/tinylic/3/checkpoint_best_loss.pth.tar \
# --cuda