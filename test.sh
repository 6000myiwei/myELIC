#!/bin/bash

data_path="/home/zhangyiwei/hdd/dataset"

# python -m compressai.utils.eval_model checkpoint \
# /hdd/zyw/ImageDataset/test -a elic_ada \
# -p pretrained/elic_ada/6/checkpoint_best_loss.pth.tar \
# --cuda

python -m compressai.utils.eval_model checkpoint \
$data_path/test -a minen18-ada \
-p pretrained/minen18-ada/1/checkpoint_best_loss.pth.tar \
--cuda

# python -m compressai.utils.eval_model checkpoint \
# /hdd/zyw/ImageDataset/test -a tinylic \
# -p ./pretrained/tinylic/3/checkpoint_best_loss.pth.tar \
# --cuda
