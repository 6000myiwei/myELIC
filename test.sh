#!/bin/bash

# python -m compressai.utils.eval_model checkpoint \
# /hdd/zyw/ImageDataset/test -a elic \
# -p ./pretrained/elic/6/lambda.0.045.pth.tar \
# --cuda


python -m compressai.utils.eval_model checkpoint \
/hdd/zyw/ImageDataset/test -a tinylic \
-p ./pretrained/tinylic/3/checkpoint_best_loss.pth.tar \
--cuda