from statistics import mode
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import transforms

from compressai.datasets import ImageFolder
from compressai.models.elic import CodecStageEnum
from compressai.zoo import image_models
import sys

from examples.train_elic import RateDistortionLoss, RateDistortionLossTwoPath,AverageMeter

import logging

model = 'elic'

dataset_path = '/hdd/zyw/ImageDataset'
# checkpoint_path = './pretrained/elic/8/lambda0.9.pth.tar'
# checkpoint_path = './pretrained/elic/8/lambda0.16.pth.tar'
# checkpoint_path = './pretrained/elic/7/checkpoint_best_loss.pth.tar'
checkpoint_path = './pretrained/elic/6/checkpoint.pth.tar'
quality_level = 8
stage = CodecStageEnum.TRAIN2

lambda_value = 0.0225

def setup_logger():
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    log_stream_handler = logging.StreamHandler(sys.stdout)
    log_stream_handler.setFormatter(log_formatter)
    root_logger.addHandler(log_stream_handler)


def test_epoch(test_dataloader, model, criterion):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    aux_loss = AverageMeter()

    with torch.no_grad():
        for d in test_dataloader:
            d = d.to(device)
            out_net = model(d)
            out_criterion = criterion(out_net, d)

            aux_loss.update(model.aux_loss())
            bpp_loss.update(out_criterion["bpp_loss"])
            loss.update(out_criterion["loss"])
            mse_loss.update(out_criterion["mse_loss"])

    logging.info(
        f"Loss: {loss.avg:.3f} | "
        f"MSE loss: {mse_loss.avg:.5f} | "
        f"PSNR: {10 * torch.log10(1 / mse_loss.avg).item():.3f} | "
        f"Bpp loss: {bpp_loss.avg:.4f} | "
        f"Aux loss: {aux_loss.avg:.2f}\n"
    )

    return loss.avg

def real_test(test_dataloader, model):
    model.eval()
    model.update()
    distortion = nn.MSELoss()
    device = next(model.parameters()).device

    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()

    with torch.no_grad():
        for d in test_dataloader:
            d = d.to(device)
            N, _, H, W = d.size()
            code = model.compress(d)
            rec = model.decompress(code['strings'], code['shape'])
            total_bytes = 0                
            strings = code['strings']
        
            for s in strings:
                    if isinstance(s, list):
                        for i in s:
                            total_bytes += len(i)
            else:
                total_bytes += len(i)

            bpp = total_bytes * 8 / H / W / N
            mse = nn.MSELoss()(rec['x_hat'], d)
            
            bpp_loss.update(bpp)
            mse_loss.update(mse)

    logging.info(
        f"MSE loss: {mse_loss.avg:.5f} | "
        f"PSNR: {10 * torch.log10(1 / mse_loss.avg).item():.3f} | "
        f"Bpp loss: {bpp_loss.avg:.4f} | \n"
    )

setup_logger()

test_transforms = transforms.Compose([transforms.ToTensor()])

test_dataset = ImageFolder(dataset_path, split="test", transform=test_transforms)

device = "cuda"

test_dataloader = DataLoader(
    test_dataset,
    batch_size=1,
    num_workers=1,
    shuffle=False,
    pin_memory=(device == "cuda"),
)

net = image_models[model](quality=int(quality_level), stage=stage)
net = net.to(device)


logging.info("Loading "+str(checkpoint_path))
checkpoint = torch.load(checkpoint_path, map_location=device)
net.load_state_dict(checkpoint["state_dict"], strict=True)


logging.info("begin Testing")
if stage == CodecStageEnum.TRAIN:
    criterion = RateDistortionLossTwoPath(lmbda=lambda_value)
else:
    criterion = RateDistortionLoss(lmbda=lambda_value)
test_epoch(test_dataloader, net, criterion)
# real_test(test_dataloader, net)