# Copyright (c) 2021-2022, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import math
import random
import shutil
import sys
import os
import time
import logging
from datetime import datetime
from black import out

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms

from compressai.datasets import ImageFolder
from compressai.models.elic import CodecStageEnum
from compressai.zoo import image_models


training_stage = [9999, 220, 240]
double_lambda_trick_epoch = 240

class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        out["mse_loss"] = self.mse(output["x_hat"], target)
        out["loss"] = self.lmbda * 255**2 * out["mse_loss"] + out["bpp_loss"]

        return out

class RateDistortionLossTwoPath(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
    
    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W
        weight = [0.5, 0.5, 1]
        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)) * w
            for likelihoods, w in zip(output["likelihoods"].values(), weight)
        )
        out["mse_loss"] = self.mse(output["x_hat"], target)
        out["loss"] = self.lmbda * 255**2 * out["mse_loss"] + out["bpp_loss"]

        return out



class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def init(args):
    base_dir = f'./pretrained/{args.model}/{args.quality_level}/'
    os.makedirs(base_dir, exist_ok=True)

    return base_dir


def setup_logger(log_dir):
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    log_file_handler = logging.FileHandler(log_dir, encoding='utf-8')
    log_file_handler.setFormatter(log_formatter)
    root_logger.addHandler(log_file_handler)

    log_stream_handler = logging.StreamHandler(sys.stdout)
    log_stream_handler.setFormatter(log_formatter)
    root_logger.addHandler(log_stream_handler)

    logging.info('Logging file is %s' % log_dir)


def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""
    
    opt_cls = optim.Adam
    try:
        from apex.optimizers import FusedAdam
        logging.info('using apex FusedAdam')
        opt_cls = FusedAdam
    except ImportError:
        logging.info('cannot load apex FusedAdam, using default Pytorch implementation')
    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = opt_cls(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
    )
    aux_optimizer = opt_cls(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=args.aux_learning_rate,
    )
    return optimizer, aux_optimizer


def train_one_epoch(
    model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm
):
    model.train()
    device = next(model.parameters()).device

    for i, d in enumerate(train_dataloader):
        d = d.to(device)

        optimizer.zero_grad()
        aux_optimizer.zero_grad()
        
        ##TODO input training stage to the model
        out_net = model(d)

        out_criterion = criterion(out_net, d)
        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        if i*len(d) % 2000 == 0:
            logging.info(
                f'[epoch: {epoch}] | '
                f'[{i*len(d)}/{len(train_dataloader.dataset)}] | '
                # f" ({100. * i / len(train_dataloader):.0f}%)]"
                f'Loss: {out_criterion["loss"].item():.3f} | '
                f'MSE loss: {out_criterion["mse_loss"].item():.5f} | '
                f'PSNR: {10 * torch.log10(1 / out_criterion["mse_loss"]).item():.3f} |'
                f'Bpp loss: {out_criterion["bpp_loss"].item():.4f} | '
                f"Aux loss: {aux_loss.item():.2f}"
            )


def test_epoch(epoch, test_dataloader, model, criterion):
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
        f"Test epoch {epoch}: Average losses: "
        f"Loss: {loss.avg:.3f} | "
        f"MSE loss: {mse_loss.avg:.5f} | "
        f"PSNR: {10 * torch.log10(1 / mse_loss.avg).item():.3f} | "
        f"Bpp loss: {bpp_loss.avg:.4f} | "
        f"Aux loss: {aux_loss.avg:.2f}\n"
    )

    return loss.avg


def save_checkpoint(state, is_best, base_dir, filename="checkpoint.pth.tar"):
    torch.save(state, base_dir+filename)
    if is_best:
        shutil.copyfile(base_dir+filename, base_dir+"checkpoint_best_loss.pth.tar")


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-m",
        "--model",
        default="elic",
        choices=image_models.keys(),
        help="Model architecture (default: %(default)s)",
    )
    parser.add_argument(
        "-d", "--dataset", type=str, required=True, help="Training dataset"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=400,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=4,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "-q",
        "--quality-level",
        type=int,
        default=3,
        help="Quality level (default: %(default)s)",
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=1e-2,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate",
        default=1e-3,
        type=float,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument(
        "--gpu-id",
        type=str,
        default=0,
        help="GPU ids (default: %(default)s)",
    )
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument(
        "--seed", type=float, help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument(
        '--name', 
        default=datetime.now().strftime('%Y-%m-%d_%H_%M_%S'), 
        type=str,
        help='Result dir name', 
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    parser.add_argument("--optimizer", action="store_true", help="Whether load optimizer")
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)
    base_dir = init(args)
    double_lambda_value = args.lmbda / 2
    ## set to 0.015 for low bitrate model
    # double_lambda_value = 0.015

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
    
    setup_logger(base_dir + '/' + time.strftime('%Y%m%d_%H%M%S') + '.log')
    msg = f'======================= {args.name} ======================='
    logging.info(msg)
    for k in args.__dict__:
        logging.info(k + ':' + str(args.__dict__[k]))
    logging.info('=' * len(msg))

    train_transforms = transforms.Compose([
        transforms.RandomCrop(args.patch_size), 
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor()
        ]
    )

    test_transforms = transforms.Compose(
        [transforms.CenterCrop(args.patch_size), transforms.ToTensor()]
    )

    train_dataset = ImageFolder(args.dataset, split="train", transform=train_transforms)
    test_dataset = ImageFolder(args.dataset, split="test", transform=test_transforms)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    if device == "cuda":
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    net = image_models[args.model](quality=int(args.quality_level), stage=CodecStageEnum.TRAIN)
    net = net.to(device)

    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)

    optimizer, aux_optimizer = configure_optimizers(net, args)
    # lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[training_stage[-1]], gamma=0.1)
    lr_scheduler_aux = optim.lr_scheduler.MultiStepLR(aux_optimizer, milestones=[training_stage[-1]], gamma=0.1)

    last_epoch = 0
    if args.checkpoint:  # load from previous checkpoint
        logging.info("Loading "+str(args.checkpoint))
        checkpoint = torch.load(args.checkpoint, map_location=device)
        net.load_state_dict(checkpoint["state_dict"], strict=False)
    
    if args.optimizer:
        logging.info("Loading optimizer")
        last_epoch = checkpoint["epoch"] + 1
        
        # checkpoint["optimizer"]['param_groups'][0]['lr'] = args.learning_rate
        # checkpoint["aux_optimizer"]['param_groups'][0]['lr'] = args.aux_learning_rate
        
        optimizer.load_state_dict(checkpoint["optimizer"])
        aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        
        ## Temp Hack lr scheduler
        # del checkpoint['lr_scheduler']['milestones']
        # checkpoint['lr_scheduler']['_last_lr'] = args.learning_rate
        
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        
        # checkpoint['lr_scheduler']['base_lrs'] = args.aux_learning_rate
        # checkpoint['lr_scheduler']['_last_lr'] = args.aux_learning_rate
        
        # TODO add aux lr scheduler loader
        lr_scheduler_aux.load_state_dict(checkpoint["lr_scheduler"])
    
    lmbda_value = double_lambda_value if last_epoch >= double_lambda_trick_epoch[0] and last_epoch < double_lambda_trick_epoch[1] else args.lmbda
    criterion = RateDistortionLossTwoPath(lmbda=lmbda_value)
    
    best_loss = float("inf")
    for epoch in range(last_epoch, args.epochs):
        logging.info('======Current epoch %s ======'%epoch)
        logging.info(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        logging.info(f"Aux Learning rate: {aux_optimizer.param_groups[0]['lr']}")
        logging.info(f"using lambda: {lmbda_value}")
        if epoch >= training_stage[0] and epoch < training_stage[1]:
            criterion = RateDistortionLoss(lmbda=lmbda_value)
            net.stage =  CodecStageEnum.TRAIN_ONEPATH
            logging.info("Training One Path")
            
        elif epoch >= training_stage[1]:
            criterion = RateDistortionLoss(lmbda=lmbda_value)
            net.stage = CodecStageEnum.TRAIN2
            logging.info("Training stage 2")
        
        if epoch >= double_lambda_trick_epoch[0] and epoch < double_lambda_trick_epoch[1]:
            lmbda_value = double_lambda_value
            criterion.lmbda = lmbda_value
        else:
            lmbda_value = args.lmbda
            criterion.lmbda = lmbda_value
        
        train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
        )
        loss = test_epoch(epoch, test_dataloader, net, criterion)
        lr_scheduler.step()
        lr_scheduler_aux.step()

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if args.save:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "lr_scheduler_aux" : lr_scheduler_aux.state_dict()
                },
                is_best,
                base_dir,
                filename = "checkpoint.pth.tar"
                # filename="checkpoint.pth.tar" if epoch != double_lambda_trick_epoch[1] - 1 else "change.pth.tar"
            )


if __name__ == "__main__":
    main(sys.argv[1:])
