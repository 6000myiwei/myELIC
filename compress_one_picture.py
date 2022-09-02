#%%
import torch
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
import torch.nn.functional as F

from compressai.models import ELIC
from compressai.models.elic import CodecStageEnum

from compressai.utils.eval_model.__main__ import psnr
from pytorch_msssim import ms_ssim, ssim

net = ELIC().cuda()
net.eval()
net.stage = CodecStageEnum.Check

checkpoint = torch.load('pretrained/elic/best/lambda0.015.pth.tar')
net.load_state_dict(checkpoint['state_dict'])

net.update(force=True)
#%%
img = Image.open('/home/zyw/original.jpg').convert("RGB")
x = ToTensor()(img).cuda()
x = x.unsqueeze(0)
B,C,H,W = x.shape
x1 = x[..., :H//2, :W//2]
x2 = x[..., H//2:, :W//2]
x3 = x[..., :H//2, W//2:]
x4 = x[..., H//2:, W//2:]

xs = [x1, x2, x3, x4]
# %%
import time
res = []
strings = []
bpp = []
for x1 in xs:
    h, w = x1.size(2), x1.size(3)
    p = 256  # maximum 6 strides of 2, and window size 4 for the smallest latent fmap: 4*2^6=256
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    x_padded = F.pad(
        x1,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )

    start = time.time()
    out_enc = net.compress(x_padded)
    enc_time = time.time() - start
    
    start = time.time()
    out_dec = net.decompress(out_enc['strings'], out_enc['shape'])
    dec_time = time.time() - start
    
    y1 = out_enc['y']
    y2 = out_dec['y']
    print((y1==y2).all())

    num_pixels = x1.size(0) * x1.size(2) * x1.size(3)
    out_dec["x_hat"] = F.pad(
        out_dec["x_hat"], (-padding_left, -padding_right, -padding_top, -padding_bottom)
    )
    print(sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels)
    print(psnr(x1, out_dec["x_hat"]))
    print("dec time: ", dec_time)
    bpp.append(sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels)
    strings.append(out_enc["strings"])
    res.append(out_dec["x_hat"])


#%%
# # %%
# out = torch.zeros_like(x)
# out[..., :H//2, :W//2] = res[0].clamp_(0,1)
# out[..., H//2:, :W//2] = res[1].clamp_(0,1)
# out[..., :H//2, W//2:] = res[2].clamp_(0,1)
# out[..., H//2:, W//2:] = res[3].clamp_(0,1)
# print(psnr(out, x))
# print(ms_ssim(out, x,data_range=1.0).item())
# print(ssim(out, x,data_range=1.0).item())

# # %%
# ToPIL = ToPILImage()
# pic = ToPIL(out.squeeze_())
# pic.save('/home/zyw/rec.png')

# %%
