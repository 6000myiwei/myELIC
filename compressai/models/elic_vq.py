#%%
import math
from operator import ilshift
import torch
import torch.nn as nn
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import conv1x1,conv3x3, CrossMaskedConv2d, StackResBottleneck
from compressai.layers import Quantizator_RT, STEQuant
from timm.models.layers import trunc_normal_
from enum import Enum
from compressai.models.utils import conv, deconv, update_registered_buffers, CheckerboardDemux, CheckerboardMux

from compressai.mcquic.loss import BasicRate

from compressai.models.elic import CodecStageEnum
from compressai.models.elic import YEncoder, YDecoder, UnevenChannelSlice, ChannelARParamTransform
from compressai.mcquic.modules.quantizer import UMGMQuantizer
from compressai.mcquic.nn.blocks import ResidualBlock, ResidualBlockWithStride, ResidualBlockShuffle, AttentionBlock
from compressai.mcquic.consts import Consts


# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64
SKIP_INDEX = -1
SCALE_BOUND = SCALES_MIN

def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))



class YEncoder(nn.Module):
    def __init__(self, in_channels=3, N=192, M=320, num_rsb=3) -> None:
        super().__init__()
        self.g_a = nn.Sequential(
            conv(in_channels, N, 5, 2),
            StackResBottleneck(N, num_rsb),
            conv(N, N, 5, 2),
            StackResBottleneck(N, num_rsb),
            AttentionBlock(N),
            conv(N, M, 5, 2),
            # StackResBottleneck(N, num_rsb),
            # conv(N, M, 5, 2),
            AttentionBlock(M)
        )
    
    def forward(self, x):
        return self.g_a(x)


class YDecoder(nn.Module):
    def __init__(self, in_channels=3, N=192, M=320, num_rsb=3) -> None:
        super().__init__()
        self.g_s = nn.Sequential(
            AttentionBlock(M),
            deconv(M, N, 5, 2),
            # StackResBottleneck(N, num_rsb),
            # deconv(N, N, 5, 2),
            AttentionBlock(N),
            StackResBottleneck(N, num_rsb),
            deconv(N, N, 5, 2),
            StackResBottleneck(N, num_rsb),
            deconv(N, in_channels, 5, 2)
        )
    
    def forward(self, x):
        return self.g_s(x)


class ChannelForwardTransform(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        '''already consider the double the channle
        '''
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
    
    def forward(self, former_quant, x):
        out = x - former_quant
        return out

class ChannelBackwardTransform(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        '''already consider the double the channle
        '''
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
    
    def forward(self, former_quant, x_quant):
        out = x_quant + former_quant
        return out


def UMGMComponents(channel):
    components = {
        "latentStageEncoder": lambda: nn.Sequential(
            ResidualBlockWithStride(channel, channel, groups=1),
            # GroupSwishConv2D(channel, 3, groups=1),
            ResidualBlock(channel, channel, groups=1),
            AttentionBlock(channel, groups=1),
        ),
        "quantizationHead": lambda: nn.Sequential(
            ResidualBlock(channel, channel, groups=1),
            AttentionBlock(channel, groups=1),
            conv3x3(channel, channel)
            # convs.conv1x1(channel, channel, groups=1)
            # GroupSwishConv2D(channel, channel, groups=1)
        ),
        "latentHead": lambda: nn.Sequential(
            ResidualBlock(channel, channel, groups=1),
            AttentionBlock(channel, groups=1),
            conv3x3(channel, channel)
            # convs.conv1x1(channel, channel, groups=1)
        ),
        "restoreHead": lambda: nn.Sequential(
            AttentionBlock(channel, groups=1),
            ResidualBlock(channel, channel, groups=1),
            ResidualBlockShuffle(channel, channel, groups=1)
        ),
        "dequantizationHead": lambda: nn.Sequential(
            AttentionBlock(channel, groups=1),
            conv3x3(channel, channel),
            ResidualBlock(channel, channel, groups=1),
        ),
        "sideHead": lambda: nn.Sequential(
            AttentionBlock(channel, groups=1),
            conv3x3(channel, channel),
            ResidualBlock(channel, channel, groups=1),
        ),
    }
    return components


class ContextIterator(nn.Module):
    
    inner_forward_cls = ChannelForwardTransform
    inner_backward_cls = ChannelBackwardTransform
    components = UMGMComponents
    m_per_group = [4, 4, 4, 2, 2]
    k_list = [8192, 2048, 512]
    
    def __init__(self, k_size=5, groups=[16,16,32,64,None]) -> None:
        super().__init__()
        assert groups[-1] is not None
        self.groups = groups
        self.n_groups = len(groups)
        group_start = [sum(groups[0:i]) for i in range(self.n_groups)]
        self.rate = BasicRate(1e-7)
        
        self._sub_modules_channel_forward = nn.ModuleList(
            self.inner_forward_cls(group_start[i+1], groups[i+1])
            for i in range(self.n_groups - 1)
        )
        
        self._sub_modules_channel_backward = nn.ModuleList(
            self.inner_backward_cls(group_start[i+1], groups[i+1])
            for i in range(self.n_groups - 1)
        )
        
        self._sub_modules_vq = nn.ModuleList(
            UMGMQuantizer(c, self.m_per_group[i], self.k_list, permutationRate=0.0, components=UMGMComponents(c))
            for i,c in enumerate(self.groups)
        )
    
    def forward(self, y_slice_iter):
        formerGroup = []
        decoder_in = []
        codes = []
        rate = []
        
        y_slices = next(y_slice_iter)
        # formerLevel, codes, logits
        out = self._sub_modules_vq[0](y_slices)
        current_rate = self.rate(out[2], self._sub_modules_vq[0].Codebooks)
        
        formerGroup.append(out[0])
        decoder_in.append(out[0])
        codes.append(out[1])
        rate.append(current_rate)
        
        for y_next_slice, channel_sub_forward, channel_sub_backward, vq_sub in  \
            zip(y_slice_iter, self._sub_modules_channel_forward,  \
                self._sub_modules_channel_backward,self._sub_modules_vq[1:]):
            
            next_level_in = channel_sub_forward(torch.cat(formerGroup, dim=1), y_next_slice)
            out = vq_sub(next_level_in)
            current_rate = self.rate(out[2], vq_sub.Codebooks)
            next_decoder_in = channel_sub_backward(torch.cat(formerGroup, dim=1), out[0])
            
            codes.append(out[1])
            formerGroup.append(out[0])
            decoder_in.append(next_decoder_in)
            rate.append(current_rate)
        return torch.cat(decoder_in, dim=1), formerGroup, codes, rate

class ELIC_VQ(nn.Module):
    def __init__(self,N=128, M=192) -> None:
        super().__init__() 
        self.groups = [12,12,24,48,None]
        self.groups[-1] = M - sum(self.groups[:-1])
        self.n_groups = len(self.groups)
        
        self.g_a = YEncoder(N=N, M=M)
        self.g_s = YDecoder(N=N, M=M)
        
        self.sliceModel = UnevenChannelSlice(M, self.groups)
        self.contextiter = ContextIterator(groups=self.groups)
    
    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = 128
        M = 192
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net
    
    @property
    def VQs(self) -> UMGMQuantizer:
        return self.contextiter._sub_modules_vq
    
    @property
    def Channel_Sub_forward(self):
        return self.contextiter._sub_modules_channel_forward
    
    @property
    def Channel_Sub_backward(self):
        return self.contextiter._sub_modules_channel_backward
    
    def reAssignCodebook(self) -> torch.Tensor:
        return [quantizer.reAssignCodebook() for quantizer in self.VQs]
    
    def syncCodebook(self):
        return [quantizer.syncCodebook() for quantizer in self.VQs]
    
    @property
    def CodeUsage(self):
        return [torch.cat(list((freq > Consts.Eps).flatten() for freq in quantizer.NormalizedFreq)).float().mean() 
                for quantizer in self.VQs]
    
    def refresh(self):
        propotion = self.reAssignCodebook()
        return propotion
        
    def forward(self, x):
        y = self.g_a(x)
        y_slice_iter = iter(self.sliceModel(y))
        y_hat, _, codes, rate= self.contextiter(y_slice_iter)
        x_hat = self.g_s(y_hat)
        
        return{
            "x_hat":x_hat,
            "likelihoods":{"y": sum(rate)}
        }
    
    @torch.inference_mode()
    def compress(self, x):
        formerGroup = []
        decoder_in = []
        binaies = []
        CodeSizes = []
        
        y = self.g_a(x)
        y_slice_iter = iter(self.sliceModel(y))
        y_slices = next(y_slice_iter)
        # return codes, binaries, codeSize
        out = self.contextiter._sub_modules_vq[0].compress(y_slices)
        y_slices_hat = self.contextiter._sub_modules_vq[0].decompress(out[1], out[2])
        
        formerGroup.append(y_slices_hat)
        decoder_in.append(y_slices_hat)
        binaies.append(out[1])
        CodeSizes.append(out[2])
        
        for y_next_slice, channel_sub_forward, channel_sub_backward, vq_sub in  \
            zip(y_slice_iter, self.contextiter._sub_modules_channel_forward,  \
                self.contextiter._sub_modules_channel_backward,self.contextiter._sub_modules_vq[1:]):
            
            next_level_in = channel_sub_forward(torch.cat(formerGroup, dim=1), y_next_slice)
            
            out = vq_sub.compress(next_level_in)
            y_slices_hat = vq_sub.decompress(out[1], out[2])
            next_decoder_in = channel_sub_backward(torch.cat(formerGroup, dim=1), y_slices_hat)
            
            binaies.append(out[1])
            CodeSizes.append(out[2])
            formerGroup.append(y_slices_hat)
            decoder_in.append(next_decoder_in)
        
        return {
            "strings": binaies,
            "shape" : CodeSizes,
            "y" : torch.cat(decoder_in, dim=1)
        }
    
    @torch.inference_mode()
    def decompress(self, strings, shapes):
        formerGroup = []
        decoder_in = []
        
        y_slices_hat = self.VQs[0].decompress(strings[0], shapes[0])
        formerGroup.append(y_slices_hat)
        decoder_in.append(y_slices_hat)
        for i in range(1, len(self.VQs)):
            y_slices_hat = self.VQs[i].decompress(strings[i], shapes[i])
            next_decoder_in = self.Channel_Sub_backward[i-1](torch.cat(formerGroup, dim=1), y_slices_hat)
            formerGroup.append(y_slices_hat)
            decoder_in.append(next_decoder_in)
        
        x_hat = self.g_s(torch.cat(decoder_in, dim=1))
        return {
            "x_hat" : x_hat,
            "y" : torch.cat(decoder_in, dim=1)
        }
#%%
if __name__ == "__main__":
    from PIL import Image
    from torchvision.transforms import ToTensor
    
    # x = torch.rand((2,3,256,256)).cuda()
    img = Image.open('/hdd/zyw/ImageDataset/kodak/kodim01.png').convert("RGB")
    x = ToTensor()(img).cuda()
    x = x.unsqueeze(0)
    elic_vq = ELIC_VQ().cuda()
    code = elic_vq.compress(x)
    rec = elic_vq.decompress(code["strings"], code["shape"])
    
    out = elic_vq(x)
    y1 = code["y"]
    y2 = rec["y"]
    total_bytes = 0                
    strings = code['strings']
    for s in strings:
        for i in s:
            for j in i:
                total_bytes += len(j)
        

    print(total_bytes * 8 / 768 / 512)
    print((y1 == y2).all())
    propotion = elic_vq.refresh()
    print([p.item() for p in propotion])

    # # 速度测试
    # import time
    # enc_time = []
    # dec_time = []
    # stage = CodecStageEnum.TEST
    # count = 20
    # for _ in range(count):  
    #     t1 = time.process_time()
    #     code = elic_vq.compress(x)
    #     t2 = time.process_time()
    #     enc_time.append(t2 - t1)
        
    #     t1 = time.process_time()
    #     rec = elic_vq.decompress(code['strings'], code['shape'])
    #     t2 = time.process_time()
    #     dec_time.append(t2 - t1)
    # enc_time = sum(enc_time) / count
    # dec_time = sum(dec_time) / count
    # print("encode time: ", enc_time * 1000)
    # print("decode time: ", dec_time * 1000)
#%%