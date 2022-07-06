#%%
import math
from tokenize import group
from turtle import forward, shape
import torch
import torch.nn as nn
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import conv1x1,conv3x3, AttentionBlock, CrossMaskedConv2d
from compressai.layers import Quantizator_RT, STEQuant
from timm.models.layers import trunc_normal_
from enum import Enum
from compressai.utils.bench.codecs import Codec

from utils import conv, deconv, update_registered_buffers, Demultiplexer, Multiplexer

# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64

def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


class CodecStageEnum(Enum):
    TRAIN = 1
    TRAIN2 = 2
    VALID = 3
    TEST = 4
    COMPRESS = 5
    DECOMPRESS = 6

class ResBottleneck(nn.Module):
    """Simple residual unit."""

    def __init__(self, N):
        super().__init__()
        self.conv = nn.Sequential(
            conv1x1(N, N // 2),
            nn.ReLU(inplace=True),
            conv3x3(N // 2, N // 2),
            nn.ReLU(inplace=True),
            conv1x1(N // 2, N),
        )

    def forward(self, x):
        identity = x
        out = self.conv(x)
        out += identity
        return out

class StackResBottleneck(nn.Module):
    def __init__(self, N=192, num_rsb=3) -> None:
        super().__init__()
        self._layers = nn.ModuleList([ResBottleneck(N) for _ in range(num_rsb)])

    def forward(self, x):
        for layer in self._layers:
            x = layer(x)
        return x

class YEncoder(nn.Module):
    def __init__(self, in_channels=3, N=192, M=320, num_rsb=3) -> None:
        super().__init__()
        self.g_a = nn.Sequential(
            conv(in_channels, N, 5, 2),
            StackResBottleneck(N, num_rsb),
            conv(N, N, 5, 2),
            StackResBottleneck(N, num_rsb),
            AttentionBlock(N),
            conv(N, N, 5, 2),
            StackResBottleneck(N, num_rsb),
            conv(N, M, 5, 2),
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
            StackResBottleneck(N, num_rsb),
            deconv(N, N, 5, 2),
            AttentionBlock(N),
            StackResBottleneck(N),
            deconv(N, N, 5, 2),
            StackResBottleneck(N),
            deconv(N, 3, 5, 2)
        )
    
    def forward(self, x):
        return self.g_s(x)
    
class ZEncoder(nn.Module):
    def __init__(self, N=192, M=320) -> None:
        super().__init__()
        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
            conv(N, N, stride=2, kernel_size=5),
            nn.ReLU(inplace=True),
            conv(N, N, stride=2, kernel_size=5),
        )
    
    def forward(self, x):
        return self.h_a(x)

class ZDecoder(nn.Module):
    def __init__(self, N=192, M=320) -> None:
        super().__init__()
        self.h_s = nn.Sequential(
            deconv(N, M, stride=2, kernel_size=5),
            nn.ReLU(inplace=True),
            deconv(M, M * 3 // 2, stride=2, kernel_size=5),
            nn.ReLU(inplace=True),
            conv(M * 3 // 2, M * 2, stride=1, kernel_size=3),
        )

    def forward(self,x):
        return self.h_s(x)


class ContextChAR(nn.Module):
    def __init__(self, in_channels, out_channles, groups) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channles = out_channles
        
        self.groups = groups
    
    def forward(self, x):
        '''
        channel-wise context ref
        the shape is [0, 16, 16, 32, 64]
        '''
        shape_in = x.shape
        c = shape_in[1]
        zero = torch.zeros(shape_in[0], self.groups[0], shape_in[2], shape_in[3])
        zero = zero.to(x.device)
        x = x[:, :-self.groups[-1],...]
        x = torch.cat([zero, x], 1)
        return x

class UnevenChannelSlice(nn.Module):
    """
    Unevenly Channel slice
    """
    
    def __init__(self, out_channles, groups=[16,16,32,64,192]) -> None:
        super().__init__()
        self.groups = groups
        self.n_groups = len(self.groups)
        # self.ckbd = CheckerboardSliceContext(in_channles, out_channles * 2, k_size=5 ,n_groups=self.n_groups)
        # self.chAr = ContextChAR(in_channles, out_channles, self.groups)
        self.group_start = [sum(self.groups[0:i]) for i in range(self.n_groups)]
    
    def forward(self,x):
        # x1 = self.ckbd(x)
        # x2 = self.chAr(x)
        # x1s = [x1[:, self.group_start[i] * 2 : (self.group_start[i] + self.groups[i]) * 2, ...] for i in range(self.n_groups)]
        # x2s = [x2[:, self.group_start[i] : self.group_start[i] + self.groups[i], ...] for i in range(self.n_groups)]
        # xs = []
        # for i in range(self.n_groups):
        #     xs += [x1s[i], x2s[i]]        
        xs = [x[:, self.group_start[i] : self.group_start[i] + self.groups[i], ...] for i in range(self.n_groups)]
        return xs

class CheckerboardContext(nn.Module):
    def __init__(self, *maskconv_args, **maskconv_kwargs) -> None:
        super().__init__()
        self.cross_conv = CrossMaskedConv2d(*maskconv_args, **maskconv_kwargs)

    def forward(self, x, stage):
        if stage == CodecStageEnum.TRAIN:
            checkerboard = self.cross_conv(x)
            anchor = torch.zeros(x.shape[0], x.shape[1] * 2, x.shape[2], x.shape[3]).to(x.device)
            return checkerboard, anchor
        else:
            checkerboard = self.cross_conv(x)
            return checkerboard

class CheckerboardSliceAlignMux(nn.Module):
    def forward(self, checkerboard, anchor):
        '''
        checkerboard:non_anchor tensor
        anchor:anchor tensor
        '''
        assert checkerboard.shape == anchor.shape
        anchor = torch.clone(anchor)
        
        anchor[..., 0::2, 1::2] = checkerboard[..., 0::2, 1::2]
        anchor[..., 1::2, 0::2] = checkerboard[..., 1::2, 0::2]
        return anchor
        
class ChannelARParamTransform(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        '''already consider the double the channle
        '''
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        c1 = self.out_channels
        c2 = self.out_channels * 3 // 2
        self._layers = nn.ModuleList([
            nn.Conv2d(
                self.in_channels, c1, 5, stride=1, padding=2,
                bias=True, padding_mode="zeros"),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                c1, c2, 5, stride=1, padding=2,
                bias=True, padding_mode="zeros"),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                c2, self.out_channels * 2, 5, stride=1, padding=2,
                bias=False, padding_mode="zeros")
        ])
    
    def forward(self, x):
        for layer in self._layers:
            x = layer(x)
        return x

class ParamAggregation(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        d = (self.out_channels - self.in_channels) // 3 // 16 * 16
        c1 = self.in_channels - d
        c2 = self.in_channels - d * 2
        self._layers = nn.ModuleList([
            conv1x1(self.in_channels, c1),
            nn.ReLU(inplace=True),
            conv1x1(c1, c2),
            nn.ReLU(inplace=True),
            conv1x1(c2, self.out_channels)
        ])
    
    def forward(self, x):
        for layer in self._layers:
            x = layer(x)
        return x

class ContextIterator(nn.Module):
    
    inner_param_cls = ChannelARParamTransform
    
    def __init__(self, k_size=5, groups=[16,16,32,64,None]) -> None:
        super().__init__()
        assert groups[-1] is not None
        self.groups = groups
        self.n_groups = len(groups)
        group_start = [sum(groups[0:i]) for i in range(self.n_groups)]
        
        self._sub_modules_channel = nn.ModuleList(
            self.inner_param_cls(group_start[i+1], groups[i+1])
            for i in range(self.n_groups - 1)
        )
        
        '''
        for stage except decompress
        space context return [non-anchor, anchor]
        '''
        self._sub_modules_space = nn.ModuleList(
             CheckerboardContext(groups[i], groups[i] * 2, k_size, stride=1, padding = k_size // 2, 
                               bias=True, padding_mode="zeros")
             for i in range(self.n_groups)
        )
    
    def forward(self, y_slice_iter, stage):
        '''
        y_slice_iter: a iterator on the channel slices of y
        it should exclude the first slice
        return [space_ctx, channle_ctx]
        '''
        y_slices = next(y_slice_iter)
        shape_in = y_slices.shape
        y_context_space = self._sub_modules_space[0](y_slices, stage)
        y_context_channel = torch.zeros((shape_in[0], self.groups[0] * 2, shape_in[2], shape_in[3])).to(y_slices.device)
        yield y_context_space, y_context_channel
        for y_next_slice, channel_sub, space_sub in zip(y_slice_iter, self._sub_modules_channel, self._sub_modules_space[1:]):
            y_context_space = space_sub(y_next_slice, stage)
            y_context_channel = channel_sub(y_slices)
            yield y_context_space, y_context_channel
            y_slices = torch.cat([y_slices, y_next_slice], dim=1)

class SpaceAndChannleParamIterater(nn.Module):
    
    inner_param_cls = ParamAggregation
    
    def __init__(self, hyper_channels,groups=[16,16,32,64,None]) -> None:
        super().__init__()
        self.n_groups = len(groups)
        self._sub_modules = nn.ModuleList(
            self.inner_param_cls(groups[i] * 2 + hyper_channels, groups[i] * 2) if i == 0
            else self.inner_param_cls(groups[i] * 4 + hyper_channels, groups[i] * 2)
            for i in range(self.n_groups)
        )
     
    def forward(self, context_iter, hyper, stage):
        '''
        sccontext: from context iter
        in train stage, the first item of space context contains two path param
        the second item is channle context
        sccontext[0][0]: checkerboard; sccontext[0][1]:anchor
        
        '''
        index = list(range(self.n_groups))
        if stage == CodecStageEnum.TRAIN:
            for sccontext, sub, i in zip(context_iter, self._sub_modules, index):
                if i != 0:
                    out1 = torch.cat([sccontext[0][0], sccontext[1], hyper], dim=1)
                    out1 = sub(out1)
                    out2 = torch.cat([sccontext[0][1], sccontext[1], hyper], dim=1)
                    out2 = sub(out2)
                else:
                    out1 = torch.cat([sccontext[0][0], hyper], dim=1)
                    out1 = sub(out1)
                    out2 = torch.cat([sccontext[0][1], hyper], dim=1)
                    out2 = sub(out2)
                yield out1, out2
        else:
            for sccontext, sub in zip(context_iter, self._sub_modules):
                out = torch.cat([*sccontext, hyper], dim=1)
                out = sub(out)
                yield out

def ParamGroup(ParamList):
    mu_list = []
    sigma_list = []
    for param in ParamList:
        mu, sigma = torch.chunk(param, 2, 1)
        mu_list.append(mu)
        sigma_list.append(sigma)
    mu = torch.cat(mu_list, 1)
    sigma = torch.cat(sigma_list, 1)
    return mu, sigma

def ParamGroupTwoPath(ParamList):
    mu1_list = []
    sigma1_list = []
    
    mu2_list = []
    sigma2_list = []
    for param in ParamList:
        mu1, sigma1 = torch.chunk(param[0], 2, 1)
        mu1_list.append(mu1)
        sigma1_list.append(sigma1)
        
        mu2, sigma2 = torch.chunk(param[1], 2, 1)
        mu2_list.append(mu2)
        sigma2_list.append(sigma2)

    mu1 = torch.cat(mu1_list, 1)
    sigma1 = torch.cat(sigma1_list, 1)
    
    mu2 = torch.cat(mu2_list, 1)
    sigma2 = torch.cat(sigma2_list, 1)
    return {1:(mu1, sigma1), 2:(mu2, sigma2)}


class ELIC(nn.Module):
    def __init__(self, N=192, M=320, stage=CodecStageEnum.TRAIN) -> None:
        super().__init__()
        self.groups = [16,16,32,64,None]
        self.groups[-1] = M - sum(self.groups[:-1])
        self.n_groups = len(self.groups)
        
        self.g_a = YEncoder(N=N, M=M)
        self.h_a = ZEncoder(N=N, M=M)
        self.h_s = ZDecoder(N=N, M=M)
        self.g_s = YDecoder(N=N, M=M)
        
        self.entropy_bottleneck = EntropyBottleneck(N)
        self.y_entorpy = GaussianConditional(None)
        
        self.sliceModel = UnevenChannelSlice(M, self.groups)
        self.contextiter = ContextIterator(groups=self.groups)
        self.y_parameter = SpaceAndChannleParamIterater(hyper_channels=M * 2, groups=self.groups)
        
        self.gate_model = CheckerboardSliceAlignMux()
        
        self.noise_quant = Quantizator_RT()
        self.ste_quant = STEQuant()
        self.stage = stage
        

    def aux_loss(self):
        """Return the aggregated loss over the auxiliary entropy bottleneck
        module(s).
        """
        aux_loss = sum(
            m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck)
        )
        return aux_loss

    def update(self, scale_table=None, force=False):
        """Updates the entropy bottleneck(s) CDF values.

        Needs to be called once after training to be able to later perform the
        evaluation with an actual entropy coder.

        Args:
            scale_table (bool): (default: None)  
            force (bool): overwrite previous values (default: False)

        Returns:
            updated (bool): True if one of the EntropyBottlenecks was updated.

        """
        if scale_table is None:
            scale_table = get_scale_table()
        self.gaussian_conditional.update_scale_table(scale_table, force=force)

        updated = False
        for m in self.children():
            if not isinstance(m, EntropyBottleneck):
                continue
            rv = m.update(force=force)
            updated |= rv
        return updated
    
    def forward(self, x, stage):
        if stage == CodecStageEnum.TRAIN:
            y = self.g_a(x)
            z = self.h_a(y)
            
            _, z_likelihoods = self.entropy_bottleneck(z) ## z_hat is noise quant
            z_round = self.ste_quant(z)
            
            hyper = self.h_s(z_round)
            
            y_hat = self.y_entorpy.quantize(
                y, "noise" if self.training else "dequantize"
            )
            
            y_round = self.ste_quant(y)
            y_slice_iter = iter(self.sliceModel(y_hat))
            y_context_iter = self.contextiter(y_slice_iter, self.stage)
            param = list(self.y_parameter(y_context_iter, hyper, self.stage))
            entropy = ParamGroupTwoPath(param)
            
            mu1, sigma1 = entropy[1]
            mu2, sigma2 = entropy[2]
            _, y_likelihoods1 = self.y_entorpy(y, sigma1, means=mu1)
            _, y_likelihoods2 = self.y_entorpy(y, sigma2, means=mu2)
            
            x_hat = self.g_s(y_round)
            return {
                "x_hat" : x_hat,
                "likelihoods": {"y1":y_likelihoods1, "y2":y_likelihoods2,"z":z_likelihoods}
            }
        else:
            y = self.g_a(x)
            z = self.h_a(y)
            _, z_likelihoods = self.entropy_bottleneck(z) ## z_hat is noise quant
            z_round = self.ste_quant(z)
            
            hyper = self.h_s(z_round)
            
            shape_in = y.shape
            mu_list = []
            sigma_list = []
            y_slice_iter = self.sliceModel(y)
            
            # process the first slice, without channle ctx
            y_slices = y_slice_iter[0]
            zero_ctx = torch.zeros(shape_in[0],self.groups[0] * 2, shape_in[2], shape_in[3]).to(y.device)
            
            mu, _ = torch.chunk(self.y_parameter._sub_modules[0](torch.cat([zero_ctx, hyper], 1)), 2, 1)
            y_sp1_anchor = self.ste_quant(y_slices - mu) + mu
            # set y_hat non anchor part to zero
            y_sp1_anchor[..., 0::2, 1::2] = 0
            y_sp1_anchor[..., 1::2, 0::2] = 0
            y_sp1_ctx = self.contextiter._sub_modules_space[0](y_sp1_anchor, stage)
            # set ctx anchor part ctx to zero
            y_sp1_ctx[..., 0::2, 0::2] = 0
            y_sp1_ctx[..., 1::2, 1::2] = 0
            mu_hat, sigma_hat = torch.chunk(self.y_parameter._sub_modules[0](torch.cat([y_sp1_ctx, hyper], 1)), 2, 1)
            mu_list.append(mu_hat)
            sigma_list.append(sigma_hat) 
            
            y_slices = self.ste_quant(y_slices - mu_hat) + mu_hat
            
            for i in range(1, self.n_groups):  
                y_new_slice = y_slice_iter[i]
                zero_ctx = torch.zeros(shape_in[0],self.groups[i] * 2, shape_in[2], shape_in[3]).to(y.device)
                y_ch_ctx = self.contextiter._sub_modules_channel[i-1](y_slices)
                mu, _ = torch.chunk(self.y_parameter._sub_modules[i](torch.cat([zero_ctx, y_ch_ctx, hyper], 1)), 2, 1)
                
                y_sp_anchor = self.ste_quant(y_new_slice - mu)
                # set y_hat non anchor part to zero
                y_sp_anchor[..., 0::2, 1::2] = 0
                y_sp_anchor[..., 1::2, 0::2] = 0
                y_sp_ctx = self.contextiter._sub_modules_space[i](y_sp_anchor, stage)
                # set ctx anchor part ctx to zero
                y_sp_ctx[..., 0::2, 0::2] = 0
                y_sp_ctx[..., 1::2, 1::2] = 0
                mu_hat, sigma_hat = torch.chunk(self.y_parameter._sub_modules[i](torch.cat([y_sp_ctx, y_ch_ctx, hyper], 1)), 2, 1)
                
                mu_list.append(mu_hat)
                sigma_list.append(sigma_hat)
                y_new_slice = self.ste_quant(y_new_slice - mu_hat) + mu_hat
                y_slices = torch.cat([y_slices, y_new_slice], 1)
            
            mu = torch.cat(mu_list, 1)
            sigma = torch.cat(sigma_list, 1)    
            _, y_likelihoods = self.y_entorpy(y_slices, sigma, means=mu)
            x_hat = self.g_s(y_slices)
            
            return {
                "x_hat" : x_hat,
                "likelihoods": {"y":y_likelihoods,"z":z_likelihoods}
            }
            
            


#%%
if __name__ == "__main__":
    groups = [16,16,32,64,None]
    stage = CodecStageEnum.TRAIN2
    
    # ga = YEncoder().cuda()
    # gs = YDecoder().cuda()
    # ha = ZEncoder().cuda()
    # hs = ZDecoder().cuda()
    # sliceModel = UnevenChannelSlice(320).cuda()
    
    # groups[-1] = 320 - sum(groups[:-1])
    # contextiter = ContextIterator(groups=groups).cuda()
    # param_model = SpaceAndChannleParamIterater(hyper_channels=320 * 2, groups=groups).cuda()
    
    
    # x = torch.rand((2,3,256,256)).cuda()
    # y = ga(x)
    # z = ha(y)
    # hyper = hs(z)
    # y_slice_iter = iter(sliceModel(y))
    # y_context_iter = contextiter(y_slice_iter, stage)
    # param = list(param_model(y_context_iter, hyper, stage))
    # entropy = ParamGroupTwoPath(param)
    # x_hat = gs(y)
    
    elic = ELIC().cuda()
    x = torch.rand((2,3,256,256)).cuda()
    data = elic(x, stage)
    
#%%