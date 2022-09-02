#%%
from codecs import Codec
from importlib.metadata import requires
import math
import torch
import torch.nn as nn
from compressai.entropy_models import EntropyBottleneck, GaussianConditional, GaussianConditionalEntropySkip
from compressai.layers import conv1x1,conv3x3, AttentionBlock, CrossMaskedConv2d, ResBottleneck,StackResBottleneck
from compressai.layers import Quantizator_RT, STEQuant
from timm.models.layers import trunc_normal_
from enum import Enum
from compressai.layers.quant import STEQuantWithSkip
from compressai.models.utils import conv, deconv, update_registered_buffers, CheckerboardDemux, CheckerboardMux
from compressai.models.elic import CodecStageEnum, YEncoder, YDecoder, UnevenChannelSlice, ZEncoder, ZDecoder, ChannelARParamTransform, ParamAggregation
from compressai.models.elic import SpaceAndChannleParamIterater, ELIC, ParamGroup, ParamGroupTwoPath

from compressai.models.utils import Sobel, gumbelSoftmax,gumbelSoftTopk
from random import choice
import numpy as np
#%%
# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64
SKIP_INDEX = 3
SCALE_BOUND = SCALES_MIN

def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


## different mask on different group
class GetMask(nn.Module):
    def __init__(self, channles, groups) -> None:
        super().__init__()
        self.h = [2, 2, 2, 2, 4]
        self.w = [2, 2, 2, 2, 4]
        self.k = [2, 2, 1, 1, 2]
        self.threshold = [0.01] * 5
        # self.threshold = [-100,-100,-100,-100,100]
        # self.threshold = [-100,-100,-100,100,-100]
        self.sobel = Sobel()
        self.groups = groups
    
    def forward(self, x_iter):
        total_mask = []
        B, _, H, W = x_iter[0].shape
        mask = torch.zeros(B, 1, H, W).to(x_iter[0].device)
        mask[..., 0::2, 0::2] = 1
        mask[..., 1::2, 1::2] = 1
        
        mask2 = torch.zeros(B, 1, H, W).to(x_iter[0].device)
        mask2[..., 0::4, 0::4] = 1
        mask2[..., 2::4, 2::4] = 1
        for x, i in zip(x_iter, range(len(self.groups))):
            x = x.mean(dim=1, keepdim=True)
            pool_pos = x <= self.threshold[i]
            m = mask*(~pool_pos) + mask2*(pool_pos)
            
            total_mask.append(m)
                        
        return total_mask
    


# select mask by edge
# class GetMask(nn.Module):
#     def __init__(self, channles, groups) -> None:
#         super().__init__()
#         self.h = 2
#         self.w = 2
#         self.threshold = 0.01
#         # self.sobel = Sobel()
    
    
#     @staticmethod
#     def find4neigh(position, H, W):
#         bottom = torch.cat([position[:,0:2], position[:,2:3] - 1, position[:,3:]], dim=1)
#         bottom[:, 2:3].clamp_(0, H-1)
#         top = torch.cat([position[:,0:2], position[:,2:3] + 1, position[:,3:]], dim=1)
#         top[:, 2:3].clamp_(0, H-1)
#         right = torch.cat([position[:,0:2], position[:,2:3], position[:,3:] + 1], dim=1)
#         right[:, 3:].clamp_(0, W-1)
#         left = torch.cat([position[:,0:2], position[:,2:3], position[:,3:] - 1], dim=1)
#         left[:, 3:].clamp_(0, W-1)
#         return bottom, top, left, right

#     def forward(self, hyper):
#         B,C,H,W = hyper.shape
#         h = hyper.mean(dim=1, keepdim=True)
#         threshold = h.view(B, 1, -1).median(dim=-1, keepdim=True)[0]
#         threshold.unsqueeze_(-1)
#         # threshold = self.threshold
#         pool_pos = h <= threshold
        
#         mask = torch.zeros(B, 1, H, W).to(hyper.device)
#         mask[..., 0::2, 0::2] = 1
#         mask[..., 1::2, 1::2] = 1
        
#         mask2 = torch.zeros(B, 1, H, W).to(hyper.device)
#         mask2[..., 0::4, 0::4] = 1
#         mask2[..., 2::4, 2::4] = 1
        
#         return mask*(~pool_pos) + mask2*(pool_pos)
        
#         # pool_pos[..., 0::2, 0::2] = False
#         # pool_pos[..., 1::2, 1::2] = False
#         # bottom, top, left, right = self.find4neigh(torch.argwhere(pool_pos), H, W)
#         # mask[bottom[:, 0], bottom[:, 1], bottom[:,2], bottom[:, 3]] = 0
#         # mask[top[:, 0], top[:, 1], top[:,2], top[:, 3]] = 0
#         # mask[right[:, 0], right[:, 1], right[:,2], right[:, 3]] = 0
#         # mask[left[:, 0], left[:, 1], left[:,2], left[:, 3]] = 0
        
#         # mask[pool_pos] = 1
#         return mask


# attention
# class GetMask(nn.Module):
#     def __init__(self, N: int, groups, k_size=5):
#         super().__init__()
#         self.h = 4
#         self.w = 4
#         self.target_elements = 4
#         self.soft_topk = gumbelSoftTopk()

#         class ResidualUnit(nn.Module):
#             """Simple residual unit."""

#             def __init__(self):
#                 super().__init__()
#                 self.conv = nn.Sequential(
#                     conv1x1(N, N // 2),
#                     nn.ReLU(inplace=True),
#                     conv3x3(N // 2, N // 2),
#                     nn.ReLU(inplace=True),
#                     conv1x1(N // 2, N),
#                 )
#                 self.relu = nn.ReLU(inplace=True)

#             def forward(self, x):
#                 identity = x
#                 out = self.conv(x)
#                 out += identity
#                 out = self.relu(out)
#                 return out

#         self.conv_a = nn.Sequential(ResidualUnit())

#         self.conv_b = nn.Sequential(
#             ResidualUnit(),
#             conv1x1(N, N),
#         )
        
#         self.last_conv = conv1x1(N, 1)

#     def forward(self, x):
#         # x = hyper.clone()
#         # x = x.detach()
#         identity = x
#         a = self.conv_a(x)
#         b = self.conv_b(x)
#         out = a * torch.sigmoid(b)
#         out += identity
#         out = self.last_conv(out)
#         B, C, H, W = out.shape
        
#         ## local windows
#         out = out.view(B, C, H // self.h, self.h, W // self.w, se1
#         # out = out.view(B, C, H * W)
#         # target_elements = H * W // 2
        
        
#         if self.training:
#             out = gumbelSoftmax(out, hard=True, dim=-1, elements=target_elements)
#             # out = self.soft_topk(out, hard=True, dim=-1, elements=target_elements)
#         else:
#             index = torch.topk(out, target_elements, -1, largest=True, sorted=False)[1]
#             # index = out.max(-1, keepdim=True)[1]
#             out = torch.zeros_like(out, memory_format=torch.legacy_contiguous_format).scatter_(-1, index, 1.0)
        
#         out = out.view(B, C, H // self.h, W // self.w, self.h, self.w).permute(0, 1, 2, 4, 3, 5).reshape(B,C,H,W)
#         return out

#         # return out.view(B, C, H, W)
        

# attention2
# class GetMask(nn.Module):
#     def __init__(self, N: int, groups, k_size=5):
#         super().__init__()
#         self.h = 4
#         self.w = 4
#         self.target_elements = 4
#         self.soft_topk = gumbelSoftTopk()

#         class ResidualUnit(nn.Module):
#             """Simple residual unit."""

#             def __init__(self):
#                 super().__init__()
#                 self.conv = nn.Sequential(
#                     conv1x1(N, N // 2),
#                     nn.ReLU(inplace=True),
#                     conv3x3(N // 2, N // 2),
#                     nn.ReLU(inplace=True),
#                     conv1x1(N // 2, N),
#                 )
#                 self.relu = nn.ReLU(inplace=True)

#             def forward(self, x):
#                 identity = x
#                 out = self.conv(x)
#                 out += identity
#                 out = self.relu(out)
#                 return out

#         self.conv_a = nn.Sequential(ResidualUnit())

#         self.conv_b = nn.Sequential(
#             ResidualUnit(),
#             conv1x1(N, N),
#         )
        
#         self.last_conv = conv1x1(N, 1)

#     def forward(self, x):
#         # x = hyper.clone()
#         # x = x.detach()
#         identity = x
#         a = self.conv_a(x)
#         b = self.conv_b(x)
#         out = a * torch.sigmoid(b)
#         out += identity
#         out = self.last_conv(out)
#         out = torch.sigmoid(out)
#         if not self.training:
#             out = torch.where(out >= 0.5, 1, 0)
#         B, C, H, W = out.shape
        
#         # # global
#         # out = out.view(B, C, H * W)
#         # target_elements = H * W // 2
        
        
#         # if self.training:
#         #     out = gumbelSoftmax(out, hard=True, dim=-1, elements=target_elements)
#         #     # out = self.soft_topk(out, hard=True, dim=-1, elements=target_elements)
#         # else:
#         #     index = torch.topk(out, target_elements, -1, largest=True, sorted=False)[1]
#         #     # index = out.max(-1, keepdim=True)[1]
#         #     out = torch.zeros_like(out, memory_format=torch.legacy_contiguous_format).scatter_(-1, index, 1.0)
        
#         # # out = out.view(B, C, H // self.h, W // self.w, self.h, self.w).permute(0, 1, 2, 4, 3, 5).reshape(B,C,H,W)
#         # # return out

#         half = torch.zeros(B, 1, H, W).to(x.device)
#         half[..., 0::2, 0::2] = 1
#         half[..., 1::2, 1::2] = 1
        
#         quarter = torch.zeros(B, 1, H, W).to(x.device)
#         quarter[..., 0::4, 0::4] = 1
#         quarter[..., 2::4, 2::4] = 1

#         out = out.view(B, C, H, W)
#         return out * half + (1 - out) * quarter
        
class AdaptiveContext(nn.Module):
    def __init__(self, in_channles, out_channles, k_size=5) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channles, out_channles, k_size, 1, (k_size - 1) // 2, bias=True)
        self.conv_mod = nn.Conv2d(1, 1, k_size, 1, (k_size - 1) // 2, bias=False)
        weight = torch.ones((5,5), dtype=self.conv_mod.weight.dtype)[None, None, :, :]
        # bias = torch.tensor([1e-4])
        self.conv_mod.weight = nn.Parameter(weight, requires_grad=False)
    
    def forward(self, x, mask, stage):
        assert mask.shape[0] == x.shape[0]
        assert mask.shape[1] == 1
        out = self.conv(x)
        mod = self.conv_mod(mask)
        out /= (mod+1)
        # zero_pos = mod == 0
        # out = out * ~zero_pos + torch.zeros_like(out) * zero_pos
        if stage == CodecStageEnum.TRAIN:
            out = out.mul(1 - mask)
            anchor = torch.zeros(x.shape[0], x.shape[1] * 2, x.shape[2], x.shape[3]).to(x.device)
            return out, anchor
        elif stage == CodecStageEnum.TRAIN_ONEPATH:
            return out.mul(1 - mask)
        else:
            return out

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
             AdaptiveContext(groups[i], groups[i] * 2, k_size)
             for i in range(self.n_groups)
        )
    
    def forward(self, y_slice_iter, mask, stage ):
        '''
        y_slice_iter: a iterator on the channel slices of y
        it should exclude the first slice
        return [space_ctx, channle_ctx]
        '''
        y_slices = next(y_slice_iter)
        shape_in = y_slices.shape
        y_context_space = self._sub_modules_space[0](y_slices * mask, mask, stage)
        yield (y_context_space,)
        for y_next_slice, channel_sub, space_sub in zip(y_slice_iter, self._sub_modules_channel, self._sub_modules_space[1:]):
            y_context_space = space_sub(y_next_slice * mask, mask, stage)
            y_context_channel = channel_sub(y_slices)
            yield (y_context_space, y_context_channel)
            y_slices = torch.cat([y_slices, y_next_slice], dim=1)



class ELIC_ADA(ELIC):
    def __init__(self, N=192, M=320, stage=CodecStageEnum.TRAIN, skipIndex=SKIP_INDEX) -> None:
        super().__init__() 
        self.groups = [16,16,32,64,None]
        self.groups[-1] = M - sum(self.groups[:-1])
        self.n_groups = len(self.groups)
        
        self.g_a = YEncoder(N=N, M=M)
        self.h_a = ZEncoder(N=N, M=M)
        self.h_s = ZDecoder(N=N, M=M)
        self.g_s = YDecoder(N=N, M=M)
        
        self.entropy_bottleneck = EntropyBottleneck(N)
        self.y_entorpy = GaussianConditionalEntropySkip(None, skipIndex=skipIndex)
        
        self.sliceModel = UnevenChannelSlice(M, self.groups)
        self.contextiter = ContextIterator(groups=self.groups)
        self.y_parameter = SpaceAndChannleParamIterater(hyper_channels=M * 2, groups=self.groups)
        
        self.noise_quant = Quantizator_RT()
        # self.ste_quant = STEQuant()
        
        self.skipIndex = skipIndex       
        self.skipThreshold = get_scale_table()[skipIndex] if skipIndex >=0 else -100
        self.ste_quant = STEQuantWithSkip(skipThreshold=self.skipThreshold)
        # self.ada_mask = GetMask(M * 2)
        self.ada_mask = GetMask(M * 2, groups=self.groups)
        self.stage = stage
        # self.get_mask(torch.zeros(1, 320, 32, 48))
    
    # def get_mask(self, x):
    #     mask = self.ada_mask(x)
    #     self.mask = mask
    #     self.mask = self.mask.to('cuda')
    
    def forward(self, x):
        if self.stage == CodecStageEnum.TRAIN or self.stage == CodecStageEnum.TRAIN_ONEPATH:
            y = self.g_a(x)
            z = self.h_a(y)
            
            _, z_likelihoods = self.entropy_bottleneck(z) ## z_hat is noise quant
            z_round = self.ste_quant(z)
            
            hyper = self.h_s(z_round)
            mask = self.ada_mask(hyper)
            
            y_hat = self.y_entorpy.quantize(
                y, "noise" if self.training else "dequantize"
            )
            
            y_round = self.ste_quant(y)
            y_slice_iter = iter(self.sliceModel(y_hat))
            y_context_iter = self.contextiter(y_slice_iter, mask, self.stage)
            param = self.y_parameter(y_context_iter, hyper, self.stage)

            if self.stage == CodecStageEnum.TRAIN:
                assert self.skipIndex < 0
                entropy = ParamGroupTwoPath(param)
                
                mu1, sigma1 = entropy[1]
                mu2, sigma2 = entropy[2]
                _, y_likelihoods1 = self.y_entorpy(y, sigma1, means=mu1)
                _, y_likelihoods2 = self.y_entorpy(y, sigma2, means=mu2)
                
                x_hat = self.g_s(y_round)
                return {
                    "x_hat" : x_hat,
                    "likelihoods": {"y1":y_likelihoods1, "y2":y_likelihoods2,"z":z_likelihoods},
                    "mask" : mask
                }
            else:
                mu, sigma = ParamGroup(param)
                _, y_likelihoods = self.y_entorpy(y, sigma, means=mu)
                skip_pos = sigma <= self.y_entorpy.skipThreshold
                y_round = ~skip_pos * y_round + skip_pos * mu
                x_hat = self.g_s(y_round)
                return {
                    "x_hat" : x_hat,
                    "likelihoods": {"y":y_likelihoods, "z":z_likelihoods},
                    "mask": mask
                }
        else:
            y = self.g_a(x)
            z = self.h_a(y)
            _, z_likelihoods = self.entropy_bottleneck(z) ## z_hat is noise quant
            z_round = self.ste_quant(z)
            
            hyper = self.h_s(z_round)
            
            hyper_iter = self.sliceModel(torch.chunk(hyper, 2, 1)[0])
            mask = self.ada_mask(hyper_iter)
            # mask = self.ada_mask(hyper)
            # mask = self.mask
                        
            shape_in = y.shape
            mu_list = []
            sigma_list = []
            y_slice_iter = self.sliceModel(y)
            
            # process the first slice, without channle ctx
            y_slices = y_slice_iter[0]
            zero_ctx = torch.zeros(shape_in[0],self.groups[0] * 2, shape_in[2], shape_in[3]).to(y.device)
            
            mu, sigma = torch.chunk(self.y_parameter._sub_modules[0](torch.cat([zero_ctx, hyper], 1)), 2, 1)
            # y_sp1_anchor = self.ste_quant(y_slices - mu) + mu
            y_sp1_anchor = self.ste_quant(y_slices, means=mu, scales=sigma)
            # set y_hat non anchor part to zero
            y_sp1_anchor *= mask[0]
            # y_sp1_anchor[..., 0::2, 1::2] = 0
            # y_sp1_anchor[..., 1::2, 0::2] = 0
            y_sp1_ctx = self.contextiter._sub_modules_space[0](y_sp1_anchor, mask[0], self.stage)
            # set ctx anchor part ctx to zero
            y_sp1_ctx *= (1 - mask[0])
            # y_sp1_ctx[..., 0::2, 0::2] = 0
            # y_sp1_ctx[..., 1::2, 1::2] = 0
            mu_hat, sigma_hat = torch.chunk(self.y_parameter._sub_modules[0](torch.cat([y_sp1_ctx, hyper], 1)), 2, 1)
            mu_list.append(mu_hat)
            sigma_list.append(sigma_hat) 
            
            # y_slices = self.ste_quant(y_slices - mu_hat) + mu_hat
            y_slices = self.ste_quant(y_slices, means=mu_hat, scales=sigma_hat)

            
            for i in range(1, self.n_groups):  
                y_new_slice = y_slice_iter[i]
                zero_ctx = torch.zeros(shape_in[0],self.groups[i] * 2, shape_in[2], shape_in[3]).to(y.device)
                y_ch_ctx = self.contextiter._sub_modules_channel[i-1](y_slices)
                mu, sigma = torch.chunk(self.y_parameter._sub_modules[i](torch.cat([zero_ctx, y_ch_ctx, hyper], 1)), 2, 1)
                
                # y_sp_anchor = self.ste_quant(y_new_slice - mu) + mu
                y_sp_anchor = self.ste_quant(y_new_slice, means=mu, scales=sigma)
                # set y_hat non anchor part to zero
                y_sp_anchor *= mask[i]
                # y_sp_anchor[..., 0::2, 1::2] = 0
                # y_sp_anchor[..., 1::2, 0::2] = 0
                y_sp_ctx = self.contextiter._sub_modules_space[i](y_sp_anchor, mask[i], self.stage)
                # set ctx anchor part ctx to zero
                y_sp_ctx *= (1 - mask[i])
                # y_sp_ctx[..., 0::2, 0::2] = 0
                # y_sp_ctx[..., 1::2, 1::2] = 0
                mu_hat, sigma_hat = torch.chunk(self.y_parameter._sub_modules[i](torch.cat([y_sp_ctx, y_ch_ctx, hyper], 1)), 2, 1)
                
                mu_list.append(mu_hat)
                sigma_list.append(sigma_hat)
                # y_new_slice = self.ste_quant(y_new_slice - mu_hat) + mu_hat
                y_new_slice = self.ste_quant(y_new_slice, means=mu_hat, scales=sigma_hat)
                y_slices = torch.cat([y_slices, y_new_slice], 1)
            
            mu = torch.cat(mu_list, 1)
            sigma = torch.cat(sigma_list, 1)
            # entropy skip, update sigma (train breakdown)
            # sigma = self.ste_quant.STEScaleUpdate(sigma)
            _, y_likelihoods = self.y_entorpy(y, sigma, means=mu)
            # _, y_likelihoods = self.y_entorpy(y_slices, sigma, means=mu)
            x_hat = self.g_s(y_slices)
            
            return {
                "x_hat" : x_hat,
                "likelihoods": {"y":y_likelihoods,"z":z_likelihoods},
                "mask" : mask,
                "y":y,
                "hyper":hyper
            }

    @torch.inference_mode()        
    def compress(self, x):
            y = self.g_a(x)
            z = self.h_a(y)
            
            z_strings = self.entropy_bottleneck.compress(z)
            z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
            
            hyper = self.h_s(z_hat)
            mask = self.ada_mask(hyper)
            # mask = self.mask
            # no_sp_flag = (mask == 0).all()
            no_sp_flag = False
            anchor_index = mask.squeeze() == 1
            non_anchor_index = ~anchor_index
            
            shape_in = y.shape
            mu_list = []
            sigma_list = []
            y_slice_iter = self.sliceModel(y)
            
            # process the first slice, without channle ctx
            y_slices = y_slice_iter[0]
            zero_ctx = torch.zeros(shape_in[0],self.groups[0] * 2, shape_in[2], shape_in[3]).to(y.device)
            
            mu, sigma = torch.chunk(self.y_parameter._sub_modules[0](torch.cat([zero_ctx, hyper], 1)), 2, 1)
            # y_sp1_anchor = self.ste_quant(y_slices - mu) + mu
            y_sp1_anchor = self.y_entorpy.quantize(y_slices, "dequantize", means=mu, scales=sigma)
            # set y_hat non anchor part to zero
            y_sp1_anchor *= mask
            # y_sp1_anchor[..., non_anchor_index] = 0
            # y_sp1_anchor[..., 0::2, 1::2] = 0
            # y_sp1_anchor[..., 1::2, 0::2] = 0
            y_sp1_ctx = self.contextiter._sub_modules_space[0](y_sp1_anchor, mask, self.stage)
            # set ctx anchor part ctx to zero
            y_sp1_ctx *= (1 - mask)
            # y_sp1_ctx[..., anchor_index] = 0
            
            # y_sp1_ctx[..., 0::2, 0::2] = 0
            # y_sp1_ctx[..., 1::2, 1::2] = 0
            mu_hat, sigma_hat = torch.chunk(self.y_parameter._sub_modules[0](torch.cat([y_sp1_ctx, hyper], 1)), 2, 1)
            mu_list.append(mu_hat)
            sigma_list.append(sigma_hat) 
            
            # y_slices = self.ste_quant(y_slices - mu_hat) + mu_hat
            y_slices = self.y_entorpy.quantize(y_slices, "dequantize", means=mu_hat, scales=sigma_hat)
            
            for i in range(1, self.n_groups):  
                y_new_slice = y_slice_iter[i]
                zero_ctx = torch.zeros(shape_in[0],self.groups[i] * 2, shape_in[2], shape_in[3]).to(y.device)
                y_ch_ctx = self.contextiter._sub_modules_channel[i-1](y_slices)
                mu, sigma = torch.chunk(self.y_parameter._sub_modules[i](torch.cat([zero_ctx, y_ch_ctx, hyper], 1)), 2, 1)
                
                # y_sp_anchor = self.ste_quant(y_new_slice - mu) + mu
                y_sp_anchor = self.y_entorpy.quantize(y_new_slice, "dequantize", means=mu, scales=sigma)
                # set y_hat non anchor part to zero
                y_sp_anchor *= mask
                # y_sp_anchor[..., non_anchor_index] = 0
                # y_sp_anchor[..., 0::2, 1::2] = 0
                # y_sp_anchor[..., 1::2, 0::2] = 0
                y_sp_ctx = self.contextiter._sub_modules_space[i](y_sp_anchor, mask, self.stage)
                # set ctx anchor part to zero
                y_sp_ctx *= (1 - mask)
                # y_sp_ctx[..., anchor_index] = 0
                # y_sp_ctx[..., 0::2, 0::2] = 0
                # y_sp_ctx[..., 1::2, 1::2] = 0
                mu_hat, sigma_hat = torch.chunk(self.y_parameter._sub_modules[i](torch.cat([y_sp_ctx, y_ch_ctx, hyper], 1)), 2, 1)
                
                mu_list.append(mu_hat)
                sigma_list.append(sigma_hat)
                # y_new_slice = self.ste_quant(y_new_slice - mu_hat) + mu_hat
                y_new_slice = self.y_entorpy.quantize(y_new_slice, "dequantize", means=mu_hat, scales=sigma_hat)
                y_slices = torch.cat([y_slices, y_new_slice], 1)
            
            mu = torch.cat(mu_list, 1)
            sigma = torch.cat(sigma_list, 1)
            
            # skip_threshold = self.y_entorpy.skipThreshold
            # skip_proportion = (sigma <= skip_threshold).sum() / sigma.numel()
            # print(skip_proportion.item())
            
            y_string_list = []
            for y_slice, mu, sigma in zip(self.sliceModel(y_slices), mu_list, sigma_list):
                if not no_sp_flag:
                    y1, y2 = y_slice[..., anchor_index], y_slice[..., non_anchor_index]
                    mu1, mu2 = mu[..., anchor_index], mu[..., non_anchor_index]
                    sigma1, sigma2 = sigma[..., anchor_index], sigma[..., non_anchor_index]
                else:
                    y1, y2 = torch.zeros(y_slice.shape[0:2], device=y_slice.device), y_slice
                    mu1, mu2 = torch.zeros(y_slice.shape[0:2], device=y_slice.device), mu
                    sigma1, sigma2 = torch.zeros(y_slice.shape[0:2], device=y_slice.device), sigma
                # y1, y2 = self.Demux(y_slice)
                # mu1, mu2 = self.Demux(mu)
                # sigma1, sigma2 = self.Demux(sigma)
                indexes_y1 = self.y_entorpy.build_indexes(sigma1)
                indexes_y2 = self.y_entorpy.build_indexes(sigma2)
                y1_strings = self.y_entorpy.compress(y1, indexes_y1, means=mu1)
                y2_strings = self.y_entorpy.compress(y2, indexes_y2, means=mu2)
                y_string_list.append(y1_strings)
                y_string_list.append(y2_strings)
            
            return {
                "strings": [*y_string_list, z_strings],
                "shape": z.size()[-2:],
                "y" : y_slices if self.stage == CodecStageEnum.Check else None,
            }
    
    @torch.inference_mode()
    def decompress(self, strings, shape):
        z_hat = self.entropy_bottleneck.decompress(strings[-1], shape)
        hyper = self.h_s(z_hat)
        mask = self.ada_mask(hyper)
        # mask = self.mask
        
        # if (mask == 0).all():
        #     raise ValueError("mask is all zero") 
        anchor_index = mask.squeeze() == 1
        non_anchor_index = ~anchor_index
        
        shape_in = hyper.shape
        
        # first stage, without channle context
        zero_ctx = torch.zeros(shape_in[0],self.groups[0] * 2, shape_in[2], shape_in[3]).to(hyper.device)
        mu, sigma = torch.chunk(self.y_parameter._sub_modules[0](torch.cat([zero_ctx, hyper], 1)), 2, 1)
        mu_hat = mu[..., anchor_index]
        sigma_hat = sigma[..., anchor_index]
        # mu_hat, _ = self.Demux(mu)
        # sigma_hat, _ = self.Demux(sigma)
        indexes_y1 = self.y_entorpy.build_indexes(sigma_hat)
        y1 = self.y_entorpy.decompress(strings[0],indexes_y1, means=mu_hat)
        
        y_sp1_anchor = torch.zeros_like(mu)
        y_sp1_anchor[..., anchor_index] = y1
        # y_sp1_anchor = self.Mux(y1, torch.zeros_like(y1))
        y_sp1_ctx = self.contextiter._sub_modules_space[0](y_sp1_anchor, mask, self.stage)
        # set ctx anchor part ctx to zero
        # decompress stage neccessary?
        # y_sp1_ctx[..., 0::2, 0::2] = 0
        # y_sp1_ctx[..., 1::2, 1::2] = 0
        y_sp1_ctx[..., anchor_index] = 0
       
        mu, sigma = torch.chunk(self.y_parameter._sub_modules[0](torch.cat([y_sp1_ctx, hyper], 1)), 2, 1)
        mu_hat = mu[..., non_anchor_index]
        sigma_hat = sigma[..., non_anchor_index]
        # _, mu_hat = self.Demux(mu)
        # _, sigma_hat = self.Demux(sigma)
        indexes_y1 = self.y_entorpy.build_indexes(sigma_hat)
        y1 = self.y_entorpy.decompress(strings[1], indexes_y1, means=mu_hat)
        
        y_slices = torch.zeros_like(mu)
        y_slices[..., non_anchor_index] = y1
        # y_slices = self.Mux(torch.zeros_like(y_slices), y_slices)
        
        y_slices += y_sp1_anchor
        
        for i in range(1, self.n_groups):
            zero_ctx = torch.zeros(shape_in[0],self.groups[i] * 2, shape_in[2], shape_in[3]).to(hyper.device)
            y_ch_ctx = self.contextiter._sub_modules_channel[i-1](y_slices)
            mu, sigma = torch.chunk(self.y_parameter._sub_modules[i](torch.cat([zero_ctx, y_ch_ctx, hyper], 1)), 2, 1)
            mu_hat = mu[..., anchor_index]
            sigma_hat = sigma[..., anchor_index]
            # mu_hat, _ = self.Demux(mu)
            # sigma_hat, _ = self.Demux(sigma)
            indexes_y = self.y_entorpy.build_indexes(sigma_hat)
            y = self.y_entorpy.decompress(strings[i * 2], indexes_y, means=mu_hat)
            y_sp_anchor = torch.zeros_like(mu)
            y_sp_anchor[..., anchor_index] = y 
            # y_sp_anchor = self.Mux(y, torch.zeros_like(y))
            y_sp_ctx = self.contextiter._sub_modules_space[i](y_sp_anchor, mask, self.stage)
            
            y_sp_ctx[..., anchor_index] = 0
            # y_sp_ctx[..., 0::2, 0::2] = 0
            # y_sp_ctx[..., 1::2, 1::2] = 0
            mu, sigma = torch.chunk(self.y_parameter._sub_modules[i](torch.cat([y_sp_ctx, y_ch_ctx, hyper], 1)), 2, 1)
            mu_hat = mu[..., non_anchor_index]
            sigma_hat = sigma[..., non_anchor_index] 
            # _, mu_hat = self.Demux(mu)
            # _, sigma_hat = self.Demux(sigma)
            indexes_y = self.y_entorpy.build_indexes(sigma_hat)
            y = self.y_entorpy.decompress(strings[i * 2 + 1], indexes_y, means=mu_hat)
            y_new_slice = torch.zeros_like(mu)
            y_new_slice[..., non_anchor_index] = y

            y_new_slice += y_sp_anchor
            # y = self.Mux(torch.zeros_like(y), y)
            # y_new_slice = y_sp_anchor + y
            y_slices = torch.cat([y_slices, y_new_slice], dim=1)
        
        x_hat = self.g_s(y_slices)
        return {
            "x_hat" : x_hat,
            "y" : y_slices if self.stage == CodecStageEnum.Check else None,
        }

#%%
if __name__ == "__main__":
    from PIL import Image
    from torchvision.transforms import ToTensor
    torch.backends.cudnn.deterministic = True
    
    groups = [16,16,32,64,None]
    
    img = Image.open('/hdd/zyw/ImageDataset/kodak/kodim01.png').convert("RGB")
    x = ToTensor()(img).cuda()
    x = x.unsqueeze(0)
    
    net = ELIC_ADA(skipIndex=SKIP_INDEX).cuda()
    net.eval()
    # net.train()
    net.stage = CodecStageEnum.Check
    

    checkpoint = torch.load('pretrained/elic_ada/6/threshold.pth.tar')
    net.load_state_dict(checkpoint['state_dict'], strict=False)
    
    net.update(force=True)
    
    out = net(x)
    
    # ## 验证编解码正确性
    # code = net.compress(x)
    # rec = net.decompress(code['strings'], code['shape'])
    # y1 = code['y']
    # y2 = rec['y']
    # total_bytes = 0                
    # strings = code['strings']
    # for s in strings:
    #     if isinstance(s, list):
    #         for i in s:
    #             total_bytes += len(i)
    #     else:
    #         total_bytes += len(i)

    # print(total_bytes * 8 / 768 / 512)
    # print((y1 == y2).all())


    # # 速度测试
    # import time
    # enc_time = []
    # dec_time = []
    # stage = CodecStageEnum.TEST
    # count = 20
    # for _ in range(count):  
    #     t1 = time.process_time()
    #     code = net.compress(x)
    #     t2 = time.process_time()
    #     enc_time.append(t2 - t1)
        
    #     t1 = time.process_time()
    #     rec = net.decompress(code['strings'], code['shape'])
    #     t2 = time.process_time()
    #     dec_time.append(t2 - t1)
    # enc_time = sum(enc_time) / count
    # dec_time = sum(dec_time) / count
    # print("encode time: ", enc_time * 1000)
    # print("decode time: ", dec_time * 1000)
    
    import matplotlib.pyplot as plt
    m = out['mask'][0]
    y = out['y']
    hyper = out['hyper']
    m = m.squeeze().cpu()
    plt.imshow(y[0,0,...].cpu().detach())
    plt.imshow(m)

    
    from examples.train_elic import RateDistortionLoss
    loss = RateDistortionLoss(0.045)
    res = loss(out, x)
    print(res['bpp_loss'].item())
    print(10 * torch.log10(1 / res["mse_loss"]).item())
    print(res['loss'])

#%%