#%%
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

# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64
SKIP_INDEX = 3
SCALE_BOUND = SCALES_MIN

def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


class CodecStageEnum(Enum):
    TRAIN = 1
    TRAIN_ONEPATH=2
    TRAIN2 = 3
    VALID = 4
    TEST = 5
    COMPRESS = 6
    DECOMPRESS = 7
    Check = 8


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
            StackResBottleneck(N, num_rsb),
            deconv(N, N, 5, 2),
            StackResBottleneck(N, num_rsb),
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
    
    def gate(self, checkerboard):
        x = torch.clone(checkerboard)
        
        x[..., 0::2, 0::2] = 0
        x[..., 1::2, 1::2] = 0
        return x

    def forward(self, x, stage):
        if stage == CodecStageEnum.TRAIN:
            checkerboard = self.cross_conv(x)
            anchor = torch.zeros(x.shape[0], x.shape[1] * 2, x.shape[2], x.shape[3]).to(x.device)
            return checkerboard, anchor
        elif stage == CodecStageEnum.TRAIN_ONEPATH:
            return self.gate(self.cross_conv(x))
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
        # TODO channle number settings may not flexible
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
        '''
        param model: aggregate the channle and space param
        '''
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
        yield (y_context_space,)
        for y_next_slice, channel_sub, space_sub in zip(y_slice_iter, self._sub_modules_channel, self._sub_modules_space[1:]):
            y_context_space = space_sub(y_next_slice, stage)
            y_context_channel = channel_sub(y_slices)
            yield (y_context_space, y_context_channel)
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


def add_sn(m):
    for name, layer in m.named_children():
        m.add_module(name, add_sn(layer))
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        return nn.utils.spectral_norm(m)
    else:
        return m

class ELIC(nn.Module):
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
        
        self.gate_model = CheckerboardSliceAlignMux()
        
        self.noise_quant = Quantizator_RT()
        # self.ste_quant = STEQuant()
        self.ste_quant = STEQuantWithSkip(skipThreshold=get_scale_table()[skipIndex])
        self.stage = stage
        self.skipIndex = skipIndex
        
        self.Mux = CheckerboardMux
        self.Demux = CheckerboardDemux
        # add_sn(self.g_a)
        # add_sn(self.h_a)
        

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = 192
        M = 320
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

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
        self.y_entorpy.update_scale_table(scale_table, force=force)

        updated = False
        for m in self.children():
            if not isinstance(m, EntropyBottleneck):
                continue
            rv = m.update(force=force)
            updated |= rv
        return updated
    
    def fixed_module(self, name_list:list):
        """select the trainbale module

        Args:
            name_list (list): submodul needed to be fixed
        """
        fixed_submodule = []
        trainable_submodule = []
        for name, module in self.named_children():
            for fixed_name in name_list:
                if fixed_name in name:
                    for param in module.parameters():
                        param.requires_grad = False
                    fixed_submodule.append(name)
                else:
                    trainable_submodule.append(name)
                        
        print("fixed submodule : \n" + str(fixed_submodule))
        print("trainable submodule : \n" + str(trainable_submodule))

    def load_state_dict(self, state_dict, strict=False):
        # Dynamically update the entropy bottleneck buffers related to the CDFs
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        update_registered_buffers(
            self.y_entorpy,
            "y_entorpy",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict, strict=strict)
    
    def forward(self, x):
        if self.stage == CodecStageEnum.TRAIN or self.stage == CodecStageEnum.TRAIN_ONEPATH:
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
            param = self.y_parameter(y_context_iter, hyper, self.stage)

            if self.stage == CodecStageEnum.TRAIN:
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
                mu, sigma = ParamGroup(param)
                _, y_likelihoods = self.y_entorpy(y, sigma, means=mu)
                x_hat = self.g_s(y_round)
                return {
                    "x_hat" : x_hat,
                    "likelihoods": {"y":y_likelihoods, "z":z_likelihoods}
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
            
            mu, sigma = torch.chunk(self.y_parameter._sub_modules[0](torch.cat([zero_ctx, hyper], 1)), 2, 1)
            # y_sp1_anchor = self.ste_quant(y_slices - mu) + mu
            y_sp1_anchor = self.ste_quant(y_slices, means=mu, scales=sigma)
            # set y_hat non anchor part to zero
            y_sp1_anchor[..., 0::2, 1::2] = 0
            y_sp1_anchor[..., 1::2, 0::2] = 0
            y_sp1_ctx = self.contextiter._sub_modules_space[0](y_sp1_anchor, self.stage)
            # set ctx anchor part ctx to zero
            y_sp1_ctx[..., 0::2, 0::2] = 0
            y_sp1_ctx[..., 1::2, 1::2] = 0
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
                y_sp_anchor[..., 0::2, 1::2] = 0
                y_sp_anchor[..., 1::2, 0::2] = 0
                y_sp_ctx = self.contextiter._sub_modules_space[i](y_sp_anchor, self.stage)
                # set ctx anchor part ctx to zero
                y_sp_ctx[..., 0::2, 0::2] = 0
                y_sp_ctx[..., 1::2, 1::2] = 0
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
                # "skip_pos": sigma <= self.y_entorpy.skipThreshold
                # "y":y,
                # "hyper":hyper
            }
    
    @torch.inference_mode()        
    def compress(self, x):
            y = self.g_a(x)
            z = self.h_a(y)
            
            z_strings = self.entropy_bottleneck.compress(z)
            z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
            
            hyper = self.h_s(z_hat)
            
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
            y_sp1_anchor[..., 0::2, 1::2] = 0
            y_sp1_anchor[..., 1::2, 0::2] = 0
            y_sp1_ctx = self.contextiter._sub_modules_space[0](y_sp1_anchor, self.stage)
            # set ctx anchor part ctx to zero
            y_sp1_ctx[..., 0::2, 0::2] = 0
            y_sp1_ctx[..., 1::2, 1::2] = 0
            mu_hat, sigma_hat = torch.chunk(self.y_parameter._sub_modules[0](torch.cat([y_sp1_ctx, hyper], 1)), 2, 1)
            mu_list.append(mu_hat)
            sigma_list.append(sigma_hat) 

            first_mu = [mu, mu_hat]
            first_sigma = [sigma, sigma_hat]
            
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
                y_sp_anchor[..., 0::2, 1::2] = 0
                y_sp_anchor[..., 1::2, 0::2] = 0
                y_sp_ctx = self.contextiter._sub_modules_space[i](y_sp_anchor, self.stage)
                # set ctx anchor part to zero
                y_sp_ctx[..., 0::2, 0::2] = 0
                y_sp_ctx[..., 1::2, 1::2] = 0
                mu_hat, sigma_hat = torch.chunk(self.y_parameter._sub_modules[i](torch.cat([y_sp_ctx, y_ch_ctx, hyper], 1)), 2, 1)
                
                mu_list.append(mu_hat)
                sigma_list.append(sigma_hat)
                # y_new_slice = self.ste_quant(y_new_slice - mu_hat) + mu_hat
                y_new_slice = self.y_entorpy.quantize(y_new_slice, "dequantize", means=mu_hat, scales=sigma_hat)
                y_slices = torch.cat([y_slices, y_new_slice], 1)
            
            mu = torch.cat(mu_list, 1)
            sigma = torch.cat(sigma_list, 1)
            
            ## skip propotion
            # skip_threshold = self.y_entorpy.skipThreshold
            # skip_proportion = (sigma <= skip_threshold).sum() / sigma.numel()
            # print(skip_proportion.item())
            
            y_string_list = []
            anchor_div_non_anchor = []
            diff = []
            for y_slice, mu, sigma in zip(self.sliceModel(y_slices), mu_list, sigma_list):
                y1, y2 = self.Demux(y_slice)
                mu1, mu2 = self.Demux(mu)
                sigma1, sigma2 = self.Demux(sigma)
                indexes_y1 = self.y_entorpy.build_indexes(sigma1)
                indexes_y2 = self.y_entorpy.build_indexes(sigma2)
                y1_strings = self.y_entorpy.compress(y1, indexes_y1, means=mu1)
                y2_strings = self.y_entorpy.compress(y2, indexes_y2, means=mu2)
                y_string_list.append(y1_strings)
                y_string_list.append(y2_strings)
                
                anchor_div_non_anchor.append(len(y1_strings[0]) / len(y2_strings[0]))
                diff.append(len(y1_strings[0]) - len(y2_strings[0]))
            
            # print(sum(anchor_div_non_anchor) / len(anchor_div_non_anchor))
            print(sum(diff))
            
            return {
                "strings": [*y_string_list, z_strings],
                "shape": z.size()[-2:],
                "y" : y_slices if self.stage == CodecStageEnum.Check else None,
                "first_mu":first_mu if self.stage == CodecStageEnum.Check else None, 
                "first_sigma":first_sigma if self.stage == CodecStageEnum.Check else None,
            }
    
    @torch.inference_mode()
    def decompress(self, strings, shape):
        z_hat = self.entropy_bottleneck.decompress(strings[-1], shape)
        hyper = self.h_s(z_hat)
        shape_in = hyper.shape
        
        # first stage, without channle context
        zero_ctx = torch.zeros(shape_in[0],self.groups[0] * 2, shape_in[2], shape_in[3]).to(hyper.device)
        mu, sigma = torch.chunk(self.y_parameter._sub_modules[0](torch.cat([zero_ctx, hyper], 1)), 2, 1)
        mu_hat, _ = self.Demux(mu)
        sigma_hat, _ = self.Demux(sigma)
        indexes_y1 = self.y_entorpy.build_indexes(sigma_hat)
        y1 = self.y_entorpy.decompress(strings[0],indexes_y1, means=mu_hat)
        
        y_sp1_anchor = self.Mux(y1, torch.zeros_like(y1))
        y_sp1_ctx = self.contextiter._sub_modules_space[0](y_sp1_anchor, self.stage)
        # set ctx anchor part ctx to zero
        # decompress stage neccessary?
        # y_sp1_ctx[..., 0::2, 0::2] = 0
        # y_sp1_ctx[..., 1::2, 1::2] = 0
        mu, sigma = torch.chunk(self.y_parameter._sub_modules[0](torch.cat([y_sp1_ctx, hyper], 1)), 2, 1)
        _, mu_hat = self.Demux(mu)
        _, sigma_hat = self.Demux(sigma)
        indexes_y1 = self.y_entorpy.build_indexes(sigma_hat)
        y_slices = self.y_entorpy.decompress(strings[1], indexes_y1, means=mu_hat)
        y_slices = self.Mux(torch.zeros_like(y_slices), y_slices)
        
        y_slices += y_sp1_anchor
        
        for i in range(1, self.n_groups):
            zero_ctx = torch.zeros(shape_in[0],self.groups[i] * 2, shape_in[2], shape_in[3]).to(hyper.device)
            y_ch_ctx = self.contextiter._sub_modules_channel[i-1](y_slices)
            mu, sigma = torch.chunk(self.y_parameter._sub_modules[i](torch.cat([zero_ctx, y_ch_ctx, hyper], 1)), 2, 1)
            mu_hat, _ = self.Demux(mu)
            sigma_hat, _ = self.Demux(sigma)
            indexes_y = self.y_entorpy.build_indexes(sigma_hat)
            y = self.y_entorpy.decompress(strings[i * 2], indexes_y, means=mu_hat)
            
            y_sp_anchor = self.Mux(y, torch.zeros_like(y))
            y_sp_ctx = self.contextiter._sub_modules_space[i](y_sp_anchor, self.stage)
            # y_sp_ctx[..., 0::2, 0::2] = 0
            # y_sp_ctx[..., 1::2, 1::2] = 0
            mu, sigma = torch.chunk(self.y_parameter._sub_modules[i](torch.cat([y_sp_ctx, y_ch_ctx, hyper], 1)), 2, 1)
            _, mu_hat = self.Demux(mu)
            _, sigma_hat = self.Demux(sigma)
            indexes_y = self.y_entorpy.build_indexes(sigma_hat)
            y = self.y_entorpy.decompress(strings[i * 2 + 1], indexes_y, means=mu_hat)
            y = self.Mux(torch.zeros_like(y), y)
            
            y_new_slice = y_sp_anchor + y
            y_slices = torch.cat([y_slices, y_new_slice], dim=1)
        
        x_hat = self.g_s(y_slices)
        return {
            "x_hat" : x_hat,
            "y" : y_slices if self.stage == CodecStageEnum.Check else None
        }
#%%
if __name__ == "__main__":
    from PIL import Image
    from torchvision.transforms import ToTensor
    from compressai.models.utils import Sobel
    sobel = Sobel()
    torch.backends.cudnn.deterministic = True
    
    groups = [16,16,32,64,None]
    
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

    # x = torch.rand((1,3,768,512)).cuda()
    img = Image.open('/hdd/zyw/ImageDataset/kodak/kodim01.png').convert("RGB")
    # img = Image.open('/home/zyw/original.jpg').convert("RGB")
    x = ToTensor()(img).cuda()
    x = x.unsqueeze(0)
    
    net = ELIC(skipIndex=1).cuda()
    net.eval()
    net.stage = CodecStageEnum.Check
    
    checkpoint = torch.load('pretrained/elic/6/skip_best.pth.tar')
    net.load_state_dict(checkpoint['state_dict'])
    
    net.update(force=True)
    
    out = net(x)
    
    ## 验证编解码正确性
    code = net.compress(x)
    rec = net.decompress(code['strings'], code['shape'])
    y1 = code['y']
    y2 = rec['y']
    total_bytes = 0                
    strings = code['strings']
    for s in strings:
        if isinstance(s, list):
            for i in s:
                total_bytes += len(i)
        else:
            total_bytes += len(i)

    print(total_bytes * 8 / 768 / 512)
    print((y1 == y2).all())

    x_hat = rec['x_hat']
    
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
    from examples.train_elic import RateDistortionLoss
    loss = RateDistortionLoss(0.045)
    res = loss(out, x)
    print(res['bpp_loss'].item())
    print(10 * torch.log10(1 / res["mse_loss"]).item())
    print(res['loss'])

    s1, s2 = code['first_sigma'][0].clone(), code['first_sigma'][1].clone()
    s1[..., 0::2, 0::2] = 0
    s1[..., 1::2, 1::2] = 0
    s2[..., 0::2, 0::2] = 0
    s2[..., 1::2, 1::2] = 0

    
    m1, m2 = code['first_mu'][0].clone(), code['first_mu'][1].clone()
    m1[..., 0::2, 0::2] = 0
    m1[..., 1::2, 1::2] = 0
    m2[..., 0::2, 0::2] = 0
    m2[..., 1::2, 1::2] = 0

    s1_pos = s1 <= 0.1592
    s2_pos = s2 <= 0.1592
    
    s1[~s1_pos] = 0
    s2[~s2_pos] = 0
    import seaborn as sns
    sns.heatmap((s2-s1)[0,0,...].cpu())
    sns.heatmap((m2-m1)[0,0,...].cpu())
    
    wrong_pos = s1_pos ^ s2_pos
    s2[s1==s1[wrong_pos].max()]
    
#%%
    # import matplotlib.pyplot as plt
    # y = out['y']
    # hyper = out['hyper']
    # y = y.detach().cpu()
    # hyper = hyper.detach().cpu()
    # sy = sobel(y[:, 0:1, ...])
    # shyper = sobel(hyper[:, 0:1, ...])
# %%
