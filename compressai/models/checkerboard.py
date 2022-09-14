from codecs import Codec
import torch
import torch.nn as nn

from compressai.layers import (
    AttentionBlock,
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    conv3x3,
    subpel_conv3x3,
)
from compressai.layers.quant import STEQuant
from compressai.models.elic import CheckerboardContext, CodecStageEnum, ParamGroup, ParamGroupTwoPath
from compressai.models.utils import CheckerboardMux, CheckerboardDemux, update_registered_buffers

from compressai.models.google import JointAutoregressiveHierarchicalPriors

from compressai.models.elic_ada5 import AdaptiveContext, GetMask, PoolConv


class Minen18Checkerboard(JointAutoregressiveHierarchicalPriors):
    def __init__(self, N=192, M=192, stage=CodecStageEnum.TRAIN, **kwargs):
        super().__init__(N=N, M=M, **kwargs)
        
        self.context_prediction = CheckerboardContext(
            M, 2 * M, kernel_size=5, padding=2, stride=1, bias=True
        )
        self.Mux = CheckerboardMux
        self.Demux = CheckerboardDemux
        self.stage = stage
        self.ste_quant = STEQuant()
    
    # def forward(self, x):
    #     y = self.g_a(x)
    #     z = self.h_a(y)
    #     z_round = self.ste_quant(z)
    #     z_hat, z_likelihoods = self.entropy_bottleneck(z)
    #     y_hat = self.gaussian_conditional.quantize(
    #         y, "noise" if self.training else "dequantize"
    #     )
    #     y_round = self.ste_quant(y)
    #     # hyper_params = self.h_s(z_hat)
    #     hyper_params = self.h_s(z_round)
    #     ctx_params = self.context_prediction(y_round, CodecStageEnum.TRAIN2)
    #     # mask anchor
    #     ctx_params[:, :, 0::2, 0::2] = 0
    #     ctx_params[:, :, 1::2, 1::2] = 0
    #     gaussian_params = self.entropy_parameters(torch.cat([ctx_params, hyper_params], dim=1))
    #     scales_hat, means_hat = gaussian_params.chunk(2, 1)
    #     _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
    #     x_hat = self.g_s(y_round)
    #     return {
    #         "x_hat": x_hat,
    #         "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
    #     }

    def forward(self, x):
        if self.stage == CodecStageEnum.TRAIN:
            y = self.g_a(x)
            z = self.h_a(y)
            z_hat, z_likelihoods = self.entropy_bottleneck(z)
            params = self.h_s(z_hat)

            y_hat = self.gaussian_conditional.quantize(
                y, "noise" if self.training else "dequantize"
            )
            ctx_params = self.context_prediction(y_hat, self.stage)
            
            gaussian_params1 = self.entropy_parameters(
                torch.cat((params, ctx_params[0]), dim=1)
            )
            scales_hat1, means_hat1 = gaussian_params1.chunk(2, 1)
            
            gaussian_params2 = self.entropy_parameters(
                torch.cat((params, ctx_params[1]), dim=1)
            )
            scales_hat2, means_hat2 = gaussian_params2.chunk(2, 1)
            
            _, y_likelihoods1 = self.gaussian_conditional(y, scales_hat1, means=means_hat1)
            _, y_likelihoods2 = self.gaussian_conditional(y, scales_hat2, means=means_hat2)
            x_hat = self.g_s(y_hat)

            return {
                "x_hat": x_hat,
                "likelihoods": {"y1": y_likelihoods1, "y2": y_likelihoods2,"z": z_likelihoods},
            }
        elif self.stage == CodecStageEnum.TRAIN2:
            y = self.g_a(x)
            z = self.h_a(y)
            z_hat, z_likelihoods = self.entropy_bottleneck(z)
            params = self.h_s(z_hat)
            
            shape_in = y.shape
            y_hat = self.gaussian_conditional.quantize(
                y, "noise" if self.training else "dequantize"
            )
            zero_ctx = torch.zeros(shape_in[0], shape_in[1] * 2, shape_in[2], shape_in[3]).to(y.device)
            
            gaussian_params1 = self.entropy_parameters(
                torch.cat((params, zero_ctx), dim=1)
            )
            scales_hat1, means_hat1 = gaussian_params1.chunk(2, 1)
            
            # set y_hat non anchor part to zero
            y_anchor = y_hat.clone()
            y_anchor[..., 0::2, 1::2] = 0
            y_anchor[..., 1::2, 0::2] = 0
            
            ctx_params = self.context_prediction(y_anchor, self.stage)
            # set ctx anchor part ctx to zero
            ctx_params[..., 0::2, 0::2] = 0
            ctx_params[..., 1::2, 1::2] = 0
            
            gaussian_params2 = self.entropy_parameters(
                torch.cat((params, ctx_params), dim=1)
            )
            
            scales_hat2, means_hat2 = gaussian_params2.chunk(2, 1)
            
            _, y_likelihoods = self.gaussian_conditional(y, scales_hat2, means=means_hat2)
            x_hat = self.g_s(y_hat)
            return {
                "x_hat": x_hat,
                "likelihoods": {"y": y_likelihoods,"z": z_likelihoods},
            }

    def compress(self, x):
        y = self.g_a(x)
        z = self.h_a(y)
        
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        
        params = self.h_s(z_hat)
        
        shape_in = y.shape

        zero_ctx = torch.zeros(shape_in[0], shape_in[1] * 2, shape_in[2], shape_in[3]).to(y.device)
        
        gaussian_params1 = self.entropy_parameters(
            torch.cat((params, zero_ctx), dim=1)
        )
        scales_hat1, means_hat1 = gaussian_params1.chunk(2, 1)

        # set y_hat non anchor part to zero
        y_anchor = self.gaussian_conditional.quantize(y, "dequantize", means=means_hat1)
        y_anchor[..., 0::2, 1::2] = 0
        y_anchor[..., 1::2, 0::2] = 0
        
        ctx_params = self.context_prediction(y_anchor, self.stage)
        # set ctx anchor part ctx to zero
        ctx_params[..., 0::2, 0::2] = 0
        ctx_params[..., 1::2, 1::2] = 0
        
        gaussian_params2 = self.entropy_parameters(
            torch.cat((params, ctx_params), dim=1)
        )
        
        scales_hat2, means_hat2 = gaussian_params2.chunk(2, 1)
        y_hat = self.gaussian_conditional.quantize(y, "dequantize", means=means_hat2)
        
        y1, y2 = self.Demux(y_hat)
        mu1, mu2 = self.Demux(means_hat2)
        sigma1, sigma2 = self.Demux(scales_hat2)
        indexes_y1 = self.gaussian_conditional.build_indexes(sigma1)
        indexes_y2 = self.gaussian_conditional.build_indexes(sigma2)
        y1_strings = self.gaussian_conditional.compress(y1, indexes_y1, means=mu1)
        y2_strings = self.gaussian_conditional.compress(y2, indexes_y2, means=mu2)
    
        return {
            "strings": [y1_strings, y2_strings, z_strings],
            "shape": z.size()[-2:],
            "y" : y_hat if self.stage == CodecStageEnum.Check else None
        }
        
    def decompress(self, strings, shape):
        z_hat = self.entropy_bottleneck.decompress(strings[-1], shape)
        params = self.h_s(z_hat)
        shape_in = params.shape
        
        zero_ctx = torch.zeros(shape_in[0], shape_in[1], shape_in[2], shape_in[3]).to(params.device)
        
        gaussian_params1 = self.entropy_parameters(
            torch.cat((params, zero_ctx), dim=1)
        )
        scales_hat1, means_hat1 = gaussian_params1.chunk(2, 1)
        
        scales_hat1, _ = self.Demux(scales_hat1)
        means_hat1, _ = self.Demux(means_hat1)
        indexes_y1 = self.gaussian_conditional.build_indexes(scales_hat1)        
        y1 = self.gaussian_conditional.decompress(strings[0],indexes_y1, means=means_hat1)
        
        y_anchor = self.Mux(y1, torch.zeros_like(y1))
        ctx_params = self.context_prediction(y_anchor, self.stage)
       
        gaussian_params2 = self.entropy_parameters(
            torch.cat((params, ctx_params), dim=1)
        )
        scales_hat2, means_hat2 = gaussian_params2.chunk(2, 1)
        _, scales_hat2 = self.Demux(scales_hat2)
        _, means_hat2 = self.Demux(means_hat2)
        
        indexes_y2 = self.gaussian_conditional.build_indexes(scales_hat2)
        y_non_anchor = self.gaussian_conditional.decompress(strings[1], indexes_y2, means=means_hat2)
        y_non_anchor = self.Mux(torch.zeros_like(y_non_anchor), y_non_anchor)
        
        y = y_anchor + y_non_anchor
        x_hat = self.g_s(y)
        
        return {
            "x_hat" : x_hat,
            "y" : y if self.stage == CodecStageEnum.Check else None
        }


class Minen18ADA(Minen18Checkerboard):
    def __init__(self, N=192, M=192, stage=CodecStageEnum.TRAIN, **kwargs):
        super().__init__(N=N, M=M, **kwargs)
        self.context_prediction = AdaptiveContext(M, 2 * M, 5)
        self.ada_mask = GetMask()
        self.pool_conv = PoolConv(M)
        self.ste_qunat = STEQuant()
        self.stage = stage
    
    def fill_pooling_value(self, y_anchor, mu, sigma, values_mask):
        values = mu * values_mask
        values = self.pool_conv(values)
        return values + y_anchor
    
    def forward(self, x):
            y = self.g_a(x)
            z = self.h_a(y)
            z_hat, z_likelihoods = self.entropy_bottleneck(z)
            params = self.h_s(z_hat)

            # z_round = self.ste_qunat(z)
            # params = self.h_s(z_round)
            
            shape_in = y.shape
            y_hat = self.gaussian_conditional.quantize(
                y, "noise" if self.training else "dequantize"
            )
            zero_ctx = torch.zeros(shape_in[0], shape_in[1] * 2, shape_in[2], shape_in[3]).to(y.device)
            
            gaussian_params1 = self.entropy_parameters(
                torch.cat((params, zero_ctx), dim=1)
            )
            scales_hat1, means_hat1 = gaussian_params1.chunk(2, 1)
            mask = self.ada_mask(scales_hat1)
            
            # set y_hat non anchor part to zero
            # y_anchor = self.ste_qunat(y - means_hat1) + means_hat1
            y_anchor = y_hat.clone()
            y_anchor *= mask[0]
            y_anchor = self.fill_pooling_value(y_anchor, means_hat1, scales_hat1, mask[1])
            
            ctx_params = self.context_prediction(y_anchor, self.stage)
            # set ctx anchor part ctx to zero
            
            gaussian_params2 = self.entropy_parameters(
                torch.cat((params, ctx_params), dim=1)
            )
            
            scales_hat2, means_hat2 = gaussian_params2.chunk(2, 1)
            scales_hat2 = scales_hat2 * (1 - mask[0]) + scales_hat1 * mask[0]
            means_hat2 = means_hat2 * (1 - mask[0]) + means_hat1 * mask[0]
            
            _, y_likelihoods = self.gaussian_conditional(y, scales_hat2, means=means_hat2)
            x_hat = self.g_s(y_hat)

            # y_quant = self.ste_qunat(y - means_hat2) + means_hat2    
            # x_hat = self.g_s(y_quant)
            return {
                "y":y,
                "x_hat": x_hat,
                "likelihoods": {"y": y_likelihoods,"z": z_likelihoods},
                "mask": mask[0]
            }
    
    def compress(self, x):
        y = self.g_a(x)
        z = self.h_a(y)
        
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        
        params = self.h_s(z_hat)
        
        shape_in = y.shape

        zero_ctx = torch.zeros(shape_in[0], shape_in[1] * 2, shape_in[2], shape_in[3]).to(y.device)
        
        gaussian_params1 = self.entropy_parameters(
            torch.cat((params, zero_ctx), dim=1)
        )
        scales_hat1, means_hat1 = gaussian_params1.chunk(2, 1)
        
        mask = self.ada_mask(scales_hat1)
        anchor_index = mask[0] == 1
        non_anchor_index = ~anchor_index

        # set y_hat non anchor part to zero
        y_anchor = self.gaussian_conditional.quantize(y, "dequantize", means=means_hat1)
        y_anchor *= mask[0]
        y_anchor = self.fill_pooling_value(y_anchor, means_hat1, scales_hat1, mask[1])
        
        ctx_params = self.context_prediction(y_anchor, self.stage)
        # set ctx anchor part ctx to zero
        gaussian_params2 = self.entropy_parameters(
            torch.cat((params, ctx_params), dim=1)
        )
        
        scales_hat2, means_hat2 = gaussian_params2.chunk(2, 1)
        scales_hat2 = scales_hat2 * (1 - mask[0]) + scales_hat1 * mask[0]
        means_hat2 = means_hat2 * (1 - mask[0]) + means_hat1 * mask[0]
        
        y_hat = self.gaussian_conditional.quantize(y, "dequantize", means=means_hat2)
        
        y1, y2 = y_hat[:, anchor_index[0]], y_hat[:, non_anchor_index[0]]
        mu1, mu2 = means_hat2[:, anchor_index[0]], means_hat2[:, non_anchor_index[0]]
        sigma1, sigma2 = scales_hat2[:, anchor_index[0]], scales_hat2[:, non_anchor_index[0]]
        
        indexes_y1 = self.gaussian_conditional.build_indexes(sigma1)
        indexes_y2 = self.gaussian_conditional.build_indexes(sigma2)
        y1_strings = self.gaussian_conditional.compress(y1, indexes_y1, means=mu1)
        y2_strings = self.gaussian_conditional.compress(y2, indexes_y2, means=mu2)
    
        return {
            "strings": [y1_strings, y2_strings, z_strings],
            "shape": z.size()[-2:],
            "y" : y_hat if self.stage == CodecStageEnum.Check else None
        }
        
    def decompress(self, strings, shape):
        z_hat = self.entropy_bottleneck.decompress(strings[-1], shape)
        params = self.h_s(z_hat)
        shape_in = params.shape
        
        zero_ctx = torch.zeros(shape_in[0], shape_in[1], shape_in[2], shape_in[3]).to(params.device)
        
        gaussian_params1 = self.entropy_parameters(
            torch.cat((params, zero_ctx), dim=1)
        )
        scales_hat, means_hat = gaussian_params1.chunk(2, 1)
        mask = self.ada_mask(scales_hat)
        anchor_index = mask[0] == 1
        non_anchor_index = ~anchor_index
        
        y_anchor = torch.zeros_like(means_hat)
        scales_hat1 = scales_hat[:, anchor_index[0]]
        means_hat1 = means_hat[:, anchor_index[0]]
        
        indexes_y1 = self.gaussian_conditional.build_indexes(scales_hat1)        
        y1 = self.gaussian_conditional.decompress(strings[0],indexes_y1, means=means_hat1)
       
        y_anchor[anchor_index] = y1 
        y_anchor_ctx_in = self.fill_pooling_value(y_anchor, means_hat, scales_hat, mask[1])
        ctx_params = self.context_prediction(y_anchor_ctx_in, self.stage)
       
        gaussian_params2 = self.entropy_parameters(
            torch.cat((params, ctx_params), dim=1)
        )
        scales_hat2, means_hat2 = gaussian_params2.chunk(2, 1)
        
        y_non_anchor = torch.zeros_like(scales_hat2)
        
        scales_hat2 = scales_hat2[:, non_anchor_index[0]]
        means_hat2 = means_hat2[:, non_anchor_index[0]]
        
        indexes_y2 = self.gaussian_conditional.build_indexes(scales_hat2)
        y2 = self.gaussian_conditional.decompress(strings[1], indexes_y2, means=means_hat2)
        y_non_anchor[non_anchor_index] = y2
        
        y = y_anchor + y_non_anchor
        x_hat = self.g_s(y)
        
        return {
            "x_hat" : x_hat,
            "y" : y if self.stage == CodecStageEnum.Check else None
        }
#%%
if __name__ == "__main__":
    from PIL import Image
    from torchvision.transforms import ToTensor
    torch.backends.cudnn.deterministic = True
    
    
    img = Image.open('/hdd/zyw/ImageDataset/kodak/kodim01.png').convert("RGB")
    # img = Image.open('/home/zyw/original.jpg').convert("RGB")
    x = ToTensor()(img).cuda()
    x = x.unsqueeze(0)
    
    net = Minen18Checkerboard().cuda()
    net.eval()
    net.stage = CodecStageEnum.Check
    
    # checkpoint = torch.load('pretrained/cheng2020-ada/6/checkpoint.pth.tar')
    # net.load_state_dict(checkpoint['state_dict'])
    
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
    
    
    # import matplotlib.pyplot as plt
    # import seaborn as sns
    # m = out['mask'][0]
    # y = out['y']
    # # hyper = out['hyper']
    # # mu, sigma = out["first_param"]
    # # mu_list, sigma_list = out['param_list']
    # plt.imshow(y[0,0,...].cpu().detach())
    # plt.imshow(m[0,0,...].cpu().detach())
#%%