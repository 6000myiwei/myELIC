import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn import functional

class Quantizator_RT(nn.Module):
    """
    uniform noise quant-estimator
    """
    def __init__(self, B=1, train=True, key_in=None, key_out=None):
        super(Quantizator_RT, self).__init__()
        self.B = B
        self.training = train
        self.factor = (1 << self.B) - 1
        self.key_in = key_in
        self.key_out = key_out

    def forward(self, input):
        factor = self.factor
        if self.training:
            x = torch.rand_like(input)
            x -= 0.5
            if factor != 1:
                x /= factor
            return input + x
        else:
            return torch.round(input * factor) / factor


class StraightThroughEstimatorFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad):
        return grad

        
class StraightThroughScaleSKip(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, skipThreshold):
        return torch.where(x > skipThreshold, x, torch.zeros_like(x))

    @staticmethod
    def backward(ctx, grad):
        return grad, None


class SkipPosZero(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, skip_pos):
        x[skip_pos] = 0.
        return x

    @staticmethod
    def backward(ctx, grad):
        return grad, None
class STEQuant(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return StraightThroughEstimatorFunc.apply(x)  # B = 1

class STEQuantWithSkip(STEQuant):
    def __init__(self, skipThreshold=None):
        super().__init__()
        self.skipThreshold = skipThreshold
    
    def STEScaleUpdate(self, scales):
        return StraightThroughScaleSKip.apply(scales, self.skipThreshold)
        # skip_scales = scales.clone()
        # skip_pos = scales <= self.skipThreshold
        # skip_scales[skip_pos] = 0
        # scales = scales + (skip_scales - scales).detach()
        # return scales
    
    def forward(self, x, means=None, scales=None):
        if means is None and scales is None:
            return super().forward(x)
        else:
            outputs = x.clone()
            skip_pos = scales <= self.skipThreshold
            
            outputs -= means
            outputs = StraightThroughEstimatorFunc.apply(outputs) + means
            outputs = ~skip_pos * outputs + skip_pos * means
            
            return outputs
        
            # # STE strategy
            # outputs -= means
            # outputs = SkipPosZero.apply(outputs, skip_pos)
            
            # skip_outputs = outputs.clone()
            # skip_outputs[skip_pos] = 0.
            # outputs = outputs + (skip_outputs - outputs).detach()
            # outputs = skip_outputs
            # return StraightThroughEstimatorFunc.apply(outputs) + means