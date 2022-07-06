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

        
class STEQuant(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return StraightThroughEstimatorFunc.apply(x)  # B = 1