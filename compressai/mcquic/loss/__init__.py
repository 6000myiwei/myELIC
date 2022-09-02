from torch import nn
import torch.nn.functional as F


class Distortion(nn.Module):
    def __init__(self, formatter: nn.Module):
        super().__init__()
        self._formatter = formatter

    def formatDistortion(self, loss):
        return self._formatter(loss)


class Rate(nn.Module):
    def __init__(self, formatter: nn.Module):
        super().__init__()
        self._formatter = formatter

    def formatRate(self, loss):
        return self._formatter(loss)


class BasicRate(Rate):
    def __init__(self, gamma: float = 1e-7):
        super().__init__(nn.Identity())
        self._gamma = gamma

    def _cosineLoss(self, codebook):
        losses = list()
        # m * [k, d]
        for c in codebook:
            # [k, k]
            pairwise = c @ c.T
            norm = (c ** 2).sum(-1)
            cos = pairwise / (norm[:, None] * norm).sqrt()
            losses.append(cos.triu(1).clamp(0.0, 2).sum())
        return sum(losses)

    def forward(self, logits, codebooks, *_):
        return sum(self._cosineLoss(codebook) for codebook in codebooks)


