from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn


class _GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return -ctx.lambd * grad_output, None


class GradientReversalLayer(nn.Module):
    def forward(self, x: torch.Tensor, *, lambd: float) -> torch.Tensor:
        return _GradientReversalFn.apply(x, float(lambd))


class FeatureExtractorPhiF(nn.Module):
    def __init__(self, *, in_channels: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.net(x)
        return torch.flatten(f, start_dim=1)


class TaskClassifierPhiC(nn.Module):
    def __init__(self, *, n_class: int):
        super().__init__()
        self.net = nn.Sequential(
            # nn.Linear(128, 128),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p=0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(64, int(n_class)),
        )

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        return self.net(f)


class DomainDiscriminatorPhiD(nn.Module):
    def __init__(self, *, n_domain: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, int(n_domain)),
        )

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        return self.net(f)


@dataclass(frozen=True)
class MSDGOutputs:
    f: torch.Tensor
    y_logits: torch.Tensor
    d_logits: torch.Tensor


class Model(nn.Module):
    def __init__(self, *, n_class: int = 8, n_domain: int = 4, in_channels: int = 2):
        super().__init__()
        self.phi_f = FeatureExtractorPhiF(in_channels=in_channels)
        self.phi_c = TaskClassifierPhiC(n_class=n_class)
        self.phi_d = DomainDiscriminatorPhiD(n_domain=n_domain)
        self.grl = GradientReversalLayer()

    def forward(self, x: torch.Tensor, *, lambd: float) -> MSDGOutputs:
        f = self.phi_f(x)
        y_logits = self.phi_c(f)

        f_grl = self.grl(f, lambd=float(lambd))
        d_logits = self.phi_d(f_grl)
        return MSDGOutputs(f=f, y_logits=y_logits, d_logits=d_logits)
