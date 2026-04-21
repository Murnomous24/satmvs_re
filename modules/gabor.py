import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaborLayer(nn.Module):
    """Generate learnable Gabor responses from a single-channel input."""

    def __init__(
            self,
            orientations = None,
            kernel_size = 7,
            sigma = 2.0,
            lambd = 4.0,
            gamma = 0.5
    ):
        super().__init__()

        if orientations is None:
            orientations = [0.0, math.pi / 2.0]

        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        kernels = [self._make_gabor_kernel(kernel_size, sigma, lambd, gamma, theta) for theta in orientations]
        weight = torch.stack(kernels, dim = 0).unsqueeze(1)

        self.weight = nn.Parameter(weight, requires_grad = True)
        self.bias = nn.Parameter(torch.zeros(len(orientations), dtype = torch.float32), requires_grad = True)

    @staticmethod
    def _make_gabor_kernel(ksize, sigma, lambd, gamma, theta):
        half = ksize // 2
        y, x = torch.meshgrid(
            torch.arange(-half, half + 1, dtype = torch.float32),
            torch.arange(-half, half + 1, dtype = torch.float32),
            indexing = "ij"
        )

        x_theta = x * math.cos(theta) + y * math.sin(theta)
        y_theta = -x * math.sin(theta) + y * math.cos(theta)

        gaussian = torch.exp(-(x_theta ** 2 + (gamma ** 2) * (y_theta ** 2)) / (2.0 * sigma ** 2))
        sinusoid = torch.cos(2.0 * math.pi * x_theta / lambd)
        kernel = gaussian * sinusoid

        norm = kernel.abs().sum()
        if norm > 0:
            kernel = kernel / norm

        return kernel

    def forward(self, x):
        if x.shape[1] != 1:
            raise ValueError(f"GaborLayer expects 1 input channel, but receive {x.shape[1]}")

        response = F.conv2d(x, self.weight, self.bias, padding = self.padding)
        return torch.abs(response)
