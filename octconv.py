import torch
from functools import partial


class OctConv(torch.nn.Module):
    """
    This module implements the OctConv paper https://arxiv.org/pdf/1904.05049v1.pdf
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, alpha=0.5):
        super(OctConv, self).__init__()
        self.kernel_size = kernel_size
        self.L2L = torch.nn.Conv2d(int(alpha * in_channels), int(alpha * out_channels),
                                   kernel_size, stride, kernel_size//2)
        self.L2H = torch.nn.Conv2d(int(alpha * in_channels), out_channels - int(alpha * out_channels),
                                   kernel_size, stride, kernel_size//2)
        self.H2L = torch.nn.Conv2d(in_channels - int(alpha * in_channels), int(alpha * out_channels),
                                   kernel_size, stride, kernel_size//2)
        self.H2H = torch.nn.Conv2d(in_channels - int(alpha * in_channels),
                                   out_channels - int(alpha * out_channels),
                                   kernel_size, stride, kernel_size//2)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')
        self.avg_pool = partial(torch.nn.functional.avg_pool2d, kernel_size=kernel_size, stride=kernel_size)

    def forward(self, x):
        hf, lf = x
        return self.H2H(hf) + self.upsample(self.L2H(lf)), self.L2L(lf) + self.avg_pool(hf)

