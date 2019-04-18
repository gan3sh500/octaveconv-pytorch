import torch
from functools import partial


class OctConv(torch.nn.Module):
    """
    This module implements the OctConv paper https://arxiv.org/pdf/1904.05049v1.pdf
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, alpha_in=0.5, alpha_out=0.5):
        super(OctConv, self).__init__()
        self.alpha_in, self.alpha_out, self.kernel_size = alpha_in, alpha_out, kernel_size
        self.H2H, self.L2L, self.H2L, self.L2H = None, None, None, None
        if not (alpha_in == 0.0 and alpha_out == 0.0):
            self.L2L = torch.nn.Conv2d(int(alpha_in * in_channels),
                                       int(alpha_out * out_channels),
                                       kernel_size, stride, kernel_size//2)
        if not (alpha_in == 0.0 and alpha_out == 1.0):
            self.L2H = torch.nn.Conv2d(int(alpha_in * in_channels),
                                       out_channels - int(alpha_out * out_channels),
                                       kernel_size, stride, kernel_size//2)
        if not (alpha_in == 1.0 and alpha_out == 0.0):
            self.H2L = torch.nn.Conv2d(in_channels - int(alpha_in * in_channels),
                                       int(alpha_out * out_channels),
                                       kernel_size, stride, kernel_size//2)
        if not (alpha_in == 1.0 and alpha_out == 1.0):
            self.H2H = torch.nn.Conv2d(in_channels - int(alpha_in * in_channels),
                                       out_channels - int(alpha_out * out_channels),
                                       kernel_size, stride, kernel_size//2)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')
        self.avg_pool = partial(torch.nn.functional.avg_pool2d, kernel_size=kernel_size, stride=kernel_size)

    def forward(self, x):
        hf, lf = x
        h2h, l2l, h2l, l2h = None, None, None, None
        if self.H2H is not None:
            h2h = self.H2H(hf)
        if self.L2L is not None:
            l2l = self.L2L(lf)
        if self.H2L is not None:
            h2l = self.H2L(self.avg_pool(hf))
        if self.L2H is not None:
            l2h = self.upsample(self.L2H(lf))
        hf_, lf_ = 0, 0
        for i in [h2h, l2h]:
            if i is not None:
                hf_ = hf_ + i
        for i in [l2l, h2l]:
            if i is not None:
                lf_ = lf_ + i
        return hf_, lf_

