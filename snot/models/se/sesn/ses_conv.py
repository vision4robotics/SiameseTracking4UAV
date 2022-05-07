import torch
import torch.nn as nn
import torch.nn.functional as F

from snot.models.se.sesn.ses_basis import steerable_A, normalize_basis_by_min_scale


class SESConv_Z2_H(nn.Module):
    '''Scale Equivariant Steerable Convolution: Z2 -> (S x Z2)
    [B, C, H, W] -> [B, C', S, H', W']

    Args:
        in_channels: Number of channels in the input image
        out_channels: Number of channels produced by the convolution
        kernel_size: Size of the convolving kernel
        effective_size: The effective size of the kernel with the same # of params
        scales: List of scales of basis
        stride: Stride of the convolution
        padding: Zero-padding added to both sides of the input
        bias: If ``True``, adds a learnable bias to the output
    '''

    def __init__(self, in_channels, out_channels, kernel_size, effective_size,
                 scales=[1.0], stride=1, padding=0, bias=False, padding_mode='constant', **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.effective_size = effective_size
        self.scales = [round(s, 3) for s in scales]
        self.num_scales = len(scales)
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode

        basis = steerable_A(kernel_size, scales, effective_size, **kwargs)
        basis = normalize_basis_by_min_scale(basis)
        self.register_buffer('basis', basis)

        self.num_funcs = self.basis.size(0)

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, self.num_funcs))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        basis = self.basis.view(self.num_funcs, -1)
        kernel = self.weight @ basis
        kernel = kernel.view(self.out_channels, self.in_channels,
                             self.num_scales, self.kernel_size, self.kernel_size)
        kernel = kernel.permute(0, 2, 1, 3, 4).contiguous()
        kernel = kernel.view(-1, self.in_channels, self.kernel_size, self.kernel_size)

        # convolution
        if self.padding > 0:
            x = F.pad(x, 4 * [self.padding], mode=self.padding_mode)
        y = F.conv2d(x, kernel, bias=None, stride=self.stride)
        B, C, H, W = y.shape
        y = y.view(B, self.out_channels, self.num_scales, H, W)

        if self.bias is not None:
            y = y + self.bias.view(1, -1, 1, 1, 1)

        return y

    def extra_repr(self):
        s = '{in_channels}->{out_channels} | scales={scales} | size={kernel_size}'
        return s.format(**self.__dict__)


class SESConv_H_H(nn.Module):
    '''Scale Equivariant Steerable Convolution: (S x Z2) -> (S x Z2)
    [B, C, S, H, W] -> [B, C', S', H', W']

    Args:
        in_channels: Number of channels in the input image
        out_channels: Number of channels produced by the convolution
        scale_size: Size of scale filter
        kernel_size: Size of the convolving kernel
        effective_size: The effective size of the kernel with the same # of params
        scales: List of scales of basis
        stride: Stride of the convolution
        padding: Zero-padding added to both sides of the input
        bias: If ``True``, adds a learnable bias to the output
    '''

    def __init__(self, in_channels, out_channels, scale_size, kernel_size, effective_size,
                 scales=[1.0], stride=1, padding=0, bias=False, padding_mode='constant', **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_size = scale_size
        self.kernel_size = kernel_size
        self.effective_size = effective_size
        self.scales = [round(s, 3) for s in scales]
        self.num_scales = len(scales)
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode

        basis = steerable_A(kernel_size, scales, effective_size, **kwargs)
        basis = normalize_basis_by_min_scale(basis)
        self.register_buffer('basis', basis)

        self.num_funcs = self.basis.size(0)

        self.weight = nn.Parameter(torch.Tensor(
            out_channels, in_channels, scale_size, self.num_funcs))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        # get kernel
        basis = self.basis.view(self.num_funcs, -1)
        kernel = self.weight @ basis
        kernel = kernel.view(self.out_channels, self.in_channels, self.scale_size,
                             self.num_scales, self.kernel_size, self.kernel_size)

        # expand kernel
        kernel = kernel.permute(3, 0, 1, 2, 4, 5).contiguous()
        kernel = kernel.view(-1, self.in_channels, self.scale_size,
                             self.kernel_size, self.kernel_size)

        # calculate padding
        if self.scale_size != 1:
            value = x.mean()
            x = F.pad(x, [0, 0, 0, 0, 0, self.scale_size - 1])

        output = 0.0
        for i in range(self.scale_size):
            x_ = x[:, :, i:i + self.num_scales]
            # expand X
            B, C, S, H, W = x_.shape
            x_ = x_.permute(0, 2, 1, 3, 4).contiguous()
            x_ = x_.view(B, -1, H, W)
            if self.padding > 0:
                x_ = F.pad(x_, 4 * [self.padding], mode=self.padding_mode)
            output += F.conv2d(x_, kernel[:, :, i], groups=S, stride=self.stride)

        # squeeze output
        B, C_, H_, W_ = output.shape
        output = output.view(B, S, -1, H_, W_)
        output = output.permute(0, 2, 1, 3, 4).contiguous()
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1, 1)
        return output

    def extra_repr(self):
        s = '{in_channels}->{out_channels} | scales={scales} | size={kernel_size}'
        return s.format(**self.__dict__)


class SESConv_H_H_1x1(nn.Module):

    def __init__(self, in_channels, out_channels, scale_size=1, stride=1, num_scales=1, bias=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_size = scale_size
        self.stride = (1, stride, stride)
        if scale_size > 1:
            # workaround for being compatible with the old-style weight init
            self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, scale_size, 1, 1))
        else:
            self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, 1, 1))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def forward(self, x):
        weight = self.weight
        if len(weight.shape) == 4:
            weight = weight[:, :, None]
        pad = self.scale_size - 1
        return F.conv3d(x, weight, padding=[pad, 0, 0], stride=self.stride)[:, :, pad:]

    def extra_repr(self):
        s = '{in_channels}->{out_channels} | scale_size={scale_size}'
        return s.format(**self.__dict__)


def ses_max_projection(x):
    return x.max(2)[0]
