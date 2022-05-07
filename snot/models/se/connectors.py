import torch
import torch.nn as nn
import torch.nn.functional as F

from snot.models.se.sesn.ses_conv import ses_max_projection


class ScaleHead(nn.Module):

    def __init__(self, out_scale=0.1, scales=[1], head='corr'):
        super().__init__()
        self.out_scale = out_scale
        self.scales = scales
        self.head = head
        print('| using scales:', scales)

        if self.head == 'corr':
            self.corr_func = self._fast_xcorr
            print('| using ordinary correlation')
        if self.head == 'greedypw':
            self.scalepooling = self.pw_maxpooling
            self.corr_func = self._fast_corr_scale
            print('| using greedypw correlation')

    def forward(self, z, x):
        z = ses_max_projection(z)
        x = ses_max_projection(x)
        pooled, heatmaps = self.corr_func(z, x)
        return pooled * self.out_scale, heatmaps

    def pw_maxpooling(self, x, scale_dim=1):
        n_scales = x.shape[scale_dim]
        n_c = x.shape[0]
        raveled = x.view(n_c, n_scales, -1)
        zero_batch_max = raveled.max(dim=-1).values[0]

        pooled = x.max(dim=1, keepdim=True).values
        return pooled

    def _fast_corr_scale(self, z, x):
        scale = self.scales
        outsize_h = x.shape[-2] - z.shape[-2] + 1
        outsize_w = x.shape[-1] - z.shape[-1] + 1

        output = torch.zeros(x.shape[0], len(scale), outsize_h, outsize_w, device=x.device)
        for i, each_scale in enumerate(scale):
            x_rescale = self.rescale4d(x, each_scale)
            y_rescale, _ = self._fast_xcorr(z, x_rescale)
            output[:, i, ...] = self.rescale4d(
                y_rescale, outsize_w / y_rescale.shape[-1]).squeeze(1)

        out = self.scalepooling(output)
        return out, output

    def _fast_xcorr(self, z, x):
        nz = z.size(0)
        nx, c, h, w = x.size()
        x = x.view(-1, nz * c, h, w)
        out = F.conv2d(x, z, groups=nz)
        out = out.view(nx, -1, out.size(-2), out.size(-1))
        output_heatmaps = torch.empty_like(out)
        return out, output_heatmaps

    def rescale4d(self, x, scale, mode='bicubic', padding_mode='constant'):
        if mode == 'nearest':
            align_corners = None
        else:
            align_corners = True

        if scale == 1:
            return x

        return F.interpolate(x, scale_factor=scale, mode=mode, align_corners=align_corners)
