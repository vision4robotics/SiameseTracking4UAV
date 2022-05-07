# Parts of this code come from https://github.com/isosnovik/SiamSE
import numpy as np
import torch
import torch.nn.functional as F

from snot.utils.utils_se import get_subwindow_tracking, make_scale_pyramid


class SESiamFCTracker(object):
    def __init__(self, net, num_scales=3, scale_step=1.0375, scale_penalty=0.9745,
                 scale_lr=0.590, response_up=16, w_influence=0.350, exemplar_size=127,
                 instance_size=255, score_size=17, total_stride=8, context_amount=0.5, **kwargs):
        super(SESiamFCTracker, self).__init__()
        self.net = net
        # config parameters
        self.num_scales = num_scales
        self.scale_step = scale_step
        self.scale_penalty = scale_penalty
        self.scale_lr = scale_lr
        self.response_up = response_up
        self.w_influence = w_influence
        self.exemplar_size = exemplar_size
        self.instance_size = instance_size
        self.score_size = score_size
        self.total_stride = total_stride
        self.context_amount = context_amount

        # constant extra parameteres
        window = np.outer(np.hanning(int(self.score_size) * int(self.response_up)),
                          np.hanning(int(self.score_size) * int(self.response_up)))
        self.window = window / window.sum()

        self.scales = self.scale_step ** (range(self.num_scales) - np.ceil(self.num_scales // 2))

        # runnning stats
        self.target_pos = None
        self.target_sz = None
        self.avg_chans = None
        self.im_h = None
        self.im_w = None
        self.s_x = None
        self.min_s_x = None
        self.max_s_x = None

    @torch.no_grad()
    def init(self, im, target_pos, target_sz):
        self.target_pos = target_pos
        self.target_sz = target_sz
        self.avg_chans = np.mean(im, axis=(0, 1))
        self.im_h = im.shape[0]
        self.im_w = im.shape[1]

        # prep
        wc_z = target_sz[0] + self.context_amount * sum(target_sz)
        hc_z = target_sz[1] + self.context_amount * sum(target_sz)
        s_z = round(np.sqrt(wc_z * hc_z))
        scale_z = self.exemplar_size / s_z

        d_search = (self.instance_size - self.exemplar_size) / 2
        pad = d_search / scale_z

        self.s_x = s_z + 2 * pad
        self.min_s_x = 0.2 * self.s_x
        self.max_s_x = 5 * self.s_x

        z_crop = get_subwindow_tracking(im, target_pos, self.exemplar_size, s_z, self.avg_chans)
        self.net.template(z_crop.unsqueeze(0).cuda())

    @torch.no_grad()
    def track(self, im):
        scaled_instance = self.s_x * self.scales
        scaled_target = [[self.target_sz[0] * self.scales], [self.target_sz[1] * self.scales]]

        x_crops = make_scale_pyramid(im, self.target_pos, scaled_instance,
                                     self.instance_size, self.avg_chans).cuda()
        response_map, _ = self.net.track(x_crops)
        up_size = self.response_up * response_map.shape[-1]
        response_map_up = F.interpolate(response_map, size=(up_size, up_size), mode='bicubic')
        response_map_up = response_map_up.squeeze(1).detach().cpu().data.numpy().transpose(1, 2, 0)

        temp_max = np.max(response_map_up, axis=(0, 1))
        s_penaltys = np.array([self.scale_penalty ** (abs(i - self.num_scales // 2))
                               for i in range(self.num_scales)])
        temp_max *= s_penaltys
        best_scale = np.argmax(temp_max)
        response_map = response_map_up[:, :, best_scale]
        response_map = response_map - response_map.min()
        response_map = response_map / response_map.sum()

        # apply windowing
        response_map = (1 - self.w_influence) * response_map + self.w_influence * self.window
        r_max, c_max = np.unravel_index(response_map.argmax(), response_map.shape)
        p_corr = [c_max, r_max]

        disp_instance_final = p_corr - np.ceil(self.score_size * self.response_up / 2)
        disp_instance_input = disp_instance_final * self.total_stride / self.response_up
        disp_instance_frame = disp_instance_input * self.s_x / self.instance_size
        target_pos = self.target_pos + disp_instance_frame
        # scale damping and saturation
        self.s_x = max(self.min_s_x, min(self.max_s_x, (1 - self.scale_lr) *
                                         self.s_x + self.scale_lr * scaled_instance[best_scale]))

        target_sz = [(1 - self.scale_lr) * self.target_sz[0] + self.scale_lr * scaled_target[0][0][best_scale],
                     (1 - self.scale_lr) * self.target_sz[1] + self.scale_lr * scaled_target[1][0][best_scale]]

        target_pos[0] = max(0, min(self.im_w, target_pos[0]))
        target_pos[1] = max(0, min(self.im_h, target_pos[1]))
        target_sz[0] = max(10, min(self.im_w, target_sz[0]))
        target_sz[1] = max(10, min(self.im_h, target_sz[1]))

        self.target_pos = target_pos
        self.target_sz = target_sz
        return target_pos, target_sz

    def __repr__(self):
        s = self.__class__.__name__ + ':\n'
        s += '  num_scales={num_scales}\n'
        s += '  scale_step={scale_step}\n'
        s += '  scale_lr={scale_lr}\n'
        s += '  response_up={response_up}\n'
        s += '  w_influence={w_influence}\n'
        s += '  exemplar_size={exemplar_size}\n'
        s += '  instance_size={instance_size}\n'
        s += '  score_size={score_size}\n'
        s += '  total_stride={total_stride}\n'
        s += '  context_amount={context_amount}\n'
        s += '  scales={scales}\n'
        return s.format(**self.__dict__)
