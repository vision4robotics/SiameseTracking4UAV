import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


eps = 1e-5

# -------------
# Single Layer
# -------------
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv3x3NP(in_planes, out_planes, stride=1):
    """3x3 convolution without padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, bias=False)



def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def down(in_planes, out_planes):
    """downsampling at the output layer"""
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=1),
            nn.BatchNorm2d(out_planes))

def down_spatial(in_planes, out_planes):
    """downsampling 21*21 to 5*5 (21-5)//4+1=5"""
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=4),
            nn.BatchNorm2d(out_planes))


# -------------------------------
# Several Kinds Bottleneck Blocks
# -------------------------------

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        padding = 2 - stride
        if downsample is not None and dilation > 1:
            dilation = dilation // 2
            padding = dilation

        assert stride == 1 or dilation == 1, \
            "stride and dilation must have one equals to zero at least"

        if dilation > 1:
            padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=padding, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        out = self.relu(out)

        return out

class Bottleneck_BIG_CI(nn.Module):
    """
    Bottleneck with center crop layer, double channels in 3*3 conv layer in shortcut branch
    """
    expansion = 4

    def __init__(self, inplanes, planes, last_relu, stride=1, downsample=None, dilation=1):
        super(Bottleneck_BIG_CI, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        padding = 1
        if abs(dilation - 2) < eps: padding = 2
        if abs(dilation - 3) < eps: padding = 3

        self.conv2 = nn.Conv2d(planes, planes*2, kernel_size=3, stride=stride, padding=padding, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes*2)
        self.conv3 = nn.Conv2d(planes*2, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.last_relu = last_relu

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        if self.last_relu:  # feature out no relu
            out = self.relu(out)

        out = self.center_crop(out)  # in-layer crop

        return out

    def center_crop(self, x):
        """
        center crop layer. crop [1:-2] to eliminate padding influence.
        Crop 1 element around the tensor
        input x can be a Variable or Tensor
        """
        return x[:, :, 1:-1, 1:-1].contiguous()

# ---------------------
# Modified ResNet
# ---------------------
class ResNet_plus2(nn.Module):
    def __init__(self, block, layers, used_layers, online=False):
        self.inplanes = 64
        super(ResNet_plus2, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=0,  # 3
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)

        self.feature_size = 128 * block.expansion
        self.used_layers = used_layers
        self.layer3_use = True if 3 in used_layers else False
        self.layer4_use = True if 4 in used_layers else False

        if self.layer3_use:
            if online:
                self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2, update=True)
                self.layeronline = self._make_layer(block, 256, layers[2], stride=2)
            else:
                self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)

            self.feature_size = (256 + 128) * block.expansion
        else:
            self.layer3 = lambda x: x  # identity

        if self.layer4_use:
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)  # 7x7, 3x3
            self.feature_size = 512 * block.expansion
        else:
            self.layer4 = lambda x: x  # identity

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, update=False):
        downsample = None
        dd = dilation
        if stride != 1 or self.inplanes != planes * block.expansion:
            if stride == 1 and dilation == 1:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                )
            else:
                if dilation > 1:
                    dd = dilation // 2
                    padding = dd
                else:
                    dd = 1
                    padding = 0
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=3, stride=stride, bias=False,
                              padding=padding, dilation=dd),
                    nn.BatchNorm2d(planes * block.expansion),
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride,
                            downsample=downsample, dilation=dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        if update: self.inplanes = int(self.inplanes / 2)  # for online
        return nn.Sequential(*layers)

    def forward(self, x, online=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x_ = self.relu(x)
        x = self.maxpool(x_)

        p1 = self.layer1(x)
        p2 = self.layer2(p1)

        if online: return self.layeronline(p2)
        p3 = self.layer3(p2)

        return [x_, p1, p2], p3


class ResNet(nn.Module):
    """
    ResNet with 22 layer utilized in CVPR2019 paper.
    Usage: ResNet(Bottleneck_CI, [3, 4], [True, False], [False, True], 64, [64, 128])
    """

    def __init__(self, block, layers, last_relus, s2p_flags, firstchannels=64, channels=[64, 128], dilation=1):
        self.inplanes = firstchannels
        self.stage_len = len(layers)
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, firstchannels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(firstchannels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # stage2
        if s2p_flags[0]:
            self.layer1 = self._make_layer(block, channels[0], layers[0], stride2pool=True, last_relu=last_relus[0])
        else:
            self.layer1 = self._make_layer(block, channels[0], layers[0], last_relu=last_relus[0])

        # stage3
        if s2p_flags[1]:
            self.layer2 = self._make_layer(block, channels[1], layers[1], stride2pool=True, last_relu=last_relus[1], dilation=dilation)
        else:
            self.layer2 = self._make_layer(block, channels[1], layers[1], last_relu=last_relus[1], dilation=dilation)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, last_relu, stride=1, stride2pool=False, dilation=1):
        """
        :param block:
        :param planes:
        :param blocks:
        :param stride:
        :param stride2pool: translate (3,2) conv to (3, 1)conv + (2, 2)pool
        :return:
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, last_relu=True, stride=stride, downsample=downsample, dilation=dilation))
        if stride2pool:
            layers.append(self.maxpool)
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            if i == blocks - 1:
                layers.append(block(self.inplanes, planes, last_relu=last_relu, dilation=dilation))
            else:
                layers.append(block(self.inplanes, planes, last_relu=True, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)     # stride = 2
        x = self.bn1(x)
        x = self.relu(x)
        x = self.center_crop7(x)
        x = self.maxpool(x)   # stride = 4

        x = self.layer1(x)
        x = self.layer2(x)    # stride = 8

        return x

    def center_crop7(self, x):
        """
        Center crop layer for stage1 of resnet. (7*7)
        input x can be a Variable or Tensor
        """

        return x[:, :, 2:-2, 2:-2].contiguous()

# ----------------------
# Modules used by ATOM
# ----------------------
class FeatureBase:
    """Base feature class.
    args:
        fparams: Feature specific parameters.
        pool_stride: Amount of average pooling to apply do downsample the feature map.
        output_size: Alternatively, specify the output size of the feature map. Adaptive average pooling will be applied.
        normalize_power: The power exponent for the normalization. None means no normalization (default).
        use_for_color: Use this feature for color images.
        use_for_gray: Use this feature for grayscale images.
    """
    def __init__(self, fparams = None, pool_stride = None, output_size = None, normalize_power = None, use_for_color = True, use_for_gray = True):
        self.fparams = fparams
        self.pool_stride = 1 if pool_stride is None else pool_stride
        self.output_size = output_size
        self.normalize_power = normalize_power
        self.use_for_color = use_for_color
        self.use_for_gray = use_for_gray

    def initialize(self):
        pass

    def free_memory(self):
        pass

    def dim(self):
        raise NotImplementedError

    def stride(self):
        raise NotImplementedError

    def size(self, im_sz):
        if self.output_size is None:
            return im_sz // self.stride()
        if isinstance(im_sz, torch.Tensor):
            return torch.Tensor([self.output_size[0], self.output_size[1]])
        return self.output_size

    def extract(self, im):
        """Performs feature extraction."""
        raise NotImplementedError

    def get_feature(self, im: torch.Tensor):
        """Get the feature. Generally, call this function.
        args:
            im: image patch as a torch.Tensor.
        """

        # Return empty tensor if it should not be used
        is_color = im.shape[1] == 3
        if is_color and not self.use_for_color or not is_color and not self.use_for_gray:
            return torch.Tensor([])

        # Extract feature
        feat = self.extract(im)

        # Pool/downsample
        if self.output_size is not None:
            feat = F.adaptive_avg_pool2d(feat, self.output_size)
        elif self.pool_stride != 1:
            feat = F.avg_pool2d(feat, self.pool_stride, self.pool_stride)

        # Normalize
        if self.normalize_power is not None:
            feat /= (torch.sum(feat.abs().view(feat.shape[0],1,1,-1)**self.normalize_power, dim=3, keepdim=True) /
                     (feat.shape[1]*feat.shape[2]*feat.shape[3]) + 1e-10)**(1/self.normalize_power)

        return feat


# -----------------
# Efficient net
# -----------------
#  --------------------------------------
# Efficient net -- lighter and stronger
# -------------------------------------
class Identity(nn.Module):
    def __init__(self, ):
        super(Identity, self).__init__()

    def forward(self, input):
        return input

class Conv2dStaticSamePadding(nn.Conv2d):
    """ 2D Convolutions like TensorFlow, for a fixed image size"""

    def __init__(self, in_channels, out_channels, kernel_size, image_size=None, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

        # Calculate padding based on image size and save it
        assert image_size is not None
        ih, iw = image_size if type(image_size) == list else [image_size, image_size]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            self.static_padding = nn.ZeroPad2d((pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))
        else:
            self.static_padding = Identity()

    def forward(self, x):
        x = self.static_padding(x)
        x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x


def get_same_padding_conv2d(image_size=None):
    """ Chooses static padding if you have specified an image size, and dynamic padding otherwise.
        Static padding is necessary for ONNX exporting of models. """
    if image_size is None:
        return Conv2dDynamicSamePadding
    else:
        return partial(Conv2dStaticSamePadding, image_size=image_size)


class Conv2dDynamicSamePadding(nn.Conv2d):
    """ 2D Convolutions like TensorFlow, for a dynamic image size """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

def round_filters(filters, global_params):
    """ Calculate and round number of filters based on depth multiplier. """
    multiplier = global_params.width_coefficient
    if not multiplier:
        return filters
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:  # prevent rounding by more than 10%
        new_filters += divisor
    return int(new_filters)

def round_repeats(repeats, global_params):
    """ Round number of filters based on depth multiplier. """
    multiplier = global_params.depth_coefficient
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))

class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)

def drop_connect(inputs, p, training):
    """ Drop connect. """
    if not training: return inputs
    batch_size = inputs.shape[0]
    keep_prob = 1 - p
    random_tensor = keep_prob
    random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype, device=inputs.device)
    binary_tensor = torch.floor(random_tensor)
    output = inputs / keep_prob * binary_tensor
    return output
