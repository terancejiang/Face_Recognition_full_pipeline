"""
Based on: https://arxiv.org/abs/2307.09283
"""

import torch.nn as nn
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, Sequential, Module


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


from timm.models.layers import SqueezeExcite

import torch


class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps) ** 0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation,
                            groups=self.c.groups,
                            device=c.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class Residual(torch.nn.Module):
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)

    @torch.no_grad()
    def fuse(self):
        if isinstance(self.m, Conv2d_BN):
            m = self.m.fuse()
            assert (m.groups == m.in_channels)
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = torch.nn.functional.pad(identity, [1, 1, 1, 1])
            m.weight += identity.to(m.weight.device)
            return m
        elif isinstance(self.m, torch.nn.Conv2d):
            m = self.m
            assert (m.groups != m.in_channels)
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = torch.nn.functional.pad(identity, [1, 1, 1, 1])
            m.weight += identity.to(m.weight.device)
            return m
        else:
            return self


class RepVGGDW(torch.nn.Module):
    def __init__(self, ed) -> None:
        super().__init__()
        self.conv = Conv2d_BN(ed, ed, 3, 1, 1, groups=ed)
        self.conv1 = torch.nn.Conv2d(ed, ed, 1, 1, 0, groups=ed)
        self.dim = ed
        self.bn = torch.nn.BatchNorm2d(ed)

    def forward(self, x):
        return self.bn((self.conv(x) + self.conv1(x)) + x)

    @torch.no_grad()
    def fuse(self):
        conv = self.conv.fuse()
        conv1 = self.conv1

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = torch.nn.functional.pad(conv1_w, [1, 1, 1, 1])

        identity = torch.nn.functional.pad(torch.ones(conv1_w.shape[0], conv1_w.shape[1], 1, 1, device=conv1_w.device),
                                           [1, 1, 1, 1])

        final_conv_w = conv_w + conv1_w + identity
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        bn = self.bn
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = conv.weight * w[:, None, None, None]
        b = bn.bias + (conv.bias - bn.running_mean) * bn.weight / \
            (bn.running_var + bn.eps) ** 0.5
        conv.weight.data.copy_(w)
        conv.bias.data.copy_(b)
        return conv


class RepViTBlock(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
        super(RepViTBlock, self).__init__()
        assert stride in [1, 2]
        # input_channel, exp_size, output_channel, k, s, use_se, use_hs
        # k, t, c, SE, HS, s
        # [3, 2, 48, 1, 0, 1],

        self.identity = stride == 1 and inp == oup
        # assert (hidden_dim == 2 * inp)

        if stride == 2:
            self.token_mixer = nn.Sequential(
                Conv2d_BN(inp, inp, kernel_size, stride, (kernel_size - 1) // 2, groups=inp),
                SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
                Conv2d_BN(inp, oup, ks=1, stride=1, pad=0)
            )
            self.channel_mixer = Residual(nn.Sequential(
                # pw
                Conv2d_BN(oup, 2 * oup, 1, 1, 0),
                nn.GELU() if use_hs else nn.GELU(),
                # pw-linear
                Conv2d_BN(2 * oup, oup, 1, 1, 0, bn_weight_init=0),
            ))
        else:
            assert (self.identity)
            self.token_mixer = nn.Sequential(
                RepVGGDW(inp),
                SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
            )
            self.channel_mixer = Residual(nn.Sequential(
                # pw
                Conv2d_BN(inp, hidden_dim, 1, 1, 0),
                nn.GELU() if use_hs else nn.GELU(),
                # pw-linear
                Conv2d_BN(hidden_dim, oup, 1, 1, 0, bn_weight_init=0),
            ))

    def forward(self, x):
        return self.channel_mixer(self.token_mixer(x))


from timm.models.vision_transformer import trunc_normal_


class BN_Linear(torch.nn.Sequential):
    def __init__(self, a, b, bias=True, std=0.02):
        super().__init__()
        self.add_module('bn', torch.nn.BatchNorm1d(a))
        self.add_module('l', torch.nn.Linear(a, b, bias=bias))
        trunc_normal_(self.l.weight, std=std)
        if bias:
            torch.nn.init.constant_(self.l.bias, 0)

    @torch.no_grad()
    def fuse(self):
        bn, l = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        b = bn.bias - self.bn.running_mean * \
            self.bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = l.weight * w[None, :]
        if l.bias is None:
            b = b @ self.l.weight.T
        else:
            b = (l.weight @ b[:, None]).view(-1) + self.l.bias
        m = torch.nn.Linear(w.size(1), w.size(0), device=l.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class LinearBlock(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(LinearBlock, self).__init__()
        self.layers = nn.Sequential(
            Conv2d(in_c, out_c, kernel, stride, padding, groups=groups, bias=False),
            BatchNorm2d(num_features=out_c)
        )

    def forward(self, x):
        return self.layers(x)


class Flatten(Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class GDC(Module):
    def __init__(self, embedding_size, input_size=512):
        super(GDC, self).__init__()
        self.layers = nn.Sequential(
            LinearBlock(input_size, input_size, groups=input_size, kernel=(7, 7), stride=(1, 1), padding=(0, 0)),
            Flatten(),
            Linear(input_size, embedding_size, bias=False),
            BatchNorm1d(embedding_size))

    def forward(self, x):
        return self.layers(x)


class ConvBlock(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(ConvBlock, self).__init__()
        self.layers = nn.Sequential(
            Conv2d(in_c, out_c, kernel, groups=groups, stride=stride, padding=padding, bias=False),
            BatchNorm2d(num_features=out_c),
            PReLU(num_parameters=out_c)
        )

    def forward(self, x):
        return self.layers(x)


class RepViT(nn.Module):
    def __init__(self, cfgs, num_classes=1000, distillation=False):
        super(RepViT, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs

        # building first layer
        input_channel = self.cfgs[0][2]
        patch_embed = torch.nn.Sequential(Conv2d_BN(3, input_channel, 2, 2), torch.nn.GELU())
        # a, b, ks = 1, stride = 1, pad = 0, dilation = 1, groups = 1, bn_weight_init = 1
        layers = [patch_embed]
        # building inverted residual blocks
        block = RepViTBlock
        for k, t, c, use_se, use_hs, s in self.cfgs:
            output_channel = _make_divisible(c, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            layers.append(block(input_channel, exp_size, output_channel, k, s, use_se, use_hs))
            input_channel = output_channel
        self.features = nn.ModuleList(layers)
        # building last several layers
        self.conv_sep = ConvBlock(input_channel, 512, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        self.gdc = GDC(512)

    def forward(self, x):
        # x = self.features(x)
        for f in self.features:
            x = f(x)
        # x = torch.nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
        x = self.conv_sep(x)

        x = self.gdc(x)
        return x


from timm.models import register_model


@register_model
def repvit_m0_9(pretrained=False, distillation=False):
    """
    Constructs a MobileNetV3-Large model
    """
    cfgs = [
        # t hidden layer
        # k, t, c, SE, HS, s
        [3, 2, 48, 0, 0, 1],

        [3, 2, 48, 0, 0, 1],

        [3, 2, 96, 0, 0, 2],

        [3, 2, 192, 0, 0, 2],
        [3, 2, 192, 0, 0, 1],
        [3, 2, 192, 0, 0, 1],
        [3, 2, 192, 0, 0, 1],
        [3, 2, 192, 0, 0, 1],
        [3, 2, 192, 0, 0, 1],
        # [3, 2, 192, 0, 0, 1],

        [3, 2, 192, 0, 0, 2],

        # [3, 2, 384, 0, 0, 1],

        # [3, 2, 48, 1, 0, 1],
        # [3, 2, 48, 0, 0, 1],
        # [3, 2, 48, 0, 0, 1],
        # [3, 2, 96, 0, 0, 2],
        # [3, 2, 96, 1, 0, 1],
        # [3, 2, 96, 0, 0, 1],
        # [3, 2, 96, 0, 0, 1],
        # [3, 2, 192, 0, 1, 2],
        # [3, 2, 192, 1, 1, 1],
        # [3, 2, 192, 0, 1, 1],
        # [3, 2, 192, 1, 1, 1],
        # [3, 2, 192, 0, 1, 1],
        # [3, 2, 192, 1, 1, 1],
        # [3, 2, 192, 0, 1, 1],
        # [3, 2, 192, 1, 1, 1],
        # [3, 2, 192, 0, 1, 1],
        # [3, 2, 192, 1, 1, 1],
        # [3, 2, 192, 0, 1, 1],
        # [3, 2, 192, 1, 1, 1],
        # [3, 2, 192, 0, 1, 1],
        # [3, 2, 192, 1, 1, 1],
        # [3, 2, 192, 0, 1, 1],
        # [3, 2, 192, 0, 1, 1],
        # [3, 2, 384, 0, 1, 2],
        # [3, 2, 384, 1, 1, 1],
        # [3, 2, 384, 0, 1, 1]
    ]
    return RepViT(cfgs, distillation=distillation)


if __name__ == '__main__':
    import torchinfo
    import torchsummary

    net = repvit_m0_9()
    # You need to define input size to calcualte parameters
    torchinfo.summary(net, input_size=(3, 112, 112), batch_dim=0, verbose=1)
    # torchsummary.summary(net, input_size=(3, 112, 112))
