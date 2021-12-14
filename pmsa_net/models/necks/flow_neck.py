import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
from mmseg.ops import resize
from ..builder import NECKS
from .psp_head import PPM


class AlignedModule(BaseModule):

    def __init__(self, in_channels, channels, align_corners, kernel_size = 3):
        super(AlignedModule, self).__init__()

        self. align_corners = align_corners
        self.h_conv = nn.Conv2d(in_channels, channels, 1, bias = False)
        self.l_conv = nn.Conv2d(in_channels, channels, 1, bias = False)
        self.flow_make = nn.Conv2d(channels*2, 2, kernel_size=kernel_size, padding=1, bias=False)

    def forward(self, low_feats, high_feats):
        # print(type(low_feats))
        # print(type(high_feats))
        low_feature = self.l_conv(low_feats)
        high_feature = self.h_conv(high_feats)
        high_feature = resize(
            high_feature,
            low_feats.size()[2:],
            mode = "bilinear",
            align_corners = self.align_corners)
        flow = self.flow_make(torch.cat([high_feature, low_feature], dim = 1))
        high_feature = self.flow_warp(high_feats, flow)

        high_feature = low_feats + high_feature

        return high_feature

    def flow_warp(self, input, flow):
        out_h, out_w = flow.size()[2:]
        n, c, h, w = input.size()
        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        h = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        w = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((w.unsqueeze(2), h.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid, align_corners = self.align_corners)
        return output


class FAM_Neck(BaseModule):
    """Semantic Flow Network.
    This neck is the implementation of
    `SFNet <https://arxiv.org/abs/2002.10120>`_.
    Args:
        in_channels (Sequence(int)): Input channels.
        channels (int): Channels after modules, before conv_seg.
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
        out_indices (tuple): Tuple of indices of output.
            Often set to (0,1,2,3,4) to enable aux. heads.
            Default: (0, 1, 2, 3, 4).
        conv_cfg (dict | None): Config of conv layers.
            Default: None
        norm_cfg (dict | None): Config of norm layers.
            Default: None
        act_cfg (dict): Config of activation layers.
            Default: None
    """
    def __init__(self,
                 inchannels,
                 channels,
                 pool_scales=(1, 2, 3, 6),
                 align_corners= False,
                 out_indices = (0, 1, 2, 3),
                 conv_cfg=None,
                 norm_cfg = None,
                 act_cfg = None,
                 **kwargs):
        super(FAM_Neck, self).__init__(**kwargs)
        assert isinstance(in_channels, list)
        assert isinstance(pool_scales, (list, tuple))
        self.inchannels = inchannels
        self.channels = channels
        self.pool_scales = pool_scales
        self.align_corners = align_corners
        self.out_indices = out_indices
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.fp16_enabled = False

        self.ppm = PPM(
            self.pool_scales,
            self.in_channels[-1],
            self.channels,
            conv_cfg = self.conv_cfg,
            norm_cfg = self.norm_cfg,
            act_cfg = self.act_cfg,
            align_corners = self.align_corners)
        self.fpn_in = nn.ModuleList()
        self.fpn_out = nn.ModuleList()
        self.fams = nn.ModuleList()
        if self.fpn_dsn:
            self.dsn = nn.ModuleList()

        for i in range(len(fpn_inchannels) - 1):  # 0, 1, 2
            self.fpn_in.append(
                ConvModule(
                    self.fpn_inchannels[i],
                    self.channels,
                    1,
                    conv_cfg = self.conv_cfg,
                    norm_cfg = self.norm_cfg,
                    act_cfg = self.act_cfg,
                    inplace = False))
            self.fpn_out.append(
                ConvModule(
                    self.channels,
                    self.channels,
                    1,
                    conv_cfg = self.conv_cfg,
                    norm_cfg = self.norm_cfg,
                    act_cfg = self.act_cfg))
            self.fams.append(
                AlignedModule(
                    in_channels = self.channels,
                    channels = self.channels//2,
                    align_corners = self.align_corners))

            self.ppm_bottleneck = ConvModule(
                len(pool_scales) * self.channels + self.in_channels[-1],
                self.channels,
                3,
                conv_cfg = self.conv_cfg,
                norm_cfg = self.norm_cfg,
                act_cfg = self.act_cfg)


    @auto_fp16()
    def forward(self, inputs):
        assert len(inputs) == len(self.inchannels)

        ppmoutputs = [inputs[-1]]
        ppmoutputs.extend(self.ppm(inputs[-1]))
        output = torch.cat(ppmoutputs, dim = 1)
        output = self.ppm_bottleneck(output)
        fpn_features = [output]
        outs = [output]

        for i in reversed(range(len(inputs) - 1)):  # 2, 1, 0
            x = self.fpn_in[i](inputs[i])
            output = self.fams[i](x, output)
            outs.append(output)
            fpn_features.append(self.fpn_out[i](output))

        fpn_features.reverse()  # [p2--p5]
        for i in range(1, len(fpn_features)):
            fpn_features[i] = resize(
                    fpn_features[i],
                    fpn_features[0].size()[2:],
                    mode = 'bilinear',
                    align_corners = self.align_corners)
        fusion_out = torch.cat(fpn_features, dim = 1)
        outs.append(fusion_out)

        outs = [outs[i] for i in self.out_indices]
        return tuple(outs)

