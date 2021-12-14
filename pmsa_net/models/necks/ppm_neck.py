# -*- coding: utf-8 -*-
# python 3.7
# @Author  : caodroid
# @Time    : 2021/8/9 9:58
# @File    : ppm_neck.py
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16
from mmseg.ops import resize
from ..builder import NECKS
from ..decode_heads.psp_head import PPM


@NECKS.register_module()
class PPM_Neck(BaseModule):
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
                 in_channels,
                 channels,
                 pool_scales=(1, 2, 3, 6),
                 out_indices = (0, 1, 2, 3, 4),
                 conv_cfg=None,
                 norm_cfg = None,
                 act_cfg=dict(type='ReLU'),
                 align_corners = False,
                 **kwargs):
        super(PPM_Neck, self).__init__(**kwargs)
        assert isinstance(pool_scales, (list, tuple))
        self.in_channels = in_channels
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
            self.in_channels,
            self.channels,
            conv_cfg = self.conv_cfg,
            norm_cfg = self.norm_cfg,
            act_cfg = self.act_cfg,
            align_corners = self.align_corners)

        self.ppm_bottleneck = ConvModule(
            len(pool_scales) * self.channels + self.in_channels,
            self.in_channels,
            3,
            conv_cfg = self.conv_cfg,
            norm_cfg = self.norm_cfg,
            act_cfg = self.act_cfg)


    @auto_fp16()
    def forward(self, inputs):
        outs = []
        outs.extend(inputs)
        ppm_outs = [inputs[-1]]
        ppm_outs.extend(self.ppm(inputs[-1]))
        ppm_outs = torch.cat(ppm_outs, dim = 1)
        outs.append(self.ppm_bottleneck(ppm_outs))
        outs = [outs[i] for i in self.out_indices]
        return tuple(outs)
