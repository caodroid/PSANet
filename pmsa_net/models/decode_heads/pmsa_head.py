from abc import ABC

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmseg.core import add_prefix
from mmseg.ops import resize
from ..builder import HEADS
from .aspp_head import ASPPHead, ASPPModule

from ..utils import SelfAttentionBlock
from mmcv.cnn import ConvModule, Scale
from torch.nn import functional as F
from .decode_head import BaseDecodeHead
from .sep_aspp_head import DepthwiseSeparableASPPHead


class DepthwiseSeparableASPPModule(ASPPModule):
    """Atrous Spatial Pyramid Pooling (ASPP) Module with depthwise separable
    conv.

        Args:
        dilations (tuple[int]): Dilation rate of each layer.  (1, 12, 24, 36) 这里可能需要串联
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict): Config of activation layers.
    """

    def __init__(self, **kwargs):
        super(DepthwiseSeparableASPPModule, self).__init__(**kwargs)
        for i, dilation in enumerate(self.dilations):
            if dilation > 1:
                self[i] = DepthwiseSeparableConvModule(
                    self.in_channels,
                    self.channels,
                    3,
                    dilation=dilation,
                    padding=0 if dilation == 1 else dilation,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg)


class PMSA(SelfAttentionBlock):

    def __init__(self, in_channels, channels, conv_cfg, norm_cfg, act_cfg):
        super(PMSA, self).__init__(
            key_in_channels = in_channels + channels,
            query_in_channels = in_channels,
            channels = channels,
            out_channels = in_channels,
            share_key_query = False,
            query_downsample = None,
            key_downsample = None,
            key_query_num_convs = 1,
            key_query_norm = False,
            value_out_num_convs = 1,
            value_out_norm = False,
            matmul_norm = False,
            with_out = False,
            conv_cfg = conv_cfg,
            norm_cfg = norm_cfg,
            act_cfg = act_cfg)

    def forward(self, query_feats, key_feats):
        """Forward function."""
        context = super(PMSA, self).forward(query_feats, key_feats)
        context = torch.cat([context, key_feats], dim = 1)
        return context

@HEADS.register_module()
class PMSAHead(ASPPHead):
    """
    PMSA-Net: Pixel-specific Multiscale Attention Network for image semantic segmentation
    Args:
        c1_in_channels (int): The input channels of c1 decoder. If is 0,
            the no decoder will be used.
        c1_channels (int): The intermediate channels of c1 decoder. # output of layer 1
    """

    def __init__(self, dilations, c1_in_channels, c1_channels, **kwargs):
        super(PMSAHead, self).__init__(**kwargs)
        assert c1_in_channels >= 0
        self.dilations = dilations
        self.aspp_modules = DepthwiseSeparableASPPModule(
            in_channels = self.in_channels,
            channels = self.channels,
            dilations = self.dilations,
            conv_cfg = self.conv_cfg,
            norm_cfg = self.norm_cfg,
            act_cfg = self.act_cfg)

        self.attention = PMSA(
            in_channels = self.in_channels,
            channels = self.channels,
            conv_cfg = self.conv_cfg,
            norm_cfg = self.norm_cfg,
            act_cfg = self.act_cfg)
        self.bottleneck = ConvModule(
            self.in_channels * 2  + self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        if c1_in_channels > 0:
            self.c1_bottleneck = ConvModule(
                c1_in_channels,
                c1_channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        else:
            self.c1_bottleneck = None

        self.sep_bottleneck = nn.Sequential(
            DepthwiseSeparableConvModule(
                self.channels + c1_channels,
                self.channels,
                3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
            DepthwiseSeparableConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))

    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)
        aspp_outs = [
            resize(
                self.image_pool(x),
                size=x.size()[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        ]
        aspp_outs.extend(self.aspp_modules(x))
        aspp_outs = torch.cat(aspp_outs, dim=1)
        aspp_outs = self.attention(x, aspp_outs)
        output = self.bottleneck(aspp_outs)
        if self.c1_bottleneck is not None:
            c1_output = self.c1_bottleneck(inputs[0])
            output = resize(
                input=output,
                size=c1_output.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            output = torch.cat([output, c1_output], dim=1)
        output = self.sep_bottleneck(output)
        output = self.cls_seg(output)
        return output

