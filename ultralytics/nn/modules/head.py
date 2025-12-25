# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Model head modules
"""

import math

import torch
import torch.nn as nn
from torch.nn.init import constant_, xavier_uniform_

from ultralytics.yolo.utils.tal import dist2bbox, make_anchors

from .block import DFL, Proto
from .conv import Conv, LightConv
from .transformer import MLP, DeformableTransformerDecoder, DeformableTransformerDecoderLayer
from .utils import bias_init_with_prob, linear_init_

__all__ = 'Detect', 'Segment', 'Pose', 'Classify', 'RTDETRDecoder'


class Detect(nn.Module):
    """YOLOv8 Detect head for detection models."""
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    cube = False
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, ch=(), cube=False):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.cube = cube
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        
        if self.cube:
            
            ### 实验二 ###
            self.no = self.no * 3
            self.dfl1 = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()
            self.dfl2 = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()
            self.dfl3 = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()
            # self.dfl4 = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()
            # self.dfl5 = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()
            # self.dfl6 = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()
            ## 3个独立
            # self.cv2_1 = nn.ModuleList(
            #     nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
            # self.cv2_2 = nn.ModuleList(
            #     nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
            # self.cv2_3 = nn.ModuleList(
            #     nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
            # self.cv3_1 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
            # self.cv3_2 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
            # self.cv3_3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
            ## box独立，cls共享
            # self.cv2_1 = nn.ModuleList(
            #     nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
            # self.cv2_2 = nn.ModuleList(
            #     nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
            # self.cv2_3 = nn.ModuleList(
            #     nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
            # self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
            ## cls共享，引入bias。方式1：每个分支三个conv，第一个conv独立，引入bias，第二、三个conv独立
            # self.cv2_1 = nn.ModuleList(Conv(x, c2, 3) for x in ch)
            # self.cv2_2 = nn.ModuleList(Conv(x, c2, 3) for x in ch)
            # self.cv2_3 = nn.ModuleList(Conv(x, c2, 3) for x in ch)
            # self.cv4_1 = nn.ModuleList(
            #     nn.Sequential(Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
            # self.cv4_2 = nn.ModuleList(
            #     nn.Sequential(Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
            # self.cv4_3 = nn.ModuleList(
            #     nn.Sequential(Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
            # self.cv3_3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
            ## box，cls都独立，但都使用lightconv
            # self.cv2_1 = nn.ModuleList(
            #     nn.Sequential(LightConv(x, c2, 3), LightConv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
            # self.cv2_2 = nn.ModuleList(
            #     nn.Sequential(LightConv(x, c2, 3), LightConv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
            # self.cv2_3 = nn.ModuleList(
            #     nn.Sequential(LightConv(x, c2, 3), LightConv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
            # self.cv3_1 = nn.ModuleList(nn.Sequential(LightConv(x, c3, 3), LightConv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
            # self.cv3_2 = nn.ModuleList(nn.Sequential(LightConv(x, c3, 3), LightConv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
            # self.cv3_3 = nn.ModuleList(nn.Sequential(LightConv(x, c3, 3), LightConv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
            ## box，cls都独立，使用lightconv，最后用bias的方式联系起来
            # self.cv2_1 = nn.ModuleList(
            #     nn.Sequential(LightConv(x, c2, 3), LightConv(c2, c2, 3), LightConv(c2, 4 * self.reg_max, 1)) for x in ch)
            # self.cv2_2 = nn.ModuleList(
            #     nn.Sequential(LightConv(x, c2, 3), LightConv(c2, c2, 3), LightConv(c2, 4 * self.reg_max, 1)) for x in ch)
            # self.cv2_3 = nn.ModuleList(
            #     nn.Sequential(LightConv(x, c2, 3), LightConv(c2, c2, 3), LightConv(c2, 4 * self.reg_max, 1)) for x in ch)
            # self.cv4_1 = nn.ModuleList(
            #     nn.Sequential(nn.Conv2d(4 * self.reg_max, 4 * self.reg_max, 1)) for x in ch)
            # self.cv4_2 = nn.ModuleList(
            #     nn.Sequential(nn.Conv2d(4 * self.reg_max, 4 * self.reg_max, 1)) for x in ch)
            # self.cv4_3 = nn.ModuleList(
            #     nn.Sequential(nn.Conv2d(4 * self.reg_max, 4 * self.reg_max, 1)) for x in ch)
            # self.cv3_1 = nn.ModuleList(nn.Sequential(LightConv(x, c3, 3), LightConv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
            # self.cv3_2 = nn.ModuleList(nn.Sequential(LightConv(x, c3, 3), LightConv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
            # self.cv3_3 = nn.ModuleList(nn.Sequential(LightConv(x, c3, 3), LightConv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
            ## lightconv，box前三个独立，最后一个bias；cls前三个独立，最后一个bias
            # self.cv2_1 = nn.ModuleList(
            #     nn.Sequential(LightConv(x, c2, 3), LightConv(c2, c2, 3), LightConv(c2, 4 * self.reg_max, 1)) for x in ch)
            # self.cv2_2 = nn.ModuleList(
            #     nn.Sequential(LightConv(x, c2, 3), LightConv(c2, c2, 3), LightConv(c2, 4 * self.reg_max, 1)) for x in ch)
            # self.cv2_3 = nn.ModuleList(
            #     nn.Sequential(LightConv(x, c2, 3), LightConv(c2, c2, 3), LightConv(c2, 4 * self.reg_max, 1)) for x in ch)
            # self.cv4_1 = nn.ModuleList(
            #     nn.Sequential(nn.Conv2d(4 * self.reg_max, 4 * self.reg_max, 1)) for x in ch)
            # self.cv4_2 = nn.ModuleList(
            #     nn.Sequential(nn.Conv2d(4 * self.reg_max, 4 * self.reg_max, 1)) for x in ch)
            # self.cv4_3 = nn.ModuleList(
            #     nn.Sequential(nn.Conv2d(4 * self.reg_max, 4 * self.reg_max, 1)) for x in ch)
            # self.cv3_1 = nn.ModuleList(nn.Sequential(LightConv(x, c3, 3), LightConv(c3, c3, 3), LightConv(c3, self.nc, 1)) for x in ch)
            # self.cv3_2 = nn.ModuleList(nn.Sequential(LightConv(x, c3, 3), LightConv(c3, c3, 3), LightConv(c3, self.nc, 1)) for x in ch)
            # self.cv3_3 = nn.ModuleList(nn.Sequential(LightConv(x, c3, 3), LightConv(c3, c3, 3), LightConv(c3, self.nc, 1)) for x in ch)
            # self.cv5_1 = nn.ModuleList(
            #     nn.Sequential(nn.Conv2d(self.nc, self.nc, 1)) for x in ch)
            # self.cv5_2 = nn.ModuleList(
            #     nn.Sequential(nn.Conv2d(self.nc, self.nc, 1)) for x in ch)
            # self.cv5_3 = nn.ModuleList(
            #     nn.Sequential(nn.Conv2d(self.nc, self.nc, 1)) for x in ch)
            ## lightconv，box前三个独立，最后一个bias；cls共享，最后独立
            # self.cv2_1 = nn.ModuleList(
            #     nn.Sequential(LightConv(x, c2, 3), LightConv(c2, c2, 3), LightConv(c2, 4 * self.reg_max, 1)) for x in ch)
            # self.cv2_2 = nn.ModuleList(
            #     nn.Sequential(LightConv(x, c2, 3), LightConv(c2, c2, 3), LightConv(c2, 4 * self.reg_max, 1)) for x in ch)
            # self.cv2_3 = nn.ModuleList(
            #     nn.Sequential(LightConv(x, c2, 3), LightConv(c2, c2, 3), LightConv(c2, 4 * self.reg_max, 1)) for x in ch)
            # self.cv4_1 = nn.ModuleList(
            #     nn.Sequential(nn.Conv2d(4 * self.reg_max, 4 * self.reg_max, 1)) for x in ch)
            # self.cv4_2 = nn.ModuleList(
            #     nn.Sequential(nn.Conv2d(4 * self.reg_max, 4 * self.reg_max, 1)) for x in ch)
            # self.cv4_3 = nn.ModuleList(
            #     nn.Sequential(nn.Conv2d(4 * self.reg_max, 4 * self.reg_max, 1)) for x in ch)
            # self.cv3_3 = nn.ModuleList(nn.Sequential(LightConv(x, c3, 3), LightConv(c3, c3, 3)) for x in ch)
            # self.cv5_1 = nn.ModuleList(
            #     nn.Sequential(nn.Conv2d(c3, self.nc, 1)) for x in ch)
            # self.cv5_2 = nn.ModuleList(
            #     nn.Sequential(nn.Conv2d(c3, self.nc, 1)) for x in ch)
            # self.cv5_3 = nn.ModuleList(
            #     nn.Sequential(nn.Conv2d(c3, self.nc, 1)) for x in ch)
            ## light;box共享、bias、独立;cls共享，最后独立
            # self.cv2_3 = nn.ModuleList(
            #     nn.Sequential(LightConv(x, c2, 3), LightConv(c2, c2, 3)) for x in ch)
            # self.cv6_1 = nn.ModuleList(
            #     nn.Sequential(LightConv(c2, 4 * self.reg_max, 1)) for x in ch)
            # self.cv6_2 = nn.ModuleList(
            #     nn.Sequential(LightConv(c2, 4 * self.reg_max, 1)) for x in ch)
            # self.cv6_3 = nn.ModuleList(
            #     nn.Sequential(LightConv(c2, 4 * self.reg_max, 1)) for x in ch)
            # self.cv4_1 = nn.ModuleList(
            #     nn.Sequential(nn.Conv2d(4 * self.reg_max, 4 * self.reg_max, 1)) for x in ch)
            # self.cv4_2 = nn.ModuleList(
            #     nn.Sequential(nn.Conv2d(4 * self.reg_max, 4 * self.reg_max, 1)) for x in ch)
            # self.cv4_3 = nn.ModuleList(
            #     nn.Sequential(nn.Conv2d(4 * self.reg_max, 4 * self.reg_max, 1)) for x in ch)
            # self.cv3_3 = nn.ModuleList(nn.Sequential(LightConv(x, c3, 3), LightConv(c3, c3, 3)) for x in ch)
            # self.cv5_1 = nn.ModuleList(
            #     nn.Sequential(nn.Conv2d(c3, self.nc, 1)) for x in ch)
            # self.cv5_2 = nn.ModuleList(
            #     nn.Sequential(nn.Conv2d(c3, self.nc, 1)) for x in ch)
            # self.cv5_3 = nn.ModuleList(
            #     nn.Sequential(nn.Conv2d(c3, self.nc, 1)) for x in ch)
            
            ## light；cls；box第一个共享
            self.cv2_3 = nn.ModuleList(
                nn.Sequential(LightConv(x, c2, 3)) for x in ch)
            self.cv6_1 = nn.ModuleList(
                nn.Sequential(LightConv(c2, c2, 3), LightConv(c2, 4 * self.reg_max, 1)) for x in ch)
            self.cv6_2 = nn.ModuleList(
                nn.Sequential(LightConv(c2, c2, 3), LightConv(c2, 4 * self.reg_max, 1)) for x in ch)
            self.cv6_3 = nn.ModuleList(
                nn.Sequential(LightConv(c2, c2, 3), LightConv(c2, 4 * self.reg_max, 1)) for x in ch)
            # self.cv6_4 = nn.ModuleList(
            #     nn.Sequential(LightConv(c2, c2, 3), LightConv(c2, 4 * self.reg_max, 1)) for x in ch)
            # self.cv6_5 = nn.ModuleList(
            #     nn.Sequential(LightConv(c2, c2, 3), LightConv(c2, 4 * self.reg_max, 1)) for x in ch)
            # self.cv6_6 = nn.ModuleList(
            #     nn.Sequential(LightConv(c2, c2, 3), LightConv(c2, 4 * self.reg_max, 1)) for x in ch)

            self.cv4_1 = nn.ModuleList(
                nn.Sequential(nn.Conv2d(4 * self.reg_max, 4 * self.reg_max, 1)) for x in ch)
            self.cv4_2 = nn.ModuleList(
                nn.Sequential(nn.Conv2d(4 * self.reg_max, 4 * self.reg_max, 1)) for x in ch)
            self.cv4_3 = nn.ModuleList(
                nn.Sequential(nn.Conv2d(4 * self.reg_max, 4 * self.reg_max, 1)) for x in ch)
            # self.cv4_4 = nn.ModuleList(
            #     nn.Sequential(nn.Conv2d(4 * self.reg_max, 4 * self.reg_max, 1)) for x in ch)
            # self.cv4_5 = nn.ModuleList(
            #     nn.Sequential(nn.Conv2d(4 * self.reg_max, 4 * self.reg_max, 1)) for x in ch)
            # self.cv4_6 = nn.ModuleList(
            #     nn.Sequential(nn.Conv2d(4 * self.reg_max, 4 * self.reg_max, 1)) for x in ch)

            self.cv3_3 = nn.ModuleList(nn.Sequential(LightConv(x, c3, 3), LightConv(c3, c3, 3)) for x in ch)
            self.cv5_1 = nn.ModuleList(
                nn.Sequential(nn.Conv2d(c3, self.nc, 1)) for x in ch)
            self.cv5_2 = nn.ModuleList(
                nn.Sequential(nn.Conv2d(c3, self.nc, 1)) for x in ch)
            self.cv5_3 = nn.ModuleList(
                nn.Sequential(nn.Conv2d(c3, self.nc, 1)) for x in ch)
            # self.cv5_4 = nn.ModuleList(
            #     nn.Sequential(nn.Conv2d(c3, self.nc, 1)) for x in ch)
            # self.cv5_5 = nn.ModuleList(
            #     nn.Sequential(nn.Conv2d(c3, self.nc, 1)) for x in ch)
            # self.cv5_6 = nn.ModuleList(
            #     nn.Sequential(nn.Conv2d(c3, self.nc, 1)) for x in ch)
            ### ###
            # self.cv2 = nn.ModuleList(
            #     nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
            # self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
        else:
            self.cv2 = nn.ModuleList(
                nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
            self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
            self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        shape = x[0].shape  # BCHW
        for i in range(self.nl):
            if self.cube:
            
                ### 实验二 ###
                ## 3个独立 / box，cls都独立，但都使用lightconv
                # x[i] = torch.cat((self.cv2_1[i](x[i]), self.cv3_1[i](x[i]), self.cv2_2[i](x[i]), self.cv3_2[i](x[i]), self.cv2_3[i](x[i]), self.cv3_3[i](x[i])), 1)
                ## box独立，cls共享
                # x_cls = self.cv3[i](x[i])
                # x[i] = torch.cat((self.cv2_1[i](x[i]), x_cls, self.cv2_2[i](x[i]), x_cls, self.cv2_3[i](x[i]), x_cls), 1)
                ## cls共享，引入bias。方式1
                # x_cls = self.cv3_3[i](x[i])
                # x_box = self.cv2_3[i](x[i])
                # x[i] = torch.cat((self.cv4_1[i](self.cv2_1[i](x[i])+x_box), x_cls, self.cv4_2[i](self.cv2_2[i](x[i])+x_box), x_cls, self.cv4_3[i](x_box), x_cls), 1)
                ## box，cls都独立，使用lightconv，box最后用bias的方式联系起来
                # x_box = self.cv2_3[i](x[i])
                # x[i] = torch.cat((self.cv4_1[i](self.cv2_1[i](x[i])+x_box), self.cv3_1[i](x[i]), self.cv4_2[i](self.cv2_2[i](x[i])+x_box), self.cv3_2[i](x[i]), self.cv4_3[i](x_box), self.cv3_3[i](x[i])), 1)
                ## lightconv，box前三个独立，最后一个bias；cls前三个独立，最后一个bias
                # x_box = self.cv2_3[i](x[i])
                # x_cls = self.cv3_3[i](x[i])
                # x[i] = torch.cat((self.cv4_1[i](self.cv2_1[i](x[i])+x_box), self.cv5_1[i](self.cv3_1[i](x[i])+x_cls), self.cv4_2[i](self.cv2_2[i](x[i])+x_box), self.cv5_2[i](self.cv3_2[i](x[i])+x_cls), self.cv4_3[i](x_box), self.cv5_3[i](x_cls)), 1)
                ## lightconv，box前三个独立，最后一个bias；cls共享，最后独立
                # x_box = self.cv2_3[i](x[i])
                # x_cls = self.cv3_3[i](x[i])
                # x[i] = torch.cat((self.cv4_1[i](self.cv2_1[i](x[i])+x_box), self.cv5_1[i](x_cls), self.cv4_2[i](self.cv2_2[i](x[i])+x_box), self.cv5_2[i](x_cls), self.cv4_3[i](x_box), self.cv5_3[i](x_cls)), 1)
                
                ## light;box共享、bias、独立;cls共享，最后独立 / light；cls；box第一个共享
                x_box = self.cv2_3[i](x[i])
                x_bias = self.cv6_3[i](x_box)
                x_cls = self.cv3_3[i](x[i])
                x[i] = torch.cat((self.cv4_1[i](self.cv6_1[i](x_box)+x_bias), self.cv5_1[i](x_cls), 
                                  self.cv4_2[i](self.cv6_2[i](x_box)+x_bias), self.cv5_2[i](x_cls),
                                #   self.cv4_3[i](self.cv6_3[i](x_box)+x_bias), self.cv5_3[i](x_cls), 
                                #   self.cv4_4[i](self.cv6_4[i](x_box)+x_bias), self.cv5_4[i](x_cls), 
                                #   self.cv4_5[i](self.cv6_5[i](x_box)+x_bias), self.cv5_5[i](x_cls), 
                                  self.cv4_3[i](x_bias), self.cv5_3[i](x_cls)), 1)
                ### ###
                # x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i]), self.cv2[i](x[i]), self.cv3[i](x[i]), self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
                
            else:
                x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:
            return x
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.cube:
            
            ### 实验二 ###
            if self.export and self.format in ('saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs'):  # avoid TF FlexSplitV ops
                box_t1 = x_cat[:, 0 : self.reg_max * 4]
                cls_t1 = x_cat[:, self.reg_max * 4 : self.reg_max * 4 + 1]
                box_t2 = x_cat[:, self.reg_max * 4 + 1 : self.reg_max * 8 + 1]
                cls_t2 = x_cat[:, self.reg_max * 8 + 1 : self.reg_max * 8 + 2]
                box_t3 = x_cat[:, self.reg_max * 8 + 2 : self.reg_max * 12 + 2]
                cls_t3 = x_cat[:, self.reg_max * 12 + 2 : self.reg_max * 12 + 3]
                # box_t4 = x_cat[:, self.reg_max * 12 + 3 : self.reg_max * 16 + 3]
                # cls_t4 = x_cat[:, self.reg_max * 16 + 3 : self.reg_max * 16 + 4]
                # box_t5 = x_cat[:, self.reg_max * 16 + 4 : self.reg_max * 20 + 4]
                # cls_t5 = x_cat[:, self.reg_max * 20 + 4 : self.reg_max * 20 + 5]
                # box_t6 = x_cat[:, self.reg_max * 20 + 5 : self.reg_max * 24 + 5]
                # cls_t6 = x_cat[:, self.reg_max * 24 + 5 : ]
            else:
                # box_t1, cls_t1, box_t2, cls_t2, box_t3, cls_t3, box_t4, cls_t4, box_t5, cls_t5, box_t6, cls_t6 = x_cat.split((self.reg_max * 4, self.nc, self.reg_max * 4, self.nc, self.reg_max * 4, self.nc, self.reg_max * 4, self.nc, self.reg_max * 4, self.nc, self.reg_max * 4, self.nc), 1)
                box_t1, cls_t1, box_t2, cls_t2, box_t3, cls_t3 = x_cat.split((self.reg_max * 4, self.nc, self.reg_max * 4, self.nc, self.reg_max * 4, self.nc), 1)
            dbox1 = dist2bbox(self.dfl1(box_t1), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
            dbox2 = dist2bbox(self.dfl2(box_t2), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
            dbox3 = dist2bbox(self.dfl3(box_t3), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
            # dbox4 = dist2bbox(self.dfl4(box_t4), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
            # dbox5 = dist2bbox(self.dfl5(box_t5), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
            # dbox6 = dist2bbox(self.dfl6(box_t6), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
            # y = torch.cat((dbox1, cls_t1.sigmoid(), dbox2, cls_t2.sigmoid(), dbox3, cls_t3.sigmoid(), dbox4, cls_t4.sigmoid(), dbox5, cls_t5.sigmoid(), dbox6, cls_t6.sigmoid()), 1)
            y = torch.cat((dbox1, cls_t1.sigmoid(), dbox2, cls_t2.sigmoid(), dbox3, cls_t3.sigmoid()), 1)
            ### ###
            
        else:
            if self.export and self.format in ('saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs'):  # avoid TF FlexSplitV ops
                box = x_cat[:, :self.reg_max * 4]
                cls = x_cat[:, self.reg_max * 4:]
            else:
                box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
            dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
            y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        if self.cube:
            
            ### 实验二 ###
            ## 3个独立
            # for a1, a2, a3, b1, b2, b3, s in zip(m.cv2_1, m.cv2_2, m.cv2_3, m.cv3_1, m.cv3_2, m.cv3_3, m.stride):  # from
            #     a1[-1].bias.data[:] = 1.0  # box
            #     a2[-1].bias.data[:] = 1.0  # box
            #     a3[-1].bias.data[:] = 1.0  # box
            #     b1[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)
            #     b2[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)
            #     b3[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)
            ## box独立，cls共享
            # for a1, a2, a3, b, s in zip(m.cv2_1, m.cv2_2, m.cv2_3, m.cv3, m.stride):  # from
            #     a1[-1].bias.data[:] = 1.0  # box
            #     a2[-1].bias.data[:] = 1.0  # box
            #     a3[-1].bias.data[:] = 1.0  # box
            #     b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)
            ## cls共享，引入bias。方式1
            # for a1, a2, a3, b, s in zip(m.cv4_1, m.cv4_2, m.cv4_3, m.cv3_3, m.stride):  # from
            #     a1[-1].bias.data[:] = 1.0  # box
            #     a2[-1].bias.data[:] = 1.0  # box
            #     a3[-1].bias.data[:] = 1.0  # box
            #     b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)
            ## box，cls都独立，使用lightconv，box最后用bias的方式联系起来
            # for a1, a2, a3, b1, b2, b3, s in zip(m.cv4_1, m.cv4_2, m.cv4_3, m.cv3_1, m.cv3_2, m.cv3_3, m.stride):  # from
            #     a1[-1].bias.data[:] = 1.0  # box
            #     a2[-1].bias.data[:] = 1.0  # box
            #     a3[-1].bias.data[:] = 1.0  # box
            #     b1[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)
            #     b2[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)
            #     b3[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)
            
            ## lightconv，box前三个独立，最后一个bias；cls前三个独立，最后一个bias / lightconv，box前三个独立，最后一个bias；cls共享，最后独立 / light;box共享、bias、独立;cls共享，最后独立 / light；cls；box第一个共享
            for a1, a2, a3, b1, b2, b3, s in zip(m.cv4_1, m.cv4_2, m.cv4_3, m.cv5_1, m.cv5_2, m.cv5_3, m.stride):  # from
                a1[-1].bias.data[:] = 1.0  # box
                a2[-1].bias.data[:] = 1.0  # box
                a3[-1].bias.data[:] = 1.0  # box
                b1[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)
                b2[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)
                b3[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)
                # a4[-1].bias.data[:] = 1.0
                # b4[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)
                # a5[-1].bias.data[:] = 1.0
                # b5[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)
                # a6[-1].bias.data[:] = 1.0
                # b6[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)
            ### ###
            # for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            #     a[-1].bias.data[:] = 1.0  # box
            #     b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)
            
        else:
            for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
                a[-1].bias.data[:] = 1.0  # box
                b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)


class Segment(Detect):
    """YOLOv8 Segment head for segmentation models."""

    def __init__(self, nc=80, nm=32, npr=256, ch=()):
        """Initialize the YOLO model attributes such as the number of masks, prototypes, and the convolution layers."""
        super().__init__(nc, ch)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos
        self.detect = Detect.forward

        c4 = max(ch[0] // 4, self.nm)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nm, 1)) for x in ch)

    def forward(self, x):
        """Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients."""
        p = self.proto(x[0])  # mask protos
        bs = p.shape[0]  # batch size

        mc = torch.cat([self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)  # mask coefficients
        x = self.detect(self, x)
        if self.training:
            return x, mc, p
        return (torch.cat([x, mc], 1), p) if self.export else (torch.cat([x[0], mc], 1), (x[1], mc, p))


class Pose(Detect):
    """YOLOv8 Pose head for keypoints models."""

    def __init__(self, nc=80, kpt_shape=(17, 3), ch=()):
        """Initialize YOLO network with default parameters and Convolutional Layers."""
        super().__init__(nc, ch)
        self.kpt_shape = kpt_shape  # number of keypoints, number of dims (2 for x,y or 3 for x,y,visible)
        self.nk = kpt_shape[0] * kpt_shape[1]  # number of keypoints total
        self.detect = Detect.forward

        c4 = max(ch[0] // 4, self.nk)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nk, 1)) for x in ch)

    def forward(self, x):
        """Perform forward pass through YOLO model and return predictions."""
        bs = x[0].shape[0]  # batch size
        kpt = torch.cat([self.cv4[i](x[i]).view(bs, self.nk, -1) for i in range(self.nl)], -1)  # (bs, 17*3, h*w)
        x = self.detect(self, x)
        if self.training:
            return x, kpt
        pred_kpt = self.kpts_decode(bs, kpt)
        return torch.cat([x, pred_kpt], 1) if self.export else (torch.cat([x[0], pred_kpt], 1), (x[1], kpt))

    def kpts_decode(self, bs, kpts):
        """Decodes keypoints."""
        ndim = self.kpt_shape[1]
        if self.export:  # required for TFLite export to avoid 'PLACEHOLDER_FOR_GREATER_OP_CODES' bug
            y = kpts.view(bs, *self.kpt_shape, -1)
            a = (y[:, :, :2] * 2.0 + (self.anchors - 0.5)) * self.strides
            if ndim == 3:
                a = torch.cat((a, y[:, :, 2:3].sigmoid()), 2)
            return a.view(bs, self.nk, -1)
        else:
            y = kpts.clone()
            if ndim == 3:
                y[:, 2::3].sigmoid_()  # inplace sigmoid
            y[:, 0::ndim] = (y[:, 0::ndim] * 2.0 + (self.anchors[0] - 0.5)) * self.strides
            y[:, 1::ndim] = (y[:, 1::ndim] * 2.0 + (self.anchors[1] - 0.5)) * self.strides
            return y


class Classify(nn.Module):
    """YOLOv8 classification head, i.e. x(b,c1,20,20) to x(b,c2)."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        c_ = 1280  # efficientnet_b0 size
        self.conv = Conv(c1, c_, k, s, p, g)
        self.pool = nn.AdaptiveAvgPool2d(1)  # to x(b,c_,1,1)
        self.drop = nn.Dropout(p=0.0, inplace=True)
        self.linear = nn.Linear(c_, c2)  # to x(b,c2)

    def forward(self, x):
        """Performs a forward pass of the YOLO model on input image data."""
        if isinstance(x, list):
            x = torch.cat(x, 1)
        x = self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))
        return x if self.training else x.softmax(1)


class RTDETRDecoder(nn.Module):
    export = False  # export mode

    def __init__(
            self,
            nc=80,
            ch=(512, 1024, 2048),
            hd=256,  # hidden dim
            nq=300,  # num queries
            ndp=4,  # num decoder points
            nh=8,  # num head
            ndl=6,  # num decoder layers
            d_ffn=1024,  # dim of feedforward
            dropout=0.,
            act=nn.ReLU(),
            eval_idx=-1,
            # training args
            nd=100,  # num denoising
            label_noise_ratio=0.5,
            box_noise_scale=1.0,
            learnt_init_query=False):
        super().__init__()
        self.hidden_dim = hd
        self.nhead = nh
        self.nl = len(ch)  # num level
        self.nc = nc
        self.num_queries = nq
        self.num_decoder_layers = ndl

        # backbone feature projection
        self.input_proj = nn.ModuleList(nn.Sequential(nn.Conv2d(x, hd, 1, bias=False), nn.BatchNorm2d(hd)) for x in ch)
        # NOTE: simplified version but it's not consistent with .pt weights.
        # self.input_proj = nn.ModuleList(Conv(x, hd, act=False) for x in ch)

        # Transformer module
        decoder_layer = DeformableTransformerDecoderLayer(hd, nh, d_ffn, dropout, act, self.nl, ndp)
        self.decoder = DeformableTransformerDecoder(hd, decoder_layer, ndl, eval_idx)

        # denoising part
        self.denoising_class_embed = nn.Embedding(nc, hd)
        self.num_denoising = nd
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(nq, hd)
        self.query_pos_head = MLP(4, 2 * hd, hd, num_layers=2)

        # encoder head
        self.enc_output = nn.Sequential(nn.Linear(hd, hd), nn.LayerNorm(hd))
        self.enc_score_head = nn.Linear(hd, nc)
        self.enc_bbox_head = MLP(hd, hd, 4, num_layers=3)

        # decoder head
        self.dec_score_head = nn.ModuleList([nn.Linear(hd, nc) for _ in range(ndl)])
        self.dec_bbox_head = nn.ModuleList([MLP(hd, hd, 4, num_layers=3) for _ in range(ndl)])

        self._reset_parameters()

    def forward(self, x, batch=None):
        from ultralytics.vit.utils.ops import get_cdn_group

        # input projection and embedding
        feats, shapes = self._get_encoder_input(x)

        # prepare denoising training
        dn_embed, dn_bbox, attn_mask, dn_meta = \
            get_cdn_group(batch,
                          self.nc,
                          self.num_queries,
                          self.denoising_class_embed.weight,
                          self.num_denoising,
                          self.label_noise_ratio,
                          self.box_noise_scale,
                          self.training)

        embed, refer_bbox, enc_bboxes, enc_scores = \
            self._get_decoder_input(feats, shapes, dn_embed, dn_bbox)

        # decoder
        dec_bboxes, dec_scores = self.decoder(embed,
                                              refer_bbox,
                                              feats,
                                              shapes,
                                              self.dec_bbox_head,
                                              self.dec_score_head,
                                              self.query_pos_head,
                                              attn_mask=attn_mask)
        x = dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta
        if self.training:
            return x
        # (bs, 300, 4+nc)
        y = torch.cat((dec_bboxes.squeeze(0), dec_scores.squeeze(0).sigmoid()), -1)
        return y if self.export else (y, x)

    def _generate_anchors(self, shapes, grid_size=0.05, dtype=torch.float32, device='cpu', eps=1e-2):
        anchors = []
        for i, (h, w) in enumerate(shapes):
            grid_y, grid_x = torch.meshgrid(torch.arange(end=h, dtype=dtype, device=device),
                                            torch.arange(end=w, dtype=dtype, device=device), indexing='ij')
            grid_xy = torch.stack([grid_x, grid_y], -1)  # (h, w, 2)

            valid_WH = torch.tensor([h, w], dtype=dtype, device=device)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH  # (1, h, w, 2)
            wh = torch.ones_like(grid_xy, dtype=dtype, device=device) * grid_size * (2.0 ** i)
            anchors.append(torch.cat([grid_xy, wh], -1).view(-1, h * w, 4))  # (1, h*w, 4)

        anchors = torch.cat(anchors, 1)  # (1, h*w*nl, 4)
        valid_mask = ((anchors > eps) * (anchors < 1 - eps)).all(-1, keepdim=True)  # 1, h*w*nl, 1
        anchors = torch.log(anchors / (1 - anchors))
        anchors = anchors.masked_fill(~valid_mask, float('inf'))
        return anchors, valid_mask

    def _get_encoder_input(self, x):
        # get projection features
        x = [self.input_proj[i](feat) for i, feat in enumerate(x)]
        # get encoder inputs
        feats = []
        shapes = []
        for feat in x:
            h, w = feat.shape[2:]
            # [b, c, h, w] -> [b, h*w, c]
            feats.append(feat.flatten(2).permute(0, 2, 1))
            # [nl, 2]
            shapes.append([h, w])

        # [b, h*w, c]
        feats = torch.cat(feats, 1)
        return feats, shapes

    def _get_decoder_input(self, feats, shapes, dn_embed=None, dn_bbox=None):
        bs = len(feats)
        # prepare input for decoder
        anchors, valid_mask = self._generate_anchors(shapes, dtype=feats.dtype, device=feats.device)
        features = self.enc_output(valid_mask * feats)  # bs, h*w, 256

        enc_outputs_scores = self.enc_score_head(features)  # (bs, h*w, nc)
        # dynamic anchors + static content
        enc_outputs_bboxes = self.enc_bbox_head(features) + anchors  # (bs, h*w, 4)

        # query selection
        # (bs, num_queries)
        topk_ind = torch.topk(enc_outputs_scores.max(-1).values, self.num_queries, dim=1).indices.view(-1)
        # (bs, num_queries)
        batch_ind = torch.arange(end=bs, dtype=topk_ind.dtype).unsqueeze(-1).repeat(1, self.num_queries).view(-1)

        # Unsigmoided
        refer_bbox = enc_outputs_bboxes[batch_ind, topk_ind].view(bs, self.num_queries, -1)
        # refer_bbox = torch.gather(enc_outputs_bboxes, 1, topk_ind.reshape(bs, self.num_queries).unsqueeze(-1).repeat(1, 1, 4))

        enc_bboxes = refer_bbox.sigmoid()
        if dn_bbox is not None:
            refer_bbox = torch.cat([dn_bbox, refer_bbox], 1)
        if self.training:
            refer_bbox = refer_bbox.detach()
        enc_scores = enc_outputs_scores[batch_ind, topk_ind].view(bs, self.num_queries, -1)

        if self.learnt_init_query:
            embeddings = self.tgt_embed.weight.unsqueeze(0).repeat(bs, 1, 1)
        else:
            embeddings = features[batch_ind, topk_ind].view(bs, self.num_queries, -1)
            if self.training:
                embeddings = embeddings.detach()
        if dn_embed is not None:
            embeddings = torch.cat([dn_embed, embeddings], 1)

        return embeddings, refer_bbox, enc_bboxes, enc_scores

    # TODO
    def _reset_parameters(self):
        # class and bbox head init
        bias_cls = bias_init_with_prob(0.01) / 80 * self.nc
        # NOTE: the weight initialization in `linear_init_` would cause NaN when training with custom datasets.
        # linear_init_(self.enc_score_head)
        constant_(self.enc_score_head.bias, bias_cls)
        constant_(self.enc_bbox_head.layers[-1].weight, 0.)
        constant_(self.enc_bbox_head.layers[-1].bias, 0.)
        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            # linear_init_(cls_)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.layers[-1].weight, 0.)
            constant_(reg_.layers[-1].bias, 0.)

        linear_init_(self.enc_output[0])
        xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        for layer in self.input_proj:
            xavier_uniform_(layer[0].weight)
