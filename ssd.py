# vggネットワークを生成する関数の定義

import re
from turtle import forward
from matplotlib.pyplot import box
from torch import conv2d
import torch.nn as nn

def make_vgg():
    layers = []
    in_channels = 3

    cfg = [
        64, 64, "M", # vgg1
        128, 128, "M", # vgg2
        256, 256, 256, "MC", # vgg3
        512, 512, 512, "M", # vgg4
        512, 512, 512 # vgg5
    ]

    # vgg1~vgg5の畳み込み層までを生成
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == "MC":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]

    return nn.ModuleList(layers)

# extrasネットワークを生成する関数の定義

def make_extras():
    layers = []
    in_channels = 1024

    cfg = [
        256, 512,
        128, 256,
        128, 256,
        128, 256
    ]

    layers += [nn.Conv2d(in_channels, cfg[0], kernel_size=(1))]
    layers += [nn.Conv2d(cfg[0], cfg[1], kernel_size=(3), stride=2, padding=1)]

    layers += [nn.Conv2d(cfg[1], cfg[2], kernel_size=(1))]
    layers += [nn.Conv2d(cfg[2], cfg[3], kernel_size=(3), stride=2, padding=1)]

    layers += [nn.Conv2d(cfg[3], cfg[4], kernel_size=(1))]
    layers += [nn.Conv2d(cfg[4], cfg[5], kernel_size=(3))]

    layers += [nn.Conv2d(cfg[5], cfg[6], kernel_size=(1))]
    layers += [nn.Conv2d(cfg[6], cfg[7], kernel_size=(3))]

    return nn.ModuleList(layers)


# locネットワークを生成する関数の定義

def make_loc(dbox_num=[4, 6, 6, 6, 4, 4]):
    loc_layers = []

    loc_layers += [nn.Conv2d(512, dbox_num[0] * 4, kernel_size=3, padding=1)]
    loc_layers += [nn.Conv2d(1024, dbox_num[1] * 4, kernel_size=3, padding=1)]
    loc_layers += [nn.Conv2d(512, dbox_num[2] * 4, kernel_size=3, padding=1)]
    loc_layers += [nn.Conv2d(256, dbox_num[3] * 4, kernel_size=3, padding=1)]
    loc_layers += [nn.Conv2d(256, dbox_num[4] * 4, kernel_size=3, padding=1)]
    loc_layers += [nn.Conv2d(256, dbox_num[5] * 4, kernel_size=3, padding=1)]

    return nn.ModuleList(loc_layers)

# confネットワークを生成する関数の定義

def make_conf(classes_num=21, dbox_num=[4, 6, 6, 6, 4, 4]):
    conf_layers = []

    conf_layers += [nn.Conv2d(512, dbox_num[0] * classes_num, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(1024, dbox_num[1] * classes_num, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(512, dbox_num[2] * classes_num, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(256, dbox_num[3] * classes_num, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(256, dbox_num[4] * classes_num, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(256, dbox_num[5] * classes_num, kernel_size=3, padding=1)]

    return nn.ModuleList(conf_layers)


# L2ノルムで正規化する層を生成するクラス

import torch
import torch.nn.init as init

class L2Norm(nn.Module):
    def __init__(self, input_channels=512, scale=20):
        super(L2Norm, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(input_channels))
        self.scale = scale
        self.reset_paramerters()
        self.eps = 1e-10

    def reset_parameters(self):
        init.constant_(self.weight, self.scale)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = torch.div(x, norm)

        weights = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).extend_as(x)
        out = weights*x

        return out

# デフォルトボックスを出力するクラス

from itertools import product
from math import sqrt

class DBox(object):
    def __init__(self, cfg):
        super(DBox, self).__init__()
        self.image_size = cfg["input_size"]
        self.feature_maps = cfg["feature_maps"]
        self.num_priors = len(cfg["feature_maps"])
        self.steps = cfg["steps"]
        self.min_sizes = cfg["min_sizes"]
        self.max_sizes = cfg["max_sizes"]
        self.aspect_ratios = cfg["aspect_ratios"]

    def make_dbox_list(self):
        mean = []
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat=2):
                f_k = self.image_size / self.steps[k]
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                s_k = self.min_sizes[k]/self.image_size
                mean += [cx, cy, s_k, s_k]

                s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]

                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k/sqrt(ar), s_k/sqrt(ar)]
                    mean += [cx, cy, s_k*sqrt(ar), s_k*sqrt(ar)]

        output = torch.Tensor(mean).view(-1, 4)
        output.clamp_(max=1, min=0)

        return output

# デフォルトボックスをバウンディングボックスに変換する関数

def decode(loc, dbox_list):
    boxes = torch.cat((
        dbox_list[:, :2] + loc[:, :2] * 0.1 * dbox_list[:, 2:],
        dbox_list[:, 2:] * torch.exp(loc[:, 2:] * 0.2)
    ), dim=1)

    boxes[:, :2] -= boxes[:, 2:] /2
    boxes[:, 2:] += boxes[:, :2]

    return boxes

def nonmaximum_suppress(boxes, scores, overlap=0.5, top_k=200):
    count = 0
    keep = scores.new(scores.size(0)).zero_().long()

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.num(x2 - x1, y2 - y1)

    tmp_x1 = boxes.new()
    tmp_y1 = boxes.new()
    tmp_x2 = boxes.new()
    tmp_y2 = boxes.new()
    tmp_w = boxes.new()
    tmp_h = boxes.new()

    v, idx = scores.sort(0)

    idx = idx[-top_k:]

    while idx.numl() > 0:
        i = idx[-1]

        keep[count] = i
        count += 1

        if idx.size(0) == 1:
            break

        idx = idx[:-1]

        torch.index_select(x1, 0, idx, out=tmp_x1)
        torch.index_select(y1, 0, idx, out=tmp_y1)
        torch.index_select(x2, 0, idx, out=tmp_x2)
        torch.index_select(y2, 0, idx, out=tmp_y2)

        tmp_x1 = torch.clamp(tmp_x1, min=x1[i])
        tmp_y1 = torch.clamp(tmp_y1, min=y1[i])
        tmp_x2 = torch.clamp(tmp_x2, max=x2[i])
        tmp_y2 = torch.clamp(tmp_y2, max=y2[i])

        tmp_w.resize_as_(tmp_x2)
        tmp_h.resize_as_(tmp_y2)

        tmp_w = tmp_x2 - tmp_x1
        tmp_h = tmp_y2 - tmp_y1

        tmp_w = torch.clamp(tmp_w, min=0.0)
        tmp_h = torch.clamp(tmp_h, min=0.0)

        inter = tmp_w * tmp_h

        rem_areas = torch.index_select(area, 0, idx)

        union = (rem_areas - inter) + area[i]
        IoU = inter/union

        idx = idx[IoU.le(overlap)]

    return keep, count


# SSDの推論時にconfとlocの出力から真のBBoxを抽出するクラス

from torch.autograd import Function

class Detect(Function):
    @staticmethod
    def forward(ctx, loc_data, conf_data, dbox_list):
        ctx.softmax = nn.Softmax(dim=-1)
        ctx.conf_thresh = 0.01
        ctx.top_k = 200
        ctx.nms_thresh = 0.45

        batch_num = conf_data.size(0)
        classes_num = conf_data.size(2)

        conf_data = ctx.softmax(conf_data)
        conf_preds = conf_data.transpose(2, 1)

        output = torch.zeros(batch_num, classes_num, ctx.top_k, 5)

        for i in range(batch_num):
            decoded_boxes = decode(loc_data[i], dbox_list)

            conf_scores = conf_preds[i].clone()

            for cl in range(1, classes_num):
                c_mask = conf_scores[cl].gt(ctx.conf_thresh)
                scores = conf_scores[cl][c_mask]
                if scores.nelement() == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)
                ids, count = nonmaximum_suppress(
                    boxes,
                    scores,
                    ctx.nms_thresh,
                    ctx.top_k
                )

                output[i, cl, :count] = torch.cat(
                    (scores[ids[:count]].unsqueeze(1), boxes[ids[:count]]),
                    1
                )

        return output

# SSDクラスを作成する

import torch.nn.functional as F

class SSD(nn.Module):
    def __init__(self, phase, cfg):
        super(SSD, self).__init__()
        self.phase = phase
        self.classes_num = cfg["classes_num"]
        
        self.vgg = make_vgg()
        self.extras = make_extras()
        self.L2Norm = L2Norm()
        self.loc = make_loc(cfg["dbox_num"])
        self.conf = make_conf(cfg["classes_num"], cfg["dbox_num"])
        
        dbox = DBox(cfg)
        self.dbox_list = dbox.make_dbox_list()

        if phase == "tesf":
            self.detect = Detect.apply

    def forward(self, x):
        out_list = []
        loc = []
        conf = []

        for k in range(23):
            x = self.vgg[k](x)
        out1 = self.L2Norm(x)
        out_list.append(out1)

        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        out_list.append(x)

        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                out_list.append(x)
        
        for (x, l, c) in zip(out_list, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        loc = loc.view(loc.size(0), -1, 4)
        conf = conf.view(conf.size(0), -1, self.classes_num)

        output = (loc, conf, self.dbox_list)

        if self.phase == "test":
            return self.detect(output[0], output[1], output[2])

        else:
            return output
