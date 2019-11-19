# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified from https://github.com/google/sg2im/blob/master/sg2im/crn.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers import actv, norm2d


class RefinementModule(nn.Module):
    def __init__(self, layout_dim, in_dim, out_dim, activation='leakyrelu', norm='batch'):
        super(RefinementModule, self).__init__()

        layers = []
        layers.append(nn.Conv2d(layout_dim + in_dim, out_dim, 3, stride=1, padding=1))
        layers.append(norm2d(norm, out_dim))
        layers.append(actv(activation))
        layers.append(nn.Conv2d(out_dim, out_dim, 3, stride=1, padding=1))
        layers.append(norm2d(norm, out_dim))
        layers.append(actv(activation))
        layers = [layer for layer in layers if layer is not None]

        self.net = nn.Sequential(*layers)

    def forward(self, layout, feats):
        _, _, layout_H, layout_W = layout.size()
        _, _, feats_H, feats_W = feats.size()
        assert layout_H >= feats_H

        if layout_H > feats_H:
            factor = round(layout_H // feats_H)
            assert layout_H % factor == 0
            assert layout_W % factor == 0 and layout_W // factor == feats_W
            layout = F.avg_pool2d(layout, kernel_size=factor, stride=factor)
        net_input = torch.cat([layout, feats], dim=1)
        out = self.net(net_input)

        return out

class RefinementNet(nn.Module):
    def __init__(self, layout_dim=128, out_dims=[1024, 512, 256, 128, 64], activation='leakyrelu', norm='batch'):
        super(RefinementNet, self).__init__()
        
        in_dim = 1
        self.refinement_modules = nn.ModuleList()
        for out_dim in out_dims:
            self.refinement_modules.append(RefinementModule(layout_dim, in_dim, out_dim, activation=activation, norm=norm))
            in_dim = out_dim

        self.conv_output = nn.Sequential(
            nn.Conv2d(out_dims[-1], out_dims[-1], 3, stride=1, padding=1),
            actv(activation),
            nn.Conv2d(out_dims[-1], 3, 1, stride=1, padding=0)
        )

    def forward(self, layout):
        """
        Output will have same size as layout
        """
        N, _, H, W = layout.size()
        input_H = H // 2**len(self.refinement_modules)
        input_W = W // 2**len(self.refinement_modules)

        assert input_H > 0
        assert input_W > 0

        feats = torch.zeros(N, 1, input_H, input_W).to(layout)
        for module in self.refinement_modules:
            feats = F.interpolate(feats, scale_factor=2, mode='nearest')
            feats = module(layout, feats)

        out = self.conv_output(feats)
        return out


if __name__ == '__main__':
    from torchsummary import summary

    refine_net = RefinementNet()
    summary(refine_net, (128, 256, 256), device='cpu')