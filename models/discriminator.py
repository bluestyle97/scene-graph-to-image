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
# Modified from https://github.com/google/sg2im/blob/master/sg2im/discriminators.py

import torch
import torch.nn as nn
import torch.nn.functional as F 

from models.layers import actv, norm2d


class ImageDiscriminator(nn.Module):
    def __init__(self, layout_dim=0, conv_dims=[64, 128, 256], fc_dim=1024, image_size=(128, 128), activation='leakyrelu-0.2', norm='batch'):
        super(ImageDiscriminator, self).__init__()

        self.image_size = image_size

        in_dim = 3 + layout_dim
        layers = []
        for conv_dim in conv_dims:
            layers.append(nn.Conv2d(in_dim, conv_dim, 4, stride=2, padding=1))
            layers.append(norm2d(norm, conv_dim))
            layers.append(actv(activation))
            in_dim = conv_dim
        layers = [layer for layer in layers if layer is not None]
        self.conv = nn.Sequential(*layers)

        H, W = image_size
        feat_size = (H // 2**len(conv_dims)) * (W // 2**len(conv_dims)) * conv_dims[-1]
        self.fc = nn.Sequential(
            nn.Linear(feat_size, fc_dim),
            actv(activation),
            nn.Linear(fc_dim, 1)
        )

    def forward(self, x, layout=None):
        assert x.size(2) == self.image_size[0] and x.size(3) == self.image_size[1]

        if layout is not None:
            x = torch.cat([x, layout], dim=1)
        feat = self.conv(x)
        out = self.fc(feat.view(feat.size(0), -1))

        return out

class ObjectDiscriminator(nn.Module):
    def __init__(self, num_classes, conv_dims=[64, 128, 256], fc_dim=512, object_size=(64, 64), activation='leakyrelu-0.2', norm='batch'):
        super(ObjectDiscriminator, self).__init__()

        self.object_size = object_size

        in_dim = 3
        layers = []
        for conv_dim in conv_dims:
            layers.append(nn.Conv2d(in_dim, conv_dim, 4, stride=2, padding=1))
            layers.append(norm2d(norm, conv_dim))
            layers.append(actv(activation))
            in_dim = conv_dim
        layers = [layer for layer in layers if layer is not None]
        layers.append(nn.AdaptiveAvgPool2d(1))
        self.conv = nn.Sequential(*layers)

        self.fc_real = nn.Sequential(
            nn.Linear(conv_dims[-1], fc_dim),
            actv(activation),
            nn.Linear(fc_dim, 1)
        )
        self.fc_obj = nn.Sequential(
            nn.Linear(conv_dims[-1], fc_dim),
            actv(activation),
            nn.Linear(fc_dim, num_classes)
        )

    def forward(self, x):
        assert x.size(2) == self.object_size[0] and x.size(3) == self.object_size[1]

        feat = self.conv(x)
        real_logits = self.fc_real(feat.view(x.size(0), -1))
        obj_logits = self.fc_obj(feat.view(x.size(0), -1))

        return real_logits, obj_logits


if __name__ == '__main__':
    from torchsummary import summary

    img_discriminator = ImageDiscriminator()
    obj_discriminator = ObjectDiscriminator(10)
    
    summary(img_discriminator, (3, 128, 128), device='cpu')
    summary(obj_discriminator, (3, 64, 64), device='cpu')