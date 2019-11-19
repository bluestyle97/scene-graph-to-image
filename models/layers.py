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
# Modified from https://github.com/google/sg2im/blob/master/sg2im/layers.py

import torch
import torch.nn as nn


def actv(name):
    if name.lower().startswith('leakyrelu'):
        if '-' in name:
            slope = float(name.split('-')[1])
            return nn.LeakyReLU(negative_slope=slope, inplace=True)
        return nn.LeakyReLU(inplace=True)
    elif name.lower() == 'relu':
        return nn.ReLU(inplace=True)
    raise ValueError('Invalid activation "{:s}'.format(name))

def norm2d(name, channels):
    assert name in ['batch', 'instance', 'none']

    if name == 'batch':
        return nn.BatchNorm2d(channels)
    elif name == 'instance':
        return nn.InstanceNorm2d(channels, affine=True, track_running_stats=True)
        
    return None

def mlp(dim_list, activation='relu', norm='none', final_nonlinearity=True):
    assert norm in ['batch', 'none']

    layers = []
    for i in range(len(dim_list) - 1):
        dim_in, dim_out = dim_list[i], dim_list[i + 1]
        layers.append(nn.Linear(dim_in, dim_out))
        final_layer = (i == len(dim_list) - 2)
        if not final_layer or final_nonlinearity:
            if norm == 'batch':
                layers.append(nn.BatchNorm1d(dim_out))
            layers.append(actv(activation))

    return nn.Sequential(*layers)