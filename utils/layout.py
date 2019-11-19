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
# Modified from https://github.com/google/sg2im/blob/master/sg2im/layout.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import redirect_stderr


def _boxes_to_grid(boxes, grid_h, grid_w):
    """
    Input:
    - boxes: FloatTensor of shape (O, 4) giving boxes in the [x0, y0, x1, y1]
    format in the [0, 1] coordinate space
    - H, W: Scalars giving size of output
    Returns:
    - grid: FloatTensor of shape (O, H, W, 2) suitable for passing to grid_sample
    """
    num_objects = boxes.size(0)
    boxes =  boxes.view(num_objects, 4, 1, 1)

    x0, y0, x1, y1 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    boxes_w = x1 - x0
    boxes_h = y1 - y0

    X = torch.linspace(0, 1, steps=grid_w).view(1, 1, grid_w).to(boxes)
    Y = torch.linspace(0, 1, steps=grid_h).view(1, grid_h, 1).to(boxes)

    X = (X - x0) / boxes_w
    Y = (Y - y0) / boxes_h

    grid = torch.stack([X.expand(num_objects, grid_h, grid_w), Y.expand(num_objects, grid_h, grid_w)], dim=3)
    grid = grid * 2 - 1

    return grid

def _pool_sample(samples, obj_to_img, pooling='sum'):
    assert pooling in ['sum', 'avg']

    O, D, H, W = samples.size()
    N = obj_to_img.data.max().item() + 1

    out = torch.zeros(N, D, H, W, dtype=samples.dtype, device=samples.device)
    idx = obj_to_img.view(O, 1, 1, 1).expand(O, D, H, W)
    out = out.scatter_add(0, idx, samples)

    if pooling == 'avg':
        ones = torch.ones(O, dtype=dtype, device=device)
        obj_counts = torch.zeros(N, dtype=dtype, device=device)
        obj_counts = obj_counts.scatter_add(0, obj_to_img, ones)
        print(obj_counts)
        obj_counts = obj_counts.clamp(min=1)
        out = out / obj_counts.view(N, 1, 1, 1)

    return out

def boxes_to_layout(obj_vectors, boxes, obj_to_img, height, width=None, pooling='sum'):
    """
    Inputs:
    - vecs: Tensor of shape (O, D) giving vectors
    - boxes: Tensor of shape (O, 4) giving bounding boxes in the format
    [x0, y0, x1, y1] in the [0, 1] coordinate space
    - obj_to_img: LongTensor of shape (O,) mapping each element of vecs to
    an image, where each element is in the range [0, N). If obj_to_img[i] = j
    then vecs[i] belongs to image j.
    - H, W: Size of the output
    Returns:
    - out: Tensor of shape (N, D, H, W)
    """
    num_objects, vector_dim = obj_vectors.size()
    if width is None:
        width = height
    
    grid = _boxes_to_grid(boxes, height, width)
    img_in = obj_vectors.view(num_objects, vector_dim, 1, 1). expand(num_objects, vector_dim, 8, 8)
    
    with redirect_stderr(None):
        sampled = F.grid_sample(img_in, grid)
    out = _pool_sample(sampled, obj_to_img, pooling=pooling)

    return out

def masks_to_layout(obj_vectors, boxes, masks, obj_to_img, height, width=None, pooling='sum'):
    num_objects, vector_dim = obj_vectors.size()
    M = masks.size(1)
    assert masks.size() == (num_objects, M, M)
    if width is None:
        width = height

    grid = _boxes_to_grid(boxes, height, width)
    img_in = obj_vectors.view(num_objects, vector_dim, 1, 1) * masks.view(num_objects, 1, M, M).float()
    with redirect_stderr(None):
        sampled = F.grid_sample(img_in, grid)
    out = _pool_sample(sampled, obj_to_img, pooling=pooling)

    return out