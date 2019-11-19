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
# Modified from https://github.com/google/sg2im/blob/master/sg2im/bilinear.py

import torch
import torch.nn.functional as F
from contextlib import redirect_stderr


def _invperm(p):
    N = p.size(0)
    eye = torch.arange(0, N).type_as(p)
    pp = (eye[:, None] == p).nonzero()[:, 1]
    return pp

def bilinear_sample(feats, X, Y):
    """
    Perform bilinear sampling on the features in feats using the sampling grid
    given by X and Y.
    Inputs:
    - feats: Tensor holding input feature map, of shape (N, C, H, W)
    - X, Y: Tensors holding x and y coordinates of the sampling
        grids; both have shape shape (N, HH, WW) and have elements in the range [0, 1].
    Returns:
    - out: Tensor of shape (B, C, HH, WW) where out[i] is computed
        by sampling from feats[idx[i]] using the sampling grid (X[i], Y[i]).
    """
    N, C, H, W = feats.size()
    assert X.size() == Y.size()
    assert X.size(0) == N
    _, HH, WW = X.size()

    X = X.mul(W)
    Y = Y.mul(H)

    # Get the x and y coordinates for the four samples
    x0 = X.floor().clamp(min=0, max=W-1)
    x1 = (x0 + 1).clamp(min=0, max=W-1)
    y0 = Y.floor().clamp(min=0, max=H-1)
    y1 = (y0 + 1).clamp(min=0, max=H-1)

    # In numpy we could do something like feats[i, :, y0, x0] to pull out
    # the elements of feats at coordinates y0 and x0, but PyTorch doesn't
    # yet support this style of indexing. Instead we have to use the gather
    # method, which only allows us to index along one dimension at a time;
    # therefore we will collapse the features (BB, C, H, W) into (BB, C, H * W)
    # and index along the last dimension. Below we generate linear indices into
    # the collapsed last dimension for each of the four combinations we need.
    y0x0_idx = (W * y0 + x0).view(N, 1, HH * WW).expand(N, C, HH * WW)
    y1x0_idx = (W * y1 + x0).view(N, 1, HH * WW).expand(N, C, HH * WW)
    y0x1_idx = (W * y0 + x1).view(N, 1, HH * WW).expand(N, C, HH * WW)
    y1x1_idx = (W * y1 + x1).view(N, 1, HH * WW).expand(N, C, HH * WW)

    # Actually use gather to pull out the values from feats corresponding
    # to our four samples, then reshape them to (BB, C, HH, WW)
    feats_flat = feats.view(N, C, H * W)
    v1 = feats_flat.gather(2, y0x0_idx.long()).view(N, C, HH, WW)
    v2 = feats_flat.gather(2, y1x0_idx.long()).view(N, C, HH, WW)
    v3 = feats_flat.gather(2, y0x1_idx.long()).view(N, C, HH, WW)
    v4 = feats_flat.gather(2, y1x1_idx.long()).view(N, C, HH, WW)

    # Compute the weights for the four samples
    w1 = ((x1 - X) * (y1 - Y)).view(N, 1, HH, WW).expand(N, C, HH, WW)
    w2 = ((x1 - X) * (Y - y0)).view(N, 1, HH, WW).expand(N, C, HH, WW)
    w3 = ((X - x0) * (y1 - Y)).view(N, 1, HH, WW).expand(N, C, HH, WW)
    w4 = ((X - x0) * (Y - y0)).view(N, 1, HH, WW).expand(N, C, HH, WW)

    # Multiply the samples by the weights to give our interpolated results.
    out = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4
    return out

def tensor_linspace(start, end, steps=10):
    """
    Vectorized version of torch.linspace.
    Inputs:
    - start: Tensor of any shape
    - end: Tensor of the same shape as start
    - steps: Integer
    Returns:
    - out: Tensor of shape start.size() + (steps,), such that
        out.select(-1, 0) == start, out.select(-1, -1) == end,
        and the other elements of out linearly interpolate between
        start and end.
    """
    assert start.size() == end.size()
    view_size = start.size() + (1,)
    w_size = (1,) * start.dim() + (steps,)
    out_size = start.size() + (steps,)

    start_w = torch.linspace(1, 0, steps=steps).to(start)
    start_w = start_w.view(w_size).expand(out_size)
    end_w = torch.linspace(0, 1, steps=steps).to(start)
    end_w = end_w.view(w_size).expand(out_size)

    start = start.contiguous().view(view_size).expand(out_size)
    end = end.contiguous().view(view_size).expand(out_size)

    out = start_w * start + end_w * end
    return out

def crop_bbox(feats, bbox, HH, WW=None, backend='cudnn'):
    """
    Take differentiable crops of feats specified by bbox.
    Inputs:
    - feats: Tensor of shape (N, C, H, W)
    - bbox: Bounding box coordinates of shape (N, 4) in the format
        [x0, y0, x1, y1] in the [0, 1] coordinate space.
    - HH, WW: Size of the output crops.
    Returns:
    - crops: Tensor of shape (N, C, HH, WW) where crops[i] is the portion of
        feats[i] specified by bbox[i], reshaped to (HH, WW) using bilinear sampling.
    """
    N = feats.size(0)
    assert bbox.size(0) == N
    assert bbox.size(1) == 4
    if WW is None: 
        WW = HH
    if backend == 'cudnn':
        # Change box from [0, 1] to [-1, 1] coordinate system
        bbox = 2 * bbox - 1
    x0, y0 = bbox[:, 0], bbox[:, 1]
    x1, y1 = bbox[:, 2], bbox[:, 3]
    X = tensor_linspace(x0, x1, steps=WW).view(N, 1, WW).expand(N, HH, WW)
    Y = tensor_linspace(y0, y1, steps=HH).view(N, HH, 1).expand(N, HH, WW)
    if backend == 'jj':
        return bilinear_sample(feats, X, Y)
    elif backend == 'cudnn':
        grid = torch.stack([X, Y], dim=3)
        with redirect_stderr(None):
            result = F.grid_sample(feats, grid)
        return result

def crop_bbox_batch_cudnn(feats, bbox, bbox_to_feats, HH, WW=None):
    N, C, H, W = feats.size()
    B = bbox.size(0)
    if WW is None: WW = HH
    dtype = feats.data.type()

    feats_flat, bbox_flat, all_idx = [], [], []
    for i in range(N):
        idx = (bbox_to_feats.data == i).nonzero()
        if idx.dim() == 0:
            continue
        idx = idx.view(-1)
        n = idx.size(0)
        cur_feats = feats[i].view(1, C, H, W).expand(n, C, H, W).contiguous()
        cur_bbox = bbox[idx]

        feats_flat.append(cur_feats)
        bbox_flat.append(cur_bbox)
        all_idx.append(idx)

    feats_flat = torch.cat(feats_flat, dim=0)
    bbox_flat = torch.cat(bbox_flat, dim=0)
    crops = crop_bbox(feats_flat, bbox_flat, HH, WW, backend='cudnn')

    # If the crops were sequential (all_idx is identity permutation) then we can
    # simply return them; otherwise we need to permute crops by the inverse
    # permutation from all_idx.
    all_idx = torch.cat(all_idx, dim=0)
    eye = torch.arange(0, B).type_as(all_idx)
    if (all_idx == eye).all():
        return crops
    return crops[_invperm(all_idx)]

def crop_bbox_batch(feats, bbox, bbox_to_feats, HH, WW=None, backend='cudnn'):
    """
    Inputs:
    - feats: FloatTensor of shape (N, C, H, W)
    - bbox: FloatTensor of shape (B, 4) giving bounding box coordinates
    - bbox_to_feats: LongTensor of shape (B,) mapping boxes to feature maps;
        each element is in the range [0, N) and bbox_to_feats[b] = i means that
        bbox[b] will be cropped from feats[i].
    - HH, WW: Size of the output crops
    Returns:
    - crops: FloatTensor of shape (B, C, HH, WW) where crops[i] uses bbox[i] to
        crop from feats[bbox_to_feats[i]].
    """
    if backend == 'cudnn':
        return crop_bbox_batch_cudnn(feats, bbox, bbox_to_feats, HH, WW)
    N, C, H, W = feats.size()
    B = bbox.size(0)
    if WW is None: WW = HH
    dtype, device = feats.dtype, feats.device
    crops = torch.zeros(B, C, HH, WW, dtype=dtype, device=device)
    for i in range(N):
        idx = (bbox_to_feats.data == i).nonzero()
        if idx.dim() == 0:
            continue
        idx = idx.view(-1)
        n = idx.size(0)
        cur_feats = feats[i].view(1, C, H, W).expand(n, C, H, W).contiguous()
        cur_bbox = bbox[idx]
        cur_crops = crop_bbox(cur_feats, cur_bbox, HH, WW)
        crops[idx] = cur_crops
    return crops