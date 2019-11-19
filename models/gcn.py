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
# Modified from https://github.com/google/sg2im/blob/master/sg2im/graph.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers import mlp


class GraphConvLayer(nn.Module):
    def __init__(self, in_dim, out_dim=None, hidden_dim=512, pooling='avg', activation='relu', mlp_norm='none'):
        super(GraphConvLayer, self).__init__()

        if out_dim is None:
            out_dim = in_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim

        assert pooling in ['sum', 'avg']
        self.pooling = pooling

        self.gp = mlp([3*in_dim, hidden_dim, out_dim], activation, mlp_norm)
        self.gs = mlp([3*in_dim, hidden_dim, out_dim], activation, mlp_norm)
        self.go = mlp([3*in_dim, hidden_dim, out_dim], activation, mlp_norm)
    
    def forward(self, obj_vectors, pred_vectors, edges):
        num_objects, num_predicates = obj_vectors.size(0), pred_vectors.size(0)

        obj_s_idxs = edges[:, 0].view(num_predicates)
        obj_o_idxs = edges[:, 1].view(num_predicates)

        old_s_vectors = obj_vectors.index_select(0, obj_s_idxs)
        old_o_vectors = obj_vectors.index_select(0, obj_o_idxs)
        old_vectors = torch.cat([old_o_vectors, pred_vectors, old_o_vectors], dim=1)
        
        new_pred_vectors = self.gp(old_vectors)

        new_s_vectors = self.gs(old_vectors)
        new_o_vectors = self.go(old_vectors)

        new_obj_vectors = torch.zeros(num_objects, self.out_dim, dtype=obj_vectors.dtype, device=obj_vectors.device)
        new_obj_vectors.scatter_add_(0, obj_s_idxs.view(-1, 1).expand_as(new_s_vectors), new_s_vectors)
        new_obj_vectors.scatter_add_(0, obj_o_idxs.view(-1, 1).expand_as(new_o_vectors), new_o_vectors)

        if self.pooling == 'avg':
            obj_counts = torch.zeros(num_objects, dtype=obj_vectors.dtype, device=obj_vectors.device)
            ones = torch.ones(num_predicates, dtype=obj_vectors.dtype, device=obj_vectors.device)
            obj_counts.scatter_add_(0, obj_s_idxs, ones)
            obj_counts.scatter_add_(0, obj_o_idxs, ones)
            obj_counts.clamp_(min=1)
            new_obj_vectors = new_obj_vectors / obj_counts.view(-1, 1)

        return new_obj_vectors, new_pred_vectors

class GraphConvNet(nn.Module):
    def __init__(self, in_dim, hidden_dim=512, num_layers=5, pooling='avg', activation='relu', mlp_norm='none'):
        super(GraphConvNet, self).__init__()

        self.num_layers = num_layers
        self.gconvs = nn.ModuleList()
        for _ in range(self.num_layers):
            self.gconvs.append(GraphConvLayer(in_dim, hidden_dim=hidden_dim, pooling=pooling, activation=activation, mlp_norm=mlp_norm))

    def forward(self, obj_vectors, pred_vectors, edges):
        for i in range(self.num_layers):
            obj_vectors, pred_vectors = self.gconvs[i](obj_vectors, pred_vectors, edges)

        return obj_vectors, pred_vectors


if __name__ == '__main__':
    gcn = GraphConvNet(128)

    # 10 objects, 15 edges
    obj_vectors = torch.rand(10, 128)
    pred_vectors = torch.rand(15, 128)
    edges = torch.randint(10, (15, 2))

    new_obj_vectors, new_pred_vectors = gcn(obj_vectors, pred_vectors, edges)
    print(new_obj_vectors.size())
    print(new_obj_vectors)
    print(new_pred_vectors.size())
    print(new_pred_vectors)