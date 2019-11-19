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
# Modified from https://github.com/google/sg2im/blob/master/sg2im/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers import mlp
from models.gcn import GraphConvLayer, GraphConvNet
from models.crn import RefinementNet
from utils.layout import boxes_to_layout, masks_to_layout


class SceneGraph2Image(nn.Module):
    def __init__(self, vocab, image_size=(128, 128), embedding_dim=128, 
    gcn_out_dim=128, gcn_hidden_dim=512, gcn_num_layers=5, gcn_pooling='avg', 
    mlp_activation='relu', mlp_norm='none', mask_size=None, 
    layout_noise_dim=0, crn_out_dims=[1024, 512, 256, 128, 64], crn_activation='leakyrelu', crn_norm='batch'):
        super(SceneGraph2Image, self).__init__()

        self.vocab = vocab
        self.image_size = image_size
        self.layout_noise_dim = layout_noise_dim

        num_objects = len(vocab['object_idx_to_name'])
        num_predicates = len(self.vocab['pred_idx_to_name'])
        self.object_embeddings = nn.Embedding(num_objects + 1, embedding_dim)   # why + 1?
        self.predicate_embeddings = nn.Embedding(num_predicates, embedding_dim)
        
        # GCN
        self.gcn_layer = GraphConvLayer(embedding_dim, gcn_out_dim, gcn_hidden_dim, gcn_pooling, mlp_activation, mlp_norm)
        self.gcn = GraphConvNet(gcn_out_dim, gcn_hidden_dim, gcn_num_layers, gcn_pooling, mlp_activation, mlp_norm)

        # OLN
        self.box_net = mlp([gcn_out_dim, gcn_hidden_dim, 4], mlp_activation, mlp_norm)
        self.mask_net = None
        if mask_size is not None and mask_size > 0:
            cur_size = 1
            layers = []
            while cur_size < mask_size:
                layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
                layers.append(nn.BatchNorm2d(gcn_out_dim))
                layers.append(nn.Conv2d(gcn_out_dim, gcn_out_dim, 3, stride=1, padding=1))
                layers.append(nn.ReLU(inplace=True))
                cur_size *= 2
            assert cur_size == mask_size
            layers.append(nn.Conv2d(gcn_out_dim, 1, 1, stride=1, padding=1))
            self.mask_net = nn.Sequential(*layers)

        # CRN
        self.crn = RefinementNet(gcn_out_dim + self.layout_noise_dim, crn_out_dims, crn_activation, crn_norm)

    def forward(self, objs, triples, obj_to_img=None, boxes_gt=None, masks_gt=None):
        num_objects, num_edges = objs.size(0), triples.size(0)
        s, p, o = triples.chunk(3, dim=1)
        edges = torch.cat([s, o], dim=1)

        if obj_to_img is None:
            obj_to_img = torch.zeros(num_objects, dtype=objs.dtype, device=objs.device)

        obj_vectors = self.object_embeddings(objs.view(-1))
        pred_vectors = self.predicate_embeddings(p.view(-1))

        # forward GCN
        obj_vectors, pred_vectors = self.gcn_layer(obj_vectors, pred_vectors, edges)
        obj_vectors, pred_vectors = self.gcn(obj_vectors, pred_vectors, edges)

        # forward OLN
        boxes = self.box_net(obj_vectors)
        masks = None
        if self.mask_net is not None:
            mask_scores = self.mask_net(obj_vectors.view(num_objects, -1, 1, 1))
            masks = F.sigmoid(mask_scores.squeeze(1))

        # generate layout
        s_boxes = boxes.index_select(0, s.view(-1))
        o_boxes = boxes.index_select(0, o.view(-1))
        H, W = self.image_size
        layout_boxes = boxes if boxes_gt is None else boxes_gt
        if masks is None:
            layout = boxes_to_layout(obj_vectors, layout_boxes, obj_to_img, H, W)
        else:
            layout_masks = masks if masks_gt is None else masks_gt
            layout = masks_to_layout(obj_vectors, layout_boxes, layout_masks, obj_to_img, H, W)

        if self.layout_noise_dim > 0:
            N, C, H, W = layout.size()
            layout_noise = torch.randn(N, self.layout_noise_dim, H, W, dtype=layout.dtype, device=layout.device)
            layout = torch.cat([layout, layout_noise], dim=1)

        # forward CRN
        image = self.crn(layout)
        
        return image, boxes, masks


if __name__ == '__main__':
    import os
    import json
    from datasets.visual_genome import get_vg_data_loader

    vocab_file = os.path.join('/p300/visual_genome/v1.4', 'vocab.json')
    with open(vocab_file, 'r') as f:
        vocab = json.load(f)
    data_loader = get_vg_data_loader('/p300/visual_genome/v1.4', vocab, batch_size=4, image_size=(256, 256), max_objects=10, include_relationships=True, use_orphaned_objects=True, split='train')
    all_imgs, all_objs, all_boxes, all_triples, all_obj_to_img, all_triple_to_img = next(iter(data_loader))

    model = SceneGraph2Image(vocab, layout_noise_dim=32)

    image, boxes, masks = model(all_objs, all_triples, all_obj_to_img, all_boxes)
    print(image.size())
    print(image)
    print(boxes.size())
    print(boxes)
    print(masks)