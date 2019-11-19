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
# Modified from https://github.com/google/sg2im/blob/master/sg2im/data/vg.py

import os
import json
import random
import h5py
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from collections import defaultdict


class VisualGenomeDataset(data.Dataset):
    def __init__(self, data_root, vocab, image_transforms=None, max_objects=10, include_relationships=True, use_orphaned_objects=True, split='train'):
        super(VisualGenomeDataset, self).__init__()

        self.data_root = data_root
        self.vocab = vocab
        self.image_transforms = image_transforms
        self.max_objects = max_objects
        self.include_relationships = include_relationships
        self.use_orphaned_objects = use_orphaned_objects
        self.split = split

        self.data = {}
        h5_file = os.path.join(data_root, '{:s}.h5'.format(self.split))
        with h5py.File(h5_file, 'r') as f:
            for k, v in f.items():
                if k == 'image_paths':
                    self.image_paths = list(v)
                else:
                    self.data[k] = torch.IntTensor(np.asarray(v))

    def __len__(self):
        return self.data['object_names'].size(0)

    def __getitem__(self, index):
        """
        Returns a tuple of:
        - image: FloatTensor of shape (C, H, W)
        - objs: LongTensor of shape (O,)
        - boxes: FloatTensor of shape (O, 4) giving boxes for objects in
        (x0, y0, x1, y1) format, in a [0, 1] coordinate system.
        - triples: LongTensor of shape (T, 3) where triples[t] = [i, p, j]
        means that (objs[i], p, objs[j]) is a triple.
        """
        image_path = os.path.join(self.data_root, self.image_paths[index])
        image = Image.open(image_path).convert('RGB')
        if self.image_transforms is not None:
            image = self.image_transforms(image)
        width, height = image.size(1), image.size(2)

        # figure out which objects appear in relationships and which don't
        obj_idxs_with_rels = set()
        obj_idxs_without_rels = set(range(self.data['objects_per_image'][index].item()))
        for r_idx in range(self.data['relationships_per_image'][index]):
            s = self.data['relationship_subjects'][index, r_idx].item()
            o = self.data['relationship_objects'][index, r_idx].item()
            obj_idxs_with_rels.add(s)
            obj_idxs_with_rels.add(o)
            obj_idxs_without_rels.discard(s)
            obj_idxs_without_rels.discard(o)

        obj_idxs = list(obj_idxs_with_rels)
        obj_idxs_without_rels = list(obj_idxs_without_rels)
        if len(obj_idxs) < self.max_objects - 1 and self.use_orphaned_objects:
            num_to_add = self.max_objects - 1 - len(obj_idxs)
            num_to_add = min(num_to_add, len(obj_idxs_without_rels))
            obj_idxs += random.sample(obj_idxs_without_rels, num_to_add)
        num_objects = len(obj_idxs) + 1
        
        objs = torch.LongTensor(num_objects).fill_(-1)

        boxes = torch.FloatTensor([[0, 0, 1, 1]]).repeat(num_objects, 1)
        obj_idx_mapping = {}
        for i, obj_idx in enumerate(obj_idxs):
            objs[i] = self.data['object_names'][index, obj_idx].item()
            x, y, w, h = self.data['object_boxes'][index, obj_idx].tolist()
            x0 = float(x) / width
            y0 = float(y) / height
            x1 = float(x + w) / width
            y1 = float(y + h) / height
            boxes[i] = torch.FloatTensor([x0, y0, x1, y1])
            obj_idx_mapping[obj_idx] = i

        # the last object will be the special __image__ object
        objs[num_objects - 1] = self.vocab['object_name_to_idx']['__image__']

        triples = []
        if self.include_relationships:
            for r_idx in range(self.data['relationships_per_image'][index].item()):
                s = self.data['relationship_subjects'][index, r_idx].item()
                p = self.data['relationship_predicates'][index, r_idx].item()
                o = self.data['relationship_objects'][index, r_idx].item()
                s = obj_idx_mapping.get(s, None)
                o = obj_idx_mapping.get(o, None)
                if s is not None and o is not None:
                    triples.append([s, p, o])
        
        in_image_idx = self.vocab['pred_name_to_idx']['__in_image__']
        for i in range(num_objects - 1):
            triples.append([i, in_image_idx, num_objects - 1])
        
        triples = torch.LongTensor(triples)
        return image, objs, boxes, triples

def vg_collate_fn(batch):
    """
    Collate function to be used when wrapping a VgSceneGraphDataset in a
    DataLoader. Returns a tuple of the following:

    - imgs: FloatTensor of shape (N, C, H, W)
    - objs: LongTensor of shape (O,) giving categories for all objects
    - boxes: FloatTensor of shape (O, 4) giving boxes for all objects
    - triples: FloatTensor of shape (T, 3) giving all triples, where
        triples[t] = [i, p, j] means that [objs[i], p, objs[j]] is a triple
    - obj_to_img: LongTensor of shape (O,) mapping objects to images;
        obj_to_img[i] = n means that objs[i] belongs to imgs[n]
    - triple_to_img: LongTensor of shape (T,) mapping triples to images;
        triple_to_img[t] = n means that triples[t] belongs to imgs[n].
    """
    all_imgs, all_objs, all_boxes, all_triples = [], [], [], []
    all_obj_to_img, all_triple_to_img = [], []
    obj_offset = 0
    for i, (img, objs, boxes, triples) in enumerate(batch):
        all_imgs.append(img[None, ...])     # add a new axis at dim 0
        num_objects, num_triples = objs.size(0), triples.size(0)
        all_objs.append(objs)
        all_boxes.append(boxes)
        triples = triples.clone()
        triples[:, 0] += obj_offset
        triples[:, 2] += obj_offset
        all_triples.append(triples)

        all_obj_to_img.append(torch.LongTensor(num_objects).fill_(i))
        all_triple_to_img.append(torch.LongTensor(num_triples).fill_(i))
        obj_offset += num_objects

    all_imgs = torch.cat(all_imgs)
    all_objs = torch.cat(all_objs)
    all_boxes = torch.cat(all_boxes)
    all_triples = torch.cat(all_triples)
    all_obj_to_img = torch.cat(all_obj_to_img)
    all_triple_to_img = torch.cat(all_triple_to_img)

    out = (all_imgs, all_objs, all_boxes, all_triples, all_obj_to_img, all_triple_to_img)
    return out

def get_vg_data_loader(data_root, vocab, batch_size=8, image_size=(256, 256), max_objects=10, include_relationships=True, use_orphaned_objects=True, split='train'):
    assert split in ['train', 'val', 'test']

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    image_transforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    dataset = VisualGenomeDataset(data_root, vocab, image_transforms, max_objects, include_relationships, use_orphaned_objects, split)
    if split == 'train':
        data_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=vg_collate_fn)
    else:
        data_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=vg_collate_fn)

    return data_loader


if __name__ == '__main__':
    vocab_file = os.path.join('/p300/visual_genome/v1.4', 'vocab.json')
    with open(vocab_file, 'r') as f:
        vocab = json.load(f)
    data_loader = get_vg_data_loader('/p300/visual_genome/v1.4', vocab, batch_size=4, image_size=(256, 256), max_objects=10, include_relationships=True, use_orphaned_objects=True, split='train')
    all_imgs, all_objs, all_boxes, all_triples, all_obj_to_img, all_triple_to_img = next(iter(data_loader))
    print(all_imgs.size())
    print(all_imgs)
    print(all_objs.size())
    print(all_objs)
    print(all_boxes.size())
    print(all_boxes)
    print(all_triples.size())
    print(all_triples)
    print(all_obj_to_img.size())
    print(all_obj_to_img)
    print(all_triple_to_img.size())
    print(all_triple_to_img)