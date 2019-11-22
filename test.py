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
# Modified from https://github.com/google/sg2im/blob/master/scripts/train.py

import os
import json
import yaml
import math
import random
import argparse
import functools
from tqdm import tqdm
from collections import defaultdict
from easydict import EasyDict

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image

from datasets.visual_genome import get_vg_data_loader
from models.network import SceneGraph2Image
from models.discriminator import ImageDiscriminator, ObjectDiscriminator
from utils.metrics import jaccard
from utils.misc import int_tuple, float_tuple, str_tuple, timeit_func, timeit_context


def parse_args(yaml_file=None):
    if yaml_file is not None:
        with open(yaml_file, 'r') as f:
            args_dict = yaml.load(f)
            args = EasyDict(args_dict)
            return args

    parser = argparse.ArgumentParser('Inference with SceneGraph2Image Model')
    parser.add_argument('--exp_name', default='exp', type=str)
    parser.add_argument('--dataset', default='vg', choices=['vg', 'coco'])

    # Optimization hyperparameters
    parser.add_argument('--batch_size', default=32, type=int)

    # Dataset options common to both VG and COCO
    parser.add_argument('--image_size', default='128,128', type=int_tuple)
    parser.add_argument('--include_relationships', default=True, type=bool)

    # VG-specific options
    parser.add_argument('--vg_data_root', default='/p300/visual_genome/v1.4', type=str)
    parser.add_argument('--vg_max_objects', default=10, type=int)
    parser.add_argument('--vg_use_orphaned_objects', default=True, type=bool)

    # COCO-specific options
    parser.add_argument('--coco_data_root', default='/root/datasets/coco', type=str)
    parser.add_argument('--coco_instance_whitelist', default=None, type=str_tuple)
    parser.add_argument('--coco_stuff_whitelist', default=None, type=str_tuple)
    parser.add_argument('--coco_include_other', default=False, type=bool)
    parser.add_argument('--coco_min_object_size', default=0.02, type=float)
    parser.add_argument('--coco_min_objects_per_image', default=3, type=int)
    parser.add_argument('--coco_stuff_only', default=True, type=bool)

    # Output options
    parser.add_argument('--output_dir', default='outputs', type=str)
    parser.add_argument('--restore_checkpoint_from', default=None, type=str)

    args = parser.parse_args()
    return args

class Tester(object):
    def __init__(self, args):
        self.args = args
        print('Arguments for testing:')
        print(args)

        self.device = torch.device('cuda') fi torch.cuda.is_available() else torch.device('cpu')

        self.build_loader()     # build infer_loader
        self.build_model()      # build model
    
    @timeit_func
    def build_loader(self):
        assert self.args.dataset in ['coco', 'vg']

        if self.args.dataset == 'coco':
            pass
        else:
            vocab_file = os.path.join(self.args.vg_data_root, 'vocab.json')
            with open(vocab_file, 'r') as f:
                self.vocab = json.load(f)
            self.test_loader = get_vg_data_loader(self.args.vg_data_root, self.vocab, self.args.batch_size, self.args.image_size, self.args.vg_max_objects, self.args.include_relationships, self.args.vg_use_orphaned_objects, split='test')

    @timeit_func
    def build_model(self):
        checkpoint = torch.load(self.args.restore_checkpoint_from)
        self.iteration = checkpoint['iteration']

        self.model_kwargs = checkpoint['model_kwargs']
        self.model = SceneGraph2Image(self.vocab, self.model_kwargs)
        model_state_dict = {k[7:]: v for k, v in checkpoint['model_state'] if k.startswith('module.')}
        self.model.load_state_dict(model_state_dict)

    def test(self):
        self.model.to(self.device)
        self.model.eval()

        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.infer_loader)):
                batch = [tensor.to(self.device) for tensor in batch]
                masks = None
                if len(batch) == 6:     # no mask
                    imgs, objs, boxes, triples, obj_to_img, triple_to_img = batch
                    masks = None
                elif len(batch) == 7:
                    imgs, objs, boxes, masks, triples, obj_to_img, triple_to_img = batch
                else:
                    raise ValueError('Invalid batch format!')
                    
                imgs_pred, boxes_pred, masks_pred = self.model(objs, triples, obj_to_img)
                save_image(self.denorm(imgs_pred.cpu()), os.path.join(self.args.output_dir, self.args.exp_name, 'test_results', 'result-{:0>4d}.jpg'.format(i+1)), nrow=4), 
            
            print('[ - ] Test finished, save images to {:s}'.format(os.path.join(self.args.output_dir, self.args.exp_name, 'test_results')))
    
    def denorm(self, x):
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
        out = x.mul(std.view(1, 3, 1, 1)).add(mean.view(1, 3, 1, 1))
        return out.clamp_(0, 1)

def main():
    args = parse_args()
    create_dirs([os.path.join(self.args.output_dir, self.args.exp_name, 'test_results'])
    tester = Tester()
    tester.test()


if __name__ == '__main__':
    main()