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

from datasets.visual_genome import get_vg_data_loader
from models.network import SceneGraph2Image
from models.discriminator import ImageDiscriminator, ObjectDiscriminator
from utils.losses import gan_g_loss, gan_d_loss, bce_loss
from utils.bbox import crop_bbox_batch
from utils.metrics import jaccard
from utils.misc import int_tuple, float_tuple, str_tuple, timeit_func, timeit_context, create_dirs, LossManager

torch.backends.cudnn.benchmark = True


def parse_args(yaml_file=None):
    if yaml_file is not None:
        with open(yaml_file, 'r') as f:
            args_dict = yaml.load(f)
            args = EasyDict(args_dict)
            return args

    parser = argparse.ArgumentParser('Train SceneGraph2Image Model')
    parser.add_argument('--exp_name', default='exp', type=str)
    parser.add_argument('--dataset', default='vg', choices=['vg', 'coco'])

    # Optimization hyperparameters
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_iterations', default=1000000, type=int)
    parser.add_argument('--learning_rate', default=1e-4, type=float)

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

    # Generator options
    parser.add_argument('--embedding_dim', default=128, type=int)
    parser.add_argument('--gcn_out_dim', default=128, type=int)
    parser.add_argument('--gcn_hidden_dim', default=512, type=int)
    parser.add_argument('--gcn_num_layers', default=5, type=int)
    parser.add_argument('--gcn_pooling', default='avg', type=str)
    parser.add_argument('--mlp_activation', default='relu', type=str)
    parser.add_argument('--mlp_norm', default='none', type=str)
    parser.add_argument('--mask_size', default=None, type=int)    # Set this to None to use no masks
    parser.add_argument('--layout_noise_dim', default=32, type=int)
    parser.add_argument('--crn_out_dims', default='1024,512,256,128,64', type=int_tuple)
    parser.add_argument('--crn_activation', default='leakyrelu-0.2')
    parser.add_argument('--crn_norm', default='batch')

    # Loss weights
    parser.add_argument('--bbox_loss_weight', default=0.5, type=float)
    parser.add_argument('--mask_loss_weight', default=0, type=float)
    parser.add_argument('--pixel_loss_weight', default=0.5, type=float)
    parser.add_argument('--img_gan_loss_weight', default=0.1, type=float)
    parser.add_argument('--obj_gan_loss_weight', default=0.1, type=float)
    parser.add_argument('--obj_cls_loss_weight', default=0.1, type=float)

    # Generic discriminator options
    parser.add_argument('--gan_loss_type', default='lsgan')
    parser.add_argument('--d_activation', default='leakyrelu-0.2')
    parser.add_argument('--d_norm', default='batch')

    # Image discriminator
    parser.add_argument('--d_img_conv_dims', default='64,128,256', type=int_tuple)
    parser.add_argument('--d_img_fc_dim', default=1024, type=int)

    # Object discriminator
    parser.add_argument('--d_obj_conv_dims', default='64,128,256', type=int_tuple)
    parser.add_argument('--d_obj_fc_dim', default=1024, type=int)
    parser.add_argument('--d_obj_object_size', default='64,64', type=int_tuple)

    # Output options
    parser.add_argument('--print_period', default=20, type=int)
    parser.add_argument('--summary_period', default=100, type=int)
    parser.add_argument('--val_period', default=5000, type=int)
    parser.add_argument('--checkpoint_period', default=10000, type=int)
    parser.add_argument('--output_dir', default='outputs', type=str)
    parser.add_argument('--restore_checkpoint_from', default=None, type=str)

    args = parser.parse_args()
    return args

class Trainer(object):
    def __init__(self, args):
        self.args = args
        print('Arguments for training:')
        print(args)

        if not torch.cuda.is_available():
            raise EnvironmentError('CUDA is not available, training process will be terminated!')

        self.build_loader()     # build vocab, train_loader, val_loader
        self.build_model()      # build model, d_img, d_obj, optimizers
        self.writer = SummaryWriter(os.path.join(args.output_dir, args.exp_name, 'summaries'))
    
    @timeit_func
    def build_loader(self):
        assert self.args.dataset in ['coco', 'vg']

        if self.args.dataset == 'coco':
            pass
        else:
            vocab_file = os.path.join(self.args.vg_data_root, 'vocab.json')
            with open(vocab_file, 'r') as f:
                self.vocab = json.load(f)
            self.train_loader = get_vg_data_loader(self.args.vg_data_root, self.vocab, self.args.batch_size, self.args.image_size, self.args.vg_max_objects, self.args.include_relationships, self.args.vg_use_orphaned_objects, split='train')
            self.val_loader = get_vg_data_loader(self.args.vg_data_root, self.vocab, self.args.batch_size, self.args.image_size, self.args.vg_max_objects, self.args.include_relationships, self.args.vg_use_orphaned_objects, split='val')

    @timeit_func
    def build_model(self):
        if self.args.restore_checkpoint_from is not None:
            checkpoint = torch.load(os.path.join(self.os.path.join(self.args.output_dir, self.args.exp_name, 'checkpoints'), self.args.restore_checkpoint_from))
            self.iteration = checkpoint['iteration']
            self.epoch = checkpoint['epoch']

            self.model_kwargs = checkpoint['model_kwargs']
            self.model = SceneGraph2Image(self.vocab, self.model_kwargs)
            model_state_dict = {k[7:]: v for k, v in checkpoint['model_state'] if k.startswith('module.')}
            self.model.load_state_dict(model_state_dict)

            self.d_img_kwargs = checkpoint['d_img_kwargs']
            self.d_img = ImageDiscriminator(**self.d_img_kwargs)
            d_img_state_dict = {k[7:]: v for k, v in checkpoint['d_img_state'] if k.startswith('module.')}
            self.d_img.load_state_dict(d_img_state_dict)

            self.d_obj_kwargs = checkpoint['d_obj_kwargs']
            self.d_obj = ObjectDiscriminator(**self.d_obj_kwargs)
            d_obj_state_dict = {k[7:]: v for k, v in checkpoint['d_obj_state'] if k.startswith('module.')}
            self.d_obj.load_state_dict(d_obj_state_dict)
        else:
            self.iteration = 0
            self.epoch = 0

            self.model_kwargs = {
                'image_size': self.args.image_size, 
                'embedding_dim': self.args.embedding_dim, 
                'gcn_out_dim': self.args.gcn_out_dim, 
                'gcn_hidden_dim': self.args.gcn_hidden_dim, 
                'gcn_num_layers': self.args.gcn_num_layers, 
                'gcn_pooling': self.args.gcn_pooling, 
                'mlp_activation': self.args.mlp_activation, 
                'mlp_norm': self.args.mlp_norm, 
                'mask_size': self.args.mask_size, 
                'layout_noise_dim': self.args.layout_noise_dim, 
                'crn_out_dims': self.args.crn_out_dims, 
                'crn_activation': self.args.crn_activation, 
                'crn_norm': self.args.crn_norm
            }
            self.model = SceneGraph2Image(self.vocab, **self.model_kwargs)

            self.d_img_kwargs = {
                'conv_dims': self.args.d_img_conv_dims,
                'fc_dim': self.args.d_img_fc_dim,
                'image_size': self.args.image_size,
                'activation': self.args.d_activation,
                'norm': self.args.d_norm
            }
            self.d_img = ImageDiscriminator(**self.d_img_kwargs)

            self.d_obj_kwargs = {
                'conv_dims': self.args.d_obj_conv_dims,
                'fc_dim': self.args.d_obj_fc_dim,
                'object_size': self.args.d_obj_object_size,
                'activation': self.args.d_activation,
                'norm': self.args.d_norm
            }
            self.d_obj = ObjectDiscriminator(len(self.vocab['object_idx_to_name']), **self.d_obj_kwargs)

    @timeit_func
    def build_optimizer(self):
        self.model_optimizer = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        self.d_img_optimizer = optim.Adam(self.d_img.parameters(), lr=self.args.learning_rate)
        self.d_obj_optimizer = optim.Adam(self.d_obj.parameters(), lr=self.args.learning_rate)

        if self.args.restore_checkpoint_from is not None:
            checkpoint = torch.load(self.args.restore_checkpoint_from)

            model_optimizer_state = checkpoint['model_optimizer_state']
            self.model_optimizer.load_state_dict(model_optimizer_state)
            d_img_optimizer_state = checkpoint['d_img_optimizer_state']
            self.d_img_optimizer.load_state_dict(d_img_optimizer_state)
            d_obj_optimizer_state = checkpoint['d_obj_optimizer_state']
            self.d_obj_optimizer.load_state_dict(d_obj_optimizer_state)

    def save_checkpoint(self):
        checkpoint = {
            'iteration': self.iteration,
            'epoch': self.epoch,
            'model_state': self.model.state_dict(),
            'd_img_state': self.d_img.state_dict(),
            'd_obj_state': self.d_obj.state_dict(),
            'model_kwargs': self.model_kwargs,
            'd_img_kwargs': self.d_img_kwargs,
            'd_obj_kwargs': self.d_obj_kwargs,
            'model_optimizer_state': self.model_optimizer.state_dict(),
            'd_img_optimizer_state': self.d_img_optimizer.state_dict(),
            'd_obj_optimizer_state': self.d_obj_optimizer.state_dict()
        }
        checkpoint_name = 'checkpoint-{:d}.pth.tar'.format(self.iteration)
        torch.save(checkpoint, os.path.join(self.os.path.join(self.args.output_dir, self.args.exp_name, 'checkpoints'), checkpoint_name))

    def train(self):
        self.model.cuda()
        self.d_img.cuda()
        self.d_obj.cuda()

        self.build_optimizer()

        self.model.train()
        self.d_img.train()
        self.d_obj.train()
            
        while True:
            print('Starting epoch {:d} ...'.format(self.epoch + 1))

            for batch in tqdm(self.train_loader):
                if self.iteration == self.args.num_iterations:
                    break
                # =============== #
                # train generator #
                # =============== #
                batch = [tensor.cuda() for tensor in batch]
                if len(batch) == 6:     # no mask
                    imgs, objs, boxes, triples, obj_to_img, triple_to_img = batch
                    masks = None
                elif len(batch) == 7:
                    imgs, objs, boxes, masks, triples, obj_to_img, triple_to_img = batch
                else:
                    raise ValueError('Invalid batch format!')
                
                imgs_pred, boxes_pred, masks_pred = self.model(objs, triples, obj_to_img, boxes, masks)
                
                g_bbox_loss = F.mse_loss(boxes_pred, boxes, reduction='mean')
                g_mask_loss = F.binary_cross_entropy(masks_pred, masks, reduction='mean') if masks is not None else None
                g_pixel_loss = F.l1_loss(imgs_pred, imgs, reduction='mean')
                
                img_score_fake = self.d_img(imgs_pred)
                g_img_gan_loss = gan_g_loss(img_score_fake, gan_loss_type=self.args.gan_loss_type)

                objs_fake = crop_bbox_batch(imgs_pred, boxes, obj_to_img, self.args.d_obj_object_size[0], self.args.d_obj_object_size[1])
                obj_score_fake, obj_class_fake = self.d_obj(objs_fake)
                g_obj_gan_loss = gan_g_loss(obj_score_fake, gan_loss_type=self.args.gan_loss_type)
                g_obj_cls_loss = F.cross_entropy(obj_class_fake, objs)

                g_loss_manager = LossManager('G/total_loss')
                g_loss_manager.add_loss(g_bbox_loss, 'G/bbox_loss', weight=self.args.bbox_loss_weight)
                g_loss_manager.add_loss(g_mask_loss, 'G/mask_loss', weight=self.args.mask_loss_weight)
                g_loss_manager.add_loss(g_pixel_loss, 'G/pixel_loss', weight=self.args.pixel_loss_weight)
                g_loss_manager.add_loss(g_img_gan_loss, 'G/img_gan_loss', weight=self.args.img_gan_loss_weight)
                g_loss_manager.add_loss(g_obj_gan_loss, 'G/obj_gan_loss', weight=self.args.obj_gan_loss_weight)
                g_loss_manager.add_loss(g_obj_cls_loss, 'G/obj_cls_loss', weight=self.args.obj_cls_loss_weight)
                g_loss = g_loss_manager.get_total_loss()

                if not math.isfinite(g_loss.cpu().item()):
                    print('WARNING: Got loss = NaN, not backpropping')
                    continue

                self.model_optimizer.zero_grad()
                g_loss.backward()
                self.model_optimizer.step()

                # =================== #
                # train discriminator #
                # =================== #
                imgs_fake = imgs_pred.detach()
                img_score_fake = self.d_img(imgs_fake)
                img_score_real = self.d_img(imgs)
                d_img_gan_loss = gan_d_loss(img_score_real, img_score_fake, gan_loss_type=self.args.gan_loss_type)

                objs_fake = crop_bbox_batch(imgs_fake, boxes, obj_to_img, self.args.d_obj_object_size[0], self.args.d_obj_object_size[1])
                obj_score_fake, obj_class_fake = self.d_obj(objs_fake)
                objs_real = crop_bbox_batch(imgs, boxes, obj_to_img, self.args.d_obj_object_size[0], self.args.d_obj_object_size[1])
                obj_score_real, obj_class_real = self.d_obj(objs_real)
                d_obj_gan_loss = gan_d_loss(obj_score_real, obj_score_fake, gan_loss_type=self.args.gan_loss_type)
                d_obj_real_cls_loss = F.cross_entropy(obj_class_real, objs)
                d_obj_fake_cls_loss = F.cross_entropy(obj_class_fake, objs)

                d_img_loss_manager = LossManager('D_img/total_loss')
                d_img_loss_manager.add_loss(d_img_gan_loss, 'D_img/img_gan_loss', weight=self.args.img_gan_loss_weight)
                d_img_loss = d_img_loss_manager.get_total_loss()
                self.d_img_optimizer.zero_grad()
                d_img_loss.backward()
                self.d_img_optimizer.step()

                d_obj_loss_manager = LossManager('D_obj/total_loss')
                d_obj_loss_manager.add_loss(d_obj_gan_loss, 'D_obj/obj_gan_loss', weight=self.args.obj_gan_loss_weight)
                d_obj_loss_manager.add_loss(d_obj_real_cls_loss, 'D_obj/obj_real_cls_loss', weight=self.args.obj_cls_loss_weight)
                d_obj_loss_manager.add_loss(d_obj_fake_cls_loss, 'D_obj/obj_fake_cls_loss', weight=self.args.obj_cls_loss_weight)
                d_obj_loss = d_obj_loss_manager.get_total_loss()
                self.d_obj_optimizer.zero_grad()
                d_obj_loss.backward()
                self.d_obj_optimizer.step()

                self.iteration += 1
                if self.iteration % self.args.print_period == 0:
                    print('[iter {:d}] G total loss: {:f}, D_img total loss: {:f}, D_obj total loss: {:f}'.format(self.iteration, g_loss.item(), d_img_loss.item(), d_obj_loss.item()))

                if self.iteration % self.args.summary_period == 0:
                    for k, v in g_loss_manager.items():
                        self.writer.add_scalar(k, v, self.iteration)
                    for k, v in d_img_loss_manager.items():
                        self.writer.add_scalar(k, v, self.iteration)
                    for k, v in d_obj_loss_manager.items():
                        self.writer.add_scalar(k, v, self.iteration)
                
                if self.iteration % self.args.checkpoint_period == 0:
                    print('[iter {:d}] Sacing checkpoint to {:s} ...'.format(self.iteration, self.os.path.join(args.output_dir, args.exp_name, 'checkpoints')))
                    self.save_checkpoint()

                if self.iteration % self.args.val_period == 0:
                    print('[iter {:d}] Start running validation ...'.format(self.iteration))
                    self.val()
                    self.model.train()      # reset to training mode

            self.epoch += 1
        
    def val(self):
        self.model.eval()

        total_iou = 0
        total_boxes = 0
        with torch.no_grad():
            for batch in tqdm(self.val_loader):
                batch = [tensor.cuda() for tensor in batch]
                masks = None
                if len(batch) == 6:     # no mask
                    imgs, objs, boxes, triples, obj_to_img, triple_to_img = batch
                    masks = None
                elif len(batch) == 7:
                    imgs, objs, boxes, masks, triples, obj_to_img, triple_to_img = batch
                else:
                    raise ValueError('Invalid batch format!')
                    
                imgs_pred, boxes_pred, masks_pred = self.model(objs, triples, obj_to_img, boxes, masks)
                
                total_iou += jaccard(boxes_pred, boxes)
                total_boxes += boxes_pred.size(0)
            
            gt_images = imgs.cpu()
            imgs_pred, boxes_pred, masks_pred = self.model(objs, triples, obj_to_img, boxes, masks)
            pred_images = imgs_pred.cpu()
            if masks is not None:
                gt_masks = masks.cpu()
                pred_masks = masks_pred.cpu()
            avg_iou = total_iou / total_boxes
            
            print('[iter {:d}] Validation finished: avg_iou={:f}'.format(self.iteration, avg_iou))
            self.writer.add_images('val/gt_images', self.denorm(gt_images), self.iteration)
            self.writer.add_images('val/pred_images', self.denorm(pred_images), self.iteration)
            self.writer.add_scalar('val/avg_iou', avg_iou, self.iteration)
    
    def denorm(self, x):
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
        out = x.mul(std.view(1, 3, 1, 1)).add(mean.view(1, 3, 1, 1))
        return out.clamp_(0, 1)

def main():
    args = parse_args()
    create_dirs([os.path.join(args.output_dir, args.exp_name, 'summaries'), 
                os.path.join(args.output_dir, args.exp_name, 'checkpoints')])
    trainer = Trainer(args)
    trainer.train()


if __name__ == '__main__':
    main()