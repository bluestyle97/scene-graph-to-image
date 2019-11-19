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
# Modified from https://github.com/google/sg2im/blob/master/sg2im/utils.py

import os
import time
import inspect
import subprocess
from contextlib import contextmanager
import torch


def int_tuple(s):
    return tuple(int(i) for i in s.split(','))

def float_tuple(s):
    return tuple(float(i) for i in s.split(','))

def str_tuple(s):
    return tuple(s.split(','))

def lineno():
    return inspect.currentframe().f_back.f_lineno

def get_gpu_memory():
    torch.cuda.synchronize()
    opts = [
        'nvidia-smi', '-q', '--gpu=' + str(0), '|', 'grep', '"Used GPU Memory"'
    ]
    cmd = str.join(' ', opts)
    ps = subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    output = ps.communicate()[0].decode('utf-8')
    output = output.split("\n")[1].split(":")
    consumed_mem = int(output[1].strip().split(" ")[0])
    return consumed_mem

def create_dirs(dirs):
    for dir_ in dirs:
        if not os.path.exists(dir_):
            os.makedirs(dir_)

def timeit_func(f):
    """ Decorator to time Any Function """

    def timed(*args, **kwargs):
        start_time = time.time()
        result = f(*args, **kwargs)
        end_time = time.time()
        total = end_time - start_time
        hours = total // 3600
        minutes = (total - 3600 * hours) // 60
        seconds = total % 60
        print("[-] {:s} : {:.2f} hours, {:.2f} minutes, {:.2f} seconds".format(f.__name__, hours, minutes, seconds))
        return result

    return timed

@contextmanager
def timeit_context(msg, should_time=True):
    if should_time:
        torch.cuda.synchronize()
        t0 = time.time()
    yield
    if should_time:
        torch.cuda.synchronize()
        t1 = time.time()
        duration = (t1 - t0) * 1000.0
        print('{:s}: {:.2f} ms' % (msg, duration))

class LossManager(object):
    def __init__(self, total_loss_name='total_loss'):
        self.total_loss_name = total_loss_name
        self.total_loss = None
        self.all_losses = {}

    def add_loss(self, loss, name, weight=1.0):
        if loss is None:
            return

        cur_loss = loss * weight
        if self.total_loss is not None:
            self.total_loss += cur_loss
        else:
            self.total_loss = cur_loss

        self.all_losses[name] = cur_loss.data.cpu().item()
        self.all_losses[self.total_loss_name] = self.total_loss.cpu().item()
    
    def get_total_loss(self):
        return self.total_loss
    
    def get_loss_item(self, name):
        if name in self.all_losses.keys():
            return self.all_losses[name]
        return None

    def items(self):
        return self.all_losses.items()