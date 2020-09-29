#!/usr/bin/env python3
# coding: utf-8

import os.path as osp
from collections import OrderedDict

import gdown
import torch
import tvm
import wget
from tvm import relay

wget.download('https://raw.githubusercontent.com/kosuke55/train_baiducnn/master/scripts/pytorch/BCNN.py')  # noqa
from BCNN import BCNN


def fix_model_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith('module.'):
            name = name[7:]  # remove 'module.' of dataparallel
        new_state_dict[name] = v
    return new_state_dict


current_dir = osp.dirname(osp.abspath(__file__))
data_dir = osp.join(current_dir, 'data')
pretrained_model = osp.join(data_dir, 'bestmodel.pt')
if not osp.exists(pretrained_model):
    print('Downloading %s' % pretrained_model)
    gdown.cached_download(
        'https://drive.google.com/uc?export=download&id=1RV5SHRohYc2Z-vyTNsDp69yw8x97hZRK',
        pretrained_model,
        md5='b1f211762b806e7d693ca62a534c4077')

input_shape = [1, 6, 672, 672]
in_channels = 6
n_class = 5
input_name = 'input'

model = BCNN(in_channels=in_channels, n_class=n_class).to('cuda')
state_dict = torch.load(pretrained_model)
model.load_state_dict(fix_model_state_dict(state_dict))
model = model.eval()

input_data = torch.randn(input_shape).cuda()
scripted_model = torch.jit.trace(
    model, input_data).eval()

shape_list = [(input_name, input_shape)]
mod, params = relay.frontend.from_pytorch(
    scripted_model, shape_list)

target = tvm.target.cuda()
target_host = 'llvm'
ctx = tvm.gpu()

with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target,
                      target_host=target_host, params=params)
with open(osp.join(data_dir, 'model_graph.json'), 'w') as fo:
    fo.write(lib.graph_json)

with open(osp.join(data_dir, 'model_graph.params'), 'wb') as fo:
    fo.write(tvm.relay.save_param_dict(lib.params))

lib.export_library(osp.join(data_dir, 'bcnn.so'))
