from collections import OrderedDict
import json
import math
import numpy as np
import os
import pandas as pd
import sys
import time

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.cuda.amp as amp
from torch.distributed.optim import ZeroRedundancyOptimizer
import torch.nn.parallel
import torchvision.transforms as transforms
import torchvision.transforms._transforms_video as transforms_video
from sklearn.metrics import confusion_matrix
import wandb

from download_videos_and_convert_to_tensor.untar_img_and_convert_to_h5.load_images_from_hdf5

from lavila.data import datasets
from lavila.data.video_transforms import Permute, SpatialCrop, TemporalCrop
from lavila.models import models
from lavila.models.tokenizer import (MyBertTokenizer, MyDistilBertTokenizer, MyGPT2Tokenizer, SimpleTokenizer)
from lavila.models.utils import inflate_positional_embeds
from lavila.utils import distributed as dist_utils
from lavila.utils.evaluation import accuracy, get_mean_accuracy
from lavila.utils.meter import AverageMeter, ProgressMeter
from lavila.utils.preprocess import generate_label_map
from lavila.utils.random import random_seed
from lavila.utils.scheduler import cosine_scheduler
from lavila.utils.evaluation_ek100cls import get_marginal_indexes, marginalize
from lavila.models.utils import inflate_positional_embeds
from lavila.models import models


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_TENSOR_DIR = 'tensor'
FEATURE_DIR = 'features_lavila'
BASE_MODEL =  'clip_openai_timesformer_base.narrator_rephraser.ep_0005.md5sum_d73a9c.pth'
FINETUNED_MODEL = 'clip_openai_timesformer_base.ft_ek100_cls.ep_0100.md5sum_4e3575.pth'

transform = transforms.Compose([
    Permute([1, 0, 3, 2]),
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms_video.NormalizeVideo(mean=[108.3272985, 116.7460125, 104.09373615000001], std=[68.5005327, 66.6321579, 70.32316305])
    ])

 to_tensor = transforms.PILToTensor()


def load_model(ckpt_path = BASE_MODEL):
    """
    loads pre-trained and then fine-tuned model,
    removes the last layer and return the fine-tuned model
    """
    old_args = ckpt['args']

    state_dict = OrderedDict()
    for k, v in ckpt['state_dict'].items():
        state_dict[k.replace('module.', '')] = v

    print(f'creating model: {old_args.model}")

    model = getattr(models, old_args.model)(
        pretrained=old_args.load_visual_pretrained,
        pretrained2d=old_args.load_visual_pretrained is not None,
        text_use_cls_token=old_args.use_cls_token,
        project_embed_dim=old_args.project_embed_dim,
        timesformer_gated_xattn=False,
        timesformer_freeze_space=False,
        num_frames= 16,
        drop_path_rate= 0.1,
    )
    if 'TIMESFORMER' in old_args.model or 'EGOVLP' in old_args.model:
        print('=> inflating PE in models due to different frame numbers')
        state_dict = inflate_positional_embeds(
        model.state_dict(), state_dict,
        num_frames= 16,
        load_temporal_fix='bilinear',
        )
    model.load_state_dict(state_dict, strict=True)
    print("=> loaded resume checkpoint '{}' (epoch {})".format(BASE_MODEL, ckpt['epoch']))

    model = models.VideoClassifierMultiHead(
            model.visual,
            dropout= 0.0,
            num_classes_list = [97, 300, 3806]
        )

    print("=> loading latest checkpoint '{}'".format(FINETUNED_MODEL))
    latest_checkpoint = torch.load(FINETUNED_MODEL, map_location='cpu')
    state_dict = OrderedDict()

    for k, v in latest_checkpoint['state_dict'].items():
        state_dict[k.replace('module.', '')] = v

    model.load_state_dict(state_dict)

    model.fc_cls = nn.ModuleList([torch.nn.Identity(), torch.nn.Identity(), torch.nn.Identity()])

    return model


def write_pickle(data, file_name):
    with open(file_name, "wb") as f:
        pickle.dump(data, f)


def main():     
    if not os.path.exists(FEATURE_DIR):
        os.makedirs(FEATURE_DIR)

    model = load_model()

    for participant in os.listdir(IMAGE_TENSOR_DIR):
        for video in os.listdir(f'{IMAGE_TENSOR_DIR}/{participant}'):

            if not os.path.exist(f'{FEATURE_DIR}/{participant}/{video}'):
                os.makedirs(f'{FEATURE_DIR}/{participant}/{video}')
        
            file_name = os.getcwd() + f'/{FEATURE_DIR}/{participant}/{video}.h5'

            with h5py.File(file_name, 'w') as file:
                for i, clip_batch in enumerate(os.listdir(f'{IMAGE_TENSOR_DIR}/{participant}/{video}')):
                    
                    clip_batch = load_images_from_hdf5(clip_batch)
                    clip_batch = ([to_tensor((clip_batch[t])) for t in clip_batch])
                    clip_batch = torch.stack(clip_batch).float()
                    clip_batch = transform(clip_batch).unsqueeze(0)
                    feature = model(clip_batch)
                    file.create_dataset(f'tensor_{i}', data= features)
     
def extract_features(file):
    extracted_features = []
    clip_batch = load_images_from_hdf5(file)
    clip_batch = ([to_tensor((clip_batch[t])) for t in clip_batch])
    length = len(clip_batch)
    while (length - 16) > start:
        start = 0
        end = 16
        stripe = 8 
        current_batch = torch.stack(clip_batch[start : end]).float()
        current_batch = transform(current_batch).unsqueeze(0)
        start += stripe 
        end += stripe 
        feature = model(current_batch)[0]
        extracted_features.append(feature)
