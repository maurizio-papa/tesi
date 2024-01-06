import sys
import os
import csv
import json
import io
import h5py
import torch

import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np


from omnivore.transforms import SpatialCrop, TemporalCrop, DepthNorm

from typing import List


from PIL import Image
from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision.transforms._transforms_video import NormalizeVideo


from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
)

model_name = "omnivore_swinB_epic"
model = torch.hub.load("facebookresearch/omnivore:main", model= model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
model = model.to(device)
model = model.eval()


def load_images_from_hdf5(input_hdf5_file):
    '''
    Load images from HDF5 file.
    Args:
        input_hdf5_file (str): Input HDF5 file containing images.
    Returns:
        images_dict (dict): Dictionary containing image names as keys and corresponding NumPy arrays as values.
    '''
    images_dict = {}
    with h5py.File(input_hdf5_file, 'r') as hf:
        for key in hf.keys():
            images_dict[key] = Image.open(io.BytesIO(np.array(hf[key])))
    return images_dict

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/gdrive/MyDrive/progetto_tesi/tensor

"""# test"""

!wget https://dl.fbaipublicfiles.com/omnivore/epic_action_classes.csv
with open('epic_action_classes.csv', mode='r') as inp:
    reader = csv.reader(inp)
    epic_id_to_action = {idx: " ".join(rows) for idx, rows in enumerate(reader)}

transform=T.Compose(
        [
        UniformTemporalSubsample(50),
        T.Lambda(lambda x: x / 255.0),
        ShortSideScale(size=224),
        NormalizeVideo(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        TemporalCrop(frames_per_clip=50, stride= ),
        SpatialCrop(crop_size=224, num_crops=3),
        ]
    )

batch = load_images_from_hdf5('hdf_img_batch_.h5')
example = np.asarray([np.asarray(batch[t]) for t in batch]).reshape(3, 50, 256, 456)
example = torch.from_numpy(example)
test = transform(example)
test = np.stack(test, axis = 0)
print(f' after transform shape is: {test.shape}')
test = test[0][None, ...]
test = torch.from_numpy(test)

"""### Get model predictions"""

# Pass the input clip through the model
with torch.no_grad():
    prediction = model(test.to(device), input_type="video")

    # Get the predicted classes
    pred_classes = prediction.topk(k=5).indices

# Map the predicted classes to the label names
pred_class_names = [epic_id_to_action[int(i)] for i in pred_classes[0]]
print("Top 5 predicted actions: %s" % ", ".join(pred_class_names))