import os 

import torch
from omnivore.transforms import SpatialCrop, TemporalCrop
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
)

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision.transforms._transforms_video import NormalizeVideo
from download_videos_and_convert_to_tensor.rgb_to_tensor.image_to_tensor_h5 import load_images_from_hdf5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_TENSOR_DIR = 'tensor'

def load_model():
    model_name = "omnivore_swinB"
    model = torch.hub.load("facebookresearch/omnivore:main", model=model_name, force_reload=True)
    model = model.to(DEVICE)
    model = model.eval()
    return model 

def remove_last_layer_from_model():
    """
    """
    print('needs to be implemented')


def reshape_video_input(video_input):
    """
    The model expects inputs of shape: B x C x T x H x W
    so we'll need to add another dim
    """
    return video_input[None, ...] 


transform=T.Compose(
        [
        UniformTemporalSubsample(50), 
        T.Lambda(lambda x: x / 255.0),  
        ShortSideScale(size=224),
        NormalizeVideo(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        TemporalCrop(frames_per_clip=32, stride=40),
        SpatialCrop(crop_size=224, num_crops=3),
        ]
    )



def main():
    model = load_model()
    remove_last_layer_from_model()
    for participant_folder in os.listdir(IMAGE_TENSOR_DIR):
        for video in os.listdir(f'{IMAGE_TENSOR_DIR}/{participant_folder}'):
            for clip_batch in os.listdir(video):
                clip_batch = load_images_from_hdf5(clip_batch)
                clip_batch = transform(clip_batch)
                video_input = reshape_video_input(clip_batch)
                with torch.no_grad():
                    features = model(video_input.to(DEVICE), input_type="video")
    

