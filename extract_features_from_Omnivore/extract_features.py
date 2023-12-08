import os 

import torch
from download_videos_and_convert_to_tensor.rgb_to_tensor.image_to_tensor_h5 import load_images_from_hdf5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    model_name = "omnivore_swinB"
    model = torch.hub.load("facebookresearch/omnivore:main", model=model_name, force_reload=True)
    model = model.to(DEVICE)
    model = model.eval()
    return model 


def reshape_video_input(video_input):
    """
    The model expects inputs of shape: B x C x T x H x W
    so we'll need to add another dim
    """
    return video_input[None, ...] 

def remove_last_layer_from_model():
    """
    """
    print('needs to be implemented')

def main():
    model = load_model()
    remove_last_layer_from_model()
    for participant_directory in os.path.listdirectory('./tensor'):
        for img_batch in participant_directory:
            video_input = reshape_video_input(img_batch)
            with torch.no_grad():
                features = model(video_input.to(DEVICE), input_type="video")
    

