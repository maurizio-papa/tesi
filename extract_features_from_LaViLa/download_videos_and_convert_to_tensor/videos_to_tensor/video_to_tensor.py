import sys
import subprocess
import av
import numpy as np
import torch
from split_videos.split_videos_in_jpg import split_video_to_jpg
from image_to_tensor.image_to_tensor_h5 import images_to_hdf5
import numpy as np
import h5py
from PIL import Image
import io
import os 

def images_to_hdf5(input_directory, output_file):
    '''
    Convert images in input_directory to HDF5 format.
    '''
    image_files = [f for f in os.listdir(input_directory) if f.endswith('.jpg')]
    
    with h5py.File(output_file, 'w') as hf:
        for image_file in image_files:
            image_path = os.path.join(input_directory, image_file)
            with open(image_path, 'rb') as img_f:
                binary_data = img_f.read()
            binary_data_np = np.asarray(binary_data)
            hf.create_dataset(image_file, data=binary_data_np)


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


def batch_images_to_hdf5(input_directory, output_directory, output_prefix, batch_size=10, stride=5):
    '''
    Convert images in input_directory to HDF5 format with specified batch size and stride.
    Create separate HDF5 files for each batch.
    '''
    image_files = [f for f in os.listdir(input_directory) if f.endswith('.jpg')]
    total_images = len(image_files)
    num_batches = (total_images - batch_size) // stride + 1

    for i in range(num_batches):
        batch_images = image_files[i * stride: i * stride + batch_size]
        print(batch_images)

        output_file = f"{output_directory}/{output_prefix}_batch_{i + 1}.h5"
        with h5py.File(output_file, 'w') as hf:
            for j, image_file in enumerate(batch_images):
                image_path = os.path.join(input_directory, image_file)
                with open(image_path, 'rb') as img_f:
                    binary_data = img_f.read()
                binary_data_np = np.asarray(binary_data)
                hf.create_dataset(f'image_{j + 1}', data=binary_data_np)


def convert_videos_to_jpg(EPIC_KITCHENS_VIDEO_DIR, EPIC_KITCHENS_IMAGE_DIR):
    '''
    creates folders for extracting jpg from each participant's video
    and then extract jpgs
    '''
    for participant_dir in os.listdir(EPIC_KITCHENS_VIDEO_DIR):
        participant_image_dir = os.path.join(EPIC_KITCHENS_IMAGE_DIR, participant_dir)
        print(participant_image_dir)

        if not os.path.exists(participant_image_dir):
            os.makedirs(participant_image_dir)
        
        participant_video_dir = os.path.join(EPIC_KITCHENS_VIDEO_DIR, participant_dir)

        for video in os.listdir(participant_video_dir):
            video_name = video.split('.')[0]

            video_image_dir = f'{participant_image_dir}/{participant_dir}/{video_name}'
            if not os.path.exists(video_image_dir):
                os.makedirs(video_image_dir)

            split_video_to_jpg(os.path.join(participant_video_dir, video), video_image_dir)
            print(f'finished converting in jpg video {idx} of participant {participant_dir}')


def convert_jpg_to_tensor(EPIC_KITCHENS_VIDEO_DIR, EPIC_KITCHENS_IMAGE_DIR, EPIC_KITCHENS_TENSOR_DIR):
    '''
    creates folder for extracting tensor for each participant
    and then convert each jpg of each participant's video in an tensor of shape (t,h,w,c)
    '''

    for participant_dir in os.listdir(EPIC_KITCHENS_IMAGE_DIR):
        participant_tensor_dir = os.path.join(EPIC_KITCHENS_TENSOR_DIR, participant_dir)
        print(participant_tensor_dir)

        if not os.path.exists(participant_tensor_dir):
            os.makedirs(participant_tensor_dir)
            
     participant_image_dir = os.path.join(EPIC_KITCHENS_IMAGE_DIR, participant_dir)

    for video in os.listdir(participant_image_dir):
        video_name = video.split('.')[0]
        video_tensor_dir = f'{participant_tensor_dir}/{video_name}
        
            if not os.path.exists(video_tensor_dir):
                os.makedirs(video_tensor_dir)
        batch_images_to_hdf5(os.path.join(participant_image_dir, video_name), video_tensor_dir, output_prefix = img, batch_size= 16, stride= 2)        
        #images_to_hdf5(os.path.join(participant_image_dir, video), f'{video_tensor_dir}/{idx}.h5')
        print(f'finished converting in tensor video {video} of participant {participant_dir}')

