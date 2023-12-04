import cv2
import numpy as np
import h5py
import os 

def images_to_hdf5(input_directory, output_file):
    '''
    Convert images in input_directory to HDF5 format.
    '''
    image_files = [f for f in os.listdir(input_directory) if f.endswith('.jpg')]
    
    with h5py.File(output_file, 'w') as hf:
        for image_file in image_files:
            image_path = os.path.join(input_directory, image_file)
            image = cv2.imread(image_path)
            hf.create_dataset(image_file, data=image)


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
            images_dict[key] = np.array(hf[key])
    return images_dict
