import os 
from image_to_tensor_h5 import load_images_from_hdf5, batch_images_to_hdf5
import tarfile
import logging 

logging.basicConfig(filename='untar_logs.log', level=logging.INFO, format= '{asctime} - {levelname} - {message}')

def extract_tar(tar_file, destination_folder):
    try:
        with tarfile.open(tar_file, 'r') as tar:
            tar.extractall(destination_folder)
        print(f"Extraction complete. Files extracted to: {destination_folder}")
    except Exception as e:
        print(f"Error extracting the tar file: {e}")

def untar_directories(source_folder, destination_folder):
    for directory in os.listdir(source_folder):
        for file in os.listdir(f'{source_folder}/{directory}'):
            file_path = f'{source_folder}/{directory}/{file}'
            dst = f'{destination_folder}/{directory}/{file.split(".")[0]}'
            if not os.path.exists(dst):
                os.makedirs(dst)
            try:
                extract_tar(file_path, dst)
                logging.info(f"Untarred: {file_path} to {dst}")
            except Exception as e:
                logging.error(f"Failed to untar {file_path}. Error: {str(e)}")


def main():
    untar_directories('images', 'untared_images')
    for directory in os.listdir('untared_images'):
        for dir in os.listdir(f'untared_images/{directory}'):
            if not os.path.exists(f'tensor/{directory}/{dir}'):
                os.makedirs(f'tensor/{directory}/{dir}')
            batch_images_to_hdf5(f'untared_images/{directory}/{dir}', f'tensor/{directory}/{dir}', 'hdf_img', batch_size=50, stride= 25)


if __name__ == '__main__':
    main()


