import os 

from image_to_tensor_h5 import load_images_from_hdf5, batch_images_to_hdf5
import tarfile

def extract_tar(tar_file, destination_folder):
    try:
        with tarfile.open(tar_file, 'r') as tar:
            tar.extractall(destination_folder)
        print(f"Extraction complete. Files extracted to: {destination_folder}")
    except Exception as e:
        print(f"Error extracting the tar file: {e}")



def untar_directories(list_of_dirs):
    for directory in list_of_dirs:
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                destination_folder = os.path.splitext(file_path)[0]  # Use the filename without extension as destination folder
                extract_tar(file_path, destination_folder)



def main():
    untar_directories('./images')
    batch_images_to_hdf5('./images', 'hdf_img', batch_size=50, stride= 25)


if __name__ == '__main__':
    main()