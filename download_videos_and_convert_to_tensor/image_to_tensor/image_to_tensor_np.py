def images_to_compressed_npz(input_directory, output_directory):
    '''
    Read images in input_directory and save them in a compressed format in output_directory.
    Args:
        input_directory (str): Directory containing input JPEG image files.
        output_directory (str): Directory to save the compressed images in NumPy format.
    '''
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    image_files = [f for f in os.listdir(input_directory) if f.endswith('.jpg')]

    images = []
    for image_file in image_files:
        image_path = os.path.join(input_directory, image_file)
        image = cv2.imread(image_path)
        images.append(image)

    output_npz_path = os.path.join(output_directory, 'images_compressed.npz')
    np.savez_compressed(output_npz_path, *images)

# Example usage:
input_images_directory = 'input_images'
output_compressed_directory = 'output_compressed'

# Example usage:
input_images_directory = 'P101_JPG'
output_matrices_directory = 'p01_mat'

images_to_compressed_npz(input_images_directory, output_matrices_directory)