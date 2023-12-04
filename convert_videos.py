from download_videos_and_convert_to_tensor.video_to_tensor import convert_videos_to_jpg, convert_jpg_to_tensor

EPIC_KITCHENS_VIDEO_DIR = 'videos'
EPIC_KITCHENS_IMAGE_DIR = 'images'
EPIC_KITCHENS_TENSOR_DIR = 'tensor'


convert_videos_to_jpg(EPIC_KITCHENS_VIDEO_DIR, EPIC_KITCHENS_IMAGE_DIR)

convert_jpg_to_tensor(EPIC_KITCHENS_VIDEO_DIR, EPIC_KITCHENS_IMAGE_DIR, EPIC_KITCHENS_TENSOR_DIR)

