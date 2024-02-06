import os 
import subprocess

def convert_fps(input_file, output_file):
    ffmpeg_cmd = [
        'ffmpeg',
        '-i', input_file,
        '-r', '30',
        output_file
    ]
    subprocess.run(ffmpeg_cmd)


def main():
    for participant in os.listdir('videos'):
        for video in os.listdir(f'videos/{participant}'):
            
            if not os.path.exists(f'videos_converted/{participant}'):
                os.makedirs(f'videos_converted/{participant}')

            convert_fps(f'videos/{participant}/{video}', f'videos_converted/{participant}/{video}')


if __name__ == '__main__':
    main()
    
