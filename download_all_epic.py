import requests
import shutil

def download_file(url, file_name):

    with requests.get(url, stream=True) as r:
        with open(file_name, 'wb') as f:
            shutil.copyfileobj(r.raw, f)

if __name__ == '__main__':
    download_file('https://data.bris.ac.uk/datasets/tar/2g1n6qdydwa9u22shpxqzp0t8m.zip')
