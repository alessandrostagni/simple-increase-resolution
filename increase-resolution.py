import os
import sys

import torch
from PIL import Image
import numpy as np
from RealESRGAN import RealESRGAN


IMAGES_FOLDER = sys.argv[1]
RESOLUTION_MULTIPLIER = int(sys.argv[2])
OUTPUT_FOLDER = sys.argv[3]

def rescale():
    if not os.path.exists(f'{OUTPUT_FOLDER}-rescaled'):
        os.mkdir(f'{OUTPUT_FOLDER}-rescaled')
    for img_file in sorted(os.listdir(IMAGES_FOLDER)):
        if os.path.isfile(os.path.join(IMAGES_FOLDER, img_file)):
            path_to_input_image = f'{IMAGES_FOLDER}/{img_file}'
            input_image = Image.open(path_to_input_image)
            input_size = input_image.size
            path_to_output_image = f'{OUTPUT_FOLDER}/{img_file}'
            print(f'Rescaling {OUTPUT_FOLDER}-rescaled...')
            output_image = Image.open(path_to_output_image).convert('RGB')
            output_image_rescaled = output_image.resize(input_size)
            output_image_rescaled.save(f'{OUTPUT_FOLDER}-rescaled/{img_file}')
    print('Done.')
        
if not os.path.exists(OUTPUT_FOLDER):
    os.mkdir(OUTPUT_FOLDER)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
model = RealESRGAN(device, scale=RESOLUTION_MULTIPLIER)
model.load_weights(
    f'weights/RealESRGAN_x{RESOLUTION_MULTIPLIER}.pth', download=True
)

# Increase images resolution
for img_file in sorted(os.listdir(IMAGES_FOLDER)):
    if os.path.isfile(os.path.join(IMAGES_FOLDER, img_file)):
        path_to_image = f'{IMAGES_FOLDER}/{img_file}'
        image = Image.open(path_to_image).convert('RGB')
        print(f'Converting {img_file}...')
        sr_image = model.predict(image)
        sr_image.save(f'{OUTPUT_FOLDER}/{img_file}')
        print('Done.')

# Scale back if needed
if len(sys.argv) == 5 and sys.argv[4] == '--rescale':
    rescale()
