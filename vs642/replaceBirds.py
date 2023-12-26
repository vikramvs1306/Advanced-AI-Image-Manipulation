import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from utils import stable_diffusion as stable_diffusion


def replaceBirds():

    dir_name = './birdmask-images'
    paths = list(filter(lambda x: 'masked_' in x, os.listdir(dir_name)))
    for path in paths:
        path = f'{dir_name}/{path}'
        if (not os.path.isfile(path)):
            print(f'{path} - file not found')
            continue
        
        prompt = "replace the mask area in the image and with another kind of bird"
        raw_image = Image.open(path.replace('masked_', '')).convert("RGB").resize((512, 512))
        masked_image = Image.open(path).convert("RGB").resize((512, 512))
        bird_replaced = stable_diffusion(prompt=prompt, raw_image=raw_image, mask_image=masked_image)
        
        rm_dir_name = './birds-replaced'
        filename = path.split('/')[-1]
        mask_path = f'./{rm_dir_name}/{filename}'
        mask_path = Path(mask_path)
        if not mask_path.is_file():
            Path(rm_dir_name).mkdir(parents=True, exist_ok=True)
        new_filename = f"{filename.split('.jpeg')[0]}-birdsReplaced.jpeg"
        plt.imsave(f'./{rm_dir_name}/{new_filename}', np.array(bird_replaced))
        plt.imsave(f'./{rm_dir_name}/raw_image_of-{new_filename}', np.array(raw_image))

if __name__ == "__main__":
    replaceBirds()
    print('bird replaced images are saved in the dir: ./birds-replaced')