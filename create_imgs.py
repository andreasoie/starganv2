import os
from typing import Tuple

import cv2
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

IMAGE_SIZE = 256

def get_images(path):
    images = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".png"):
                images.append(os.path.join(root, file))
    return images

def load_images_from_path(dirpath: str, mode: str) -> Tuple[list, int]:
    imgs = get_images(dirpath)
    imgs = [cv2.imread(img) for img in imgs]
    imgs = [cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE)) for img in imgs]
    if mode == "RGB":
        imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in imgs]
    imgs = [Image.fromarray(img) for img in imgs]
    return imgs, len(imgs)


plt.figure(figsize=(110, 10))

for idx in tqdm(range(1, 5), desc="Generating images", leave=False):
    # REAL
    real_image_optical, n_real_image_optical = load_images_from_path(f"data/test_{idx}/val/optical", mode="RGB")
    real_image_infrared, n_real_image_infrared = load_images_from_path(f"data/test_{idx}/val/infrared", mode="L")
    assert n_real_image_optical == 1, "There should be only one real image"
    assert n_real_image_infrared == 1, "There should be only one real image"
    
    # GENERATED
    test_images_infrared, n_test_images_infraed = load_images_from_path(f"test_{idx}/optical2infrared", "L")
    assert n_test_images_infraed == 10, "There should be 10 test images"
    
    N_ROWS = 1
    N_COLS = n_real_image_optical + n_real_image_infrared + n_test_images_infraed
    
    titles = ["Real Optical"] + [f"Fake Infrared {i}" for i in range(n_test_images_infraed)] + ["Real Optical"]
    all_images = real_image_optical + test_images_infrared + real_image_infrared
    
    for i, subtitle in enumerate(titles):
        plt.subplot(N_ROWS, N_COLS, i + 1)
        if i == 0:
            plt.imshow(all_images[i].convert("RGB"))
        else:
            plt.imshow(all_images[i], cmap='gray')
        plt.title(subtitle, fontsize=30)
        plt.axis('off')
    plt.savefig(f"test_{idx}.png")
   