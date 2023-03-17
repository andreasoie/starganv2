
import os
from pprint import pprint
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np

# EO          20k     40k     ...     100k        IR
# kayak       kay20k  kay40k  ...     kay100k     kayak
# boat        bot20k  bot40k  ...     bot100k     boat
# ....        ....    ....    ...     ....        ....        

"""
    eo_boat         eo_kayak       ...         ....
    20k_boat        20k_kayak      ...         ....
    40k_boat        40k_kayak      ...         ....
    60k_boat        60k_kayak      ...         ....
    80k_boat        80k_kayak      ...         ....
    100k_boat       100k_kayak     ...         ....
    ir_boat         ir_kayak       ...         ....
"""

N_OBJECTS = 6

def find_files(dirpath: str, suffix: str) -> List[str]:
    files = [os.path.join(rootdir, f) for rootdir, _, files in os.walk(dirpath) for f in files if f.lower().endswith(f".{suffix}")]
    files.sort()
    return files

eo_images = find_files("/home/andy/Dropbox/largefiles1/complete_dataset_processed/autoferry/study_cases_cherry/optical", "png")
ir_images = find_files("/home/andy/Dropbox/largefiles1/complete_dataset_processed/autoferry/study_cases_cherry/infrared", "png")

step_images = []
resume_iters = [20_000, 40_000, 60_000, 80_000, 100_000]
for resume_iter in resume_iters:
    result_path = f"inferences/rgb2ir_gpu1/step_{resume_iter:06d}"
    result_img = find_files(result_path, "png")
    step_images.append(result_img)
    
all_images = []
all_images.append(eo_images)
for st in step_images:
    all_images.append(st)
all_images.append(ir_images)

fig, axs = plt.subplots(2+len(resume_iters), 6)
for i, row in enumerate(all_images):
    for j, img_file in enumerate(row):
        img_data = cv2.imread(img_file)
        img_data = cv2.resize(img_data, (256, 256))
        img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
        axs[i][j].imshow(img_data)
        axs[i][j].axis('off')
fig.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
plt.savefig("starganv2_bs8_cherrypicks.png", dpi=300, transparent=True)
