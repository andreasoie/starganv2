
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
    return sorted([os.path.join(rootdir, f) for rootdir, _, files in os.walk(dirpath) for f in files if f.lower().endswith(f".{suffix}")])

eo_images = find_files("/home/andy/Dropbox/largefiles1/autoferry_processed/autoferry/study_cases_cherry/optical", "png")
ir_images = find_files("/home/andy/Dropbox/largefiles1/autoferry_processed/autoferry/study_cases_cherry/infrared", "png")

mode = "reference"

step_images = []
resume_iters = [20_000, 40_000, 60_000, 80_000, 100_000]
for resume_iter in resume_iters:
    result_path = f"inferences/rgb2ir_{mode}/step_{resume_iter:06d}"
    result_img = find_files(result_path, "png")
    step_images.append(result_img)
    
all_images = []
all_images.append(eo_images)
for st in step_images:
    all_images.append(st)
all_images.append(ir_images)

transposed_images = []
import torch
import torchvision.utils as vutils

step_images = list(zip(*step_images))

for eo, genimgs, ir in zip(eo_images, step_images, ir_images):
    transposed_images.append(eo)
    for ge in genimgs:
        transposed_images.append(ge)
    transposed_images.append(ir)

tensor_list = []

accumulated_size = 0
for img_file in transposed_images:
    accumulated_size += 1
    img_data = cv2.imread(img_file)
    img_data = cv2.resize(img_data, (256, 256))
    img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
    img_data = torch.from_numpy(img_data).permute(2, 0, 1).float() / 255.0  # Convert to PyTorch tensor
    img_data = torch.nn.functional.interpolate(img_data.unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False).squeeze(0)
    tensor_list.append(img_data)

grid = vutils.make_grid(tensor_list, nrow=7, padding=0, pad_value=1)
grid = grid.numpy().transpose((1, 2, 0))
fig, ax = plt.subplots(figsize=(20, 15))
ax.imshow(grid)
ax.axis('off')
fig.tight_layout()
fig.subplots_adjust(wspace=0, hspace=0)
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
plt.savefig(f"starganv2_bs8_{mode}_cherrypicks.png", dpi=300, transparent=True)

# fig, axs = plt.subplots(2+len(resume_iters), 6)
# for i, row in enumerate(all_images):
#     for j, img_file in enumerate(row):
#         img_data = cv2.imread(img_file)
#         img_data = cv2.resize(img_data, (256, 256))
#         img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
#         axs[i][j].imshow(img_data)
#         axs[i][j].axis('off')
# fig.subplots_adjust(wspace=0, hspace=0)
# plt.tight_layout()
# plt.show()
# # plt.savefig("starganv2_bs8_cherrypicks.png", dpi=300, transparent=True)
