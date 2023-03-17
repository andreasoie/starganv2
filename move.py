

"""Objects


                        OPTICAL			        INFRARED
wooden boat + man		765XJBMN4YYBQ8.png		0R63B510F4BTZH.png
kayak + man		        HQAYVAWNQQRALN.png		D89GJVAPBMWY30.png
research vessel		    5969NVS5ZQQJG7.png		4GZOP19HFG5R3R.png
speed? boat + man		02VF5CQLCEL6ZP.png		RIGTBZGXE2Z2OL.png
white boat		        G85E2YGP46GBZU.png		Q7A9BI09LG997P.png
milliAmpere 1		    ZMEGO7TLO28PFX.png		PU1EFCIF8SD48G.png

"""

optical_images = [
    "765XJBMN4YYBQ8.png",
    "HQAYVAWNQQRALN.png",
    "5969NVS5ZQQJG7.png",
    "02VF5CQLCEL6ZP.png",
    "G85E2YGP46GBZU.png",
    "ZMEGO7TLO28PFX.png",
]
infra_images = [
    "0R63B510F4BTZH.png",
    "D89GJVAPBMWY30.png",
    "4GZOP19HFG5R3R.png",
    "RIGTBZGXE2Z2OL.png",
    "Q7A9BI09LG997P.png",
    "PU1EFCIF8SD48G.png",
]

import os

OUTDIR = "/home/andy/Dropbox/largefiles1/complete_dataset_processed/autoferry/study_cases_cherry"
os.makedirs(OUTDIR, exist_ok=True)

path_optical_in = "/home/andy/Dropbox/largefiles1/complete_dataset_processed/autoferry/testA"
path_infrared_in = "/home/andy/Dropbox/largefiles1/complete_dataset_processed/autoferry/testB"

path_optical_out = f"{OUTDIR}/optical"
path_infrared_out = f"{OUTDIR}/infrared"

os.makedirs(path_optical_out, exist_ok=True)
os.makedirs(path_infrared_out, exist_ok=True)

for i, optical_img in enumerate(optical_images):
    imgpath = f"{path_optical_in}/{optical_img}"
    newname = f"object_{i}.png"
    newpath = f"{path_optical_out}/{newname}"
    if not os.path.exists(imgpath):
        print(f"WARNING: {imgpath} does not exist")
    else:
        os.system(f"cp {imgpath} {newpath}")
    
for i, infrared_img in enumerate(infra_images):
    imgpath = f"{path_infrared_in}/{infrared_img}"
    newname = f"object_{i}.png"
    newpath = f"{path_infrared_out}/{newname}"
    if not os.path.exists(imgpath):
        print(f"WARNING: {imgpath} does not exist")
    else:
        os.system(f"cp {imgpath} {newpath}")
        
