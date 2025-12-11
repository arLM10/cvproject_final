import os
import shutil

URBAN_ROOT = "/home/fdbdfg/VScode/cvproject/data_test"             # path to your downloaded Urban100 dataset
DEST_TEST = "/home/fdbdfg/VScode/cvproject/test_data20"          # where your config expects test HR images

os.makedirs(DEST_TEST, exist_ok=True)

count = 0

# Urban100 has HR/LR pairs stored together inside images_SRF2 and images_SRF4
for subfolder in ["image_SRF_2", "image_SRF_3" , "image_SRF_4"]:
    folder_path = os.path.join(URBAN_ROOT, subfolder)

    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        continue

    for filename in os.listdir(folder_path):
        if filename.endswith("_HR.png"):     # select the HR files ONLY
            src = os.path.join(folder_path, filename)
            dst = os.path.join(DEST_TEST, filename)
            shutil.copy(src, dst)
            count += 1

print(f"✔ Done! Copied {count} HR test images to {DEST_TEST}")
