import os, random, shutil

source = "/home/fdbdfg/VScode/cvproject/data_test"   # folder with original images
dest = "/home/fdbdfg/VScode/cvproject/test_data20"  # new reduced folder
keep = 20  # how many images you want to keep

os.makedirs(dest, exist_ok=True)

files = os.listdir(source)
selected = random.sample(files, keep)

for f in selected:
    shutil.copy(os.path.join(source, f), os.path.join(dest, f))

print("Done! Selected", len(selected), "images.")
