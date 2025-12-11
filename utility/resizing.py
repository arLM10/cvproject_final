from PIL import Image
import os

source = "/home/fdbdfg/VScode/cvproject/div2k_train"              # your original folder
dest = "/home/fdbdfg/VScode/cvproject/train512"            # new resized folder

os.makedirs(dest, exist_ok=True)

for fname in os.listdir(source):
    fpath = os.path.join(source, fname)

    # skip non-image files
    try:
        img = Image.open(fpath)
    except:
        print(f"Skipping non-image file: {fname}")
        continue

    # force RGB to avoid problems
    img = img.convert("RGB")

    # high-quality resize
    img = img.resize((512, 512), Image.LANCZOS)

    # save to new folder
    outpath = os.path.join(dest, fname)
    img.save(outpath, quality=95)

print("All images resized and saved to:", dest)
