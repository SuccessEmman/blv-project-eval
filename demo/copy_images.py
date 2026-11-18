# copy_images.py - copies the 250 sample photos the website needs into images/
#
#   python copy_images.py "<folder that contains the VizWiz photos>"
#
# The folder can be raw_data itself, or any folder above it: the script searches all
# subfolders (train, val, ...) for the 250 file names listed in sample_images.txt.
import os, shutil, sys

if len(sys.argv) != 2:
    sys.exit('usage: python copy_images.py "<folder with the VizWiz photos>"')
src = sys.argv[1]
here = os.path.dirname(os.path.abspath(__file__))
wanted = set(open(os.path.join(here, "sample_images.txt")).read().split())
dest = os.path.join(here, "images")
os.makedirs(dest, exist_ok=True)

found = 0
for root, _, files in os.walk(src):
    for f in files:
        if f in wanted:
            shutil.copy2(os.path.join(root, f), os.path.join(dest, f))
            wanted.discard(f)
            found += 1
    if not wanted:
        break
print(f"copied {found} photos into {dest}")
if wanted:
    print(f"{len(wanted)} still missing, e.g. {sorted(wanted)[:5]}")
