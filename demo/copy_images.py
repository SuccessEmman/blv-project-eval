# copy_images.py
# Copies the 250 photos in our sample into demo/images so the website can show them.
# The full VizWiz download has tens of thousands of photos, we only need ours.
#
#   python copy_images.py "<folder that has the VizWiz photos>"
#
# The folder can be raw_data or any folder above it. The script looks through every
# subfolder (train, val, ...) for the 250 file names listed in sample_images.txt.
import os, shutil, sys

# we need exactly one thing from the command line: the folder to search
if len(sys.argv) != 2:
    sys.exit('usage: python copy_images.py "<folder with the VizWiz photos>"')
src = sys.argv[1]
# the folder this script is in (demo/), so it works from wherever you run it
here = os.path.dirname(os.path.abspath(__file__))
# the 250 file names we need, one per line in sample_images.txt
wanted = set(open(os.path.join(here, "sample_images.txt")).read().split())
# photos go into demo/images, which gets created if it isn't there yet
dest = os.path.join(here, "images")
os.makedirs(dest, exist_ok=True)

# go through every subfolder looking for our file names.
# copy2 keeps each photo's original file date. We stop early once all 250 are found.
found = 0
for root, _, files in os.walk(src):
    for f in files:
        if f in wanted:
            shutil.copy2(os.path.join(root, f), os.path.join(dest, f))
            wanted.discard(f)
            found += 1
    if not wanted:
        break
# say what happened, so we know if any photo is missing
print(f"copied {found} photos into {dest}")
if wanted:
    print(f"{len(wanted)} still missing, e.g. {sorted(wanted)[:5]}")
