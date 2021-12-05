from histolab.slide import Slide
from histolab.tiler import GridTiler
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob

def mkdir_if_not_exist(inputdir):
    if not os.path.exists(inputdir):
        os.makedirs(inputdir)
    return inputdir


PATH_PANNET = "/kuacc/users/skoc21/dataset/pannet/wsi/"


extension = 'tiff'
lst_tiff = [f for f in glob.glob(PATH_PANNET + "/*." + extension)]
print(lst_tiff)

for tif_file in lst_tiff:
    fname_img = tif_file.split('/')[-1].split('.')[0]
    PATH_PROCESSED = "/kuacc/users/skoc21/dataset/pannet/wsi-tiles/" + fname_img

    pannet_slide = Slide(tif_file, processed_path=mkdir_if_not_exist(PATH_PROCESSED))
    
    print(f"Slide name: {pannet_slide.name}")
    print(f"Levels: {pannet_slide.levels}")
    print(f"Dimensions at level 0: {pannet_slide.dimensions}")
    print(f"Dimensions at level 1: {pannet_slide.level_dimensions(level=1)}")
    print(f"Dimensions at level 2: {pannet_slide.level_dimensions(level=2)}")

    
    grid_tiles_extractor = GridTiler(
       tile_size=(512, 512),
       level=1,
       check_tissue=True,
       tissue_percent=70,
       pixel_overlap=0, # default
       prefix="grid", # save tiles in the "grid" subdirectory of slide's processed_path
       suffix=".png"
    )
    grid_tiles_extractor.extract(pannet_slide)