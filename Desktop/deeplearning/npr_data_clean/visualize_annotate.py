import numpy as np
import scipy.ndimage
from util import *
import json
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import glob

K_GUASSIAN = [0.0773, 0.2019, 0.2780, 0.2019, 0.0773,0.2019, 0.5273, 0.7261, 0.5273, 0.2019,0.2780, 0.7261, 1.0000, 0.7261, 0.2780,0.2019, 0.5273, 0.7261, 0.5273, 0.2019,0.0773, 0.2019, 0.2780, 0.2019, 0.0773]
K_GUASSIAN = np.array(K_GUASSIAN).reshape(5, 5)
IMG_SCALE = (64, 64)


def scale_to_255(img):
    img = img * 255
    img = img.round()
    img = img.astype("uint8")        
    return(img)

def get_heatmap(img, method = 'jet'):
    # the least effect i know: normalize to 0-1
    cmap = plt.get_cmap(method)
    rgba_img = cmap(img)
    rgb_img = rgba_img[:,:,:3]
    return rgb_img


def click_to_heatmap(click_obj):
    heatmap = np.zeros(IMG_SCALE)
    for x in click_obj["data"]:
        heatmap[int(round(x["y"]*IMG_SCALE[1])), int(round(x["x"]*IMG_SCALE[0]))] = 10
    heatmap = scipy.ndimage.convolve(heatmap, K_GUASSIAN, mode="constant")
    heatmap = np.array(heatmap)
    return heatmap

def visualize_dormer(heatmap):
    heat_cp = heatmap.copy()
    # heat_cp = heat_cp*10
    heat_cp = heat_cp/10
    heat_cp.reshape(heat_cp.shape[0], heat_cp.shape[1])
    heatmap = get_heatmap(heat_cp)
    heatmap = scale_to_255(heatmap) 
    heatmap = Image.fromarray(heatmap,'RGB')
    return heatmap


def process_h5():
    dir_h5 = "/Users/huayizeng/Desktop/proj1/NPR/data_further_label/l_flat_test"
    dir_json = "/Users/huayizeng/Desktop/proj1/NPR/data_further_label/dormer/l_flat_test-done20180130"
    files = list(glob.glob(os.path.join(dir_h5, "*.hdf5.h5")))
    for filepath in files:
        path_json = os.path.join(dir_json, filepath.split("/")[-1].split(".hdf5.h5")[0] + ".png.json")
        with open(path_json) as f:
            click_obj = json.load(f)
        rotation = float(click_obj["img_orientation"])
        img_d = load_h5py(filepath)
        img_d = Image.fromarray(img_d).resize(IMG_SCALE, resample=Image.BILINEAR)
        img_d = img_d.rotate(rotation, resample=Image.BILINEAR)
        img_d = pil_to_np(img_d)
        img_rgb = computeRGB(img_d)

        
        path_out = filepath.split(".")[0] + "_check.png"
        heatmap = click_to_heatmap(click_obj)
        heatmap = visualize_dormer(heatmap)
        out = Image.blend(img_rgb, heatmap, 0.5)
        out.save(path_out)


if __name__ == '__main__':
    process_h5()
