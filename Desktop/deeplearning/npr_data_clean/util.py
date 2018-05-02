import os
import re
import math
import h5py
import numpy as np
from PIL import Image
from PIL import ImageDraw 
from PIL import ImageFont
import cffi
import scipy.ndimage
import sys
sys.path.insert(0, "/Users/huayizeng/Desktop/opencv-new/lib/python2.7/site-packages")
import cv2
import matplotlib.pyplot as plt


# IMG_SCALE = (224, 224)
IMG_SCALE = (64, 64)
MAX_VAL = 31
MIN_VAL = 0

K_GUASSIAN = [0.0773, 0.2019, 0.2780, 0.2019, 0.0773,0.2019, 0.5273, 0.7261, 0.5273, 0.2019,0.2780, 0.7261, 1.0000, 0.7261, 0.2780,0.2019, 0.5273, 0.7261, 0.5273, 0.2019,0.0773, 0.2019, 0.2780, 0.2019, 0.0773]
K_GUASSIAN = np.array(K_GUASSIAN).reshape(5, 5)

COLORS = [(10, 10, 10), (0, 10, 10), (10, 0, 10), (10, 10, 0), (0, 0, 10), (0, 10, 0), (10, 0, 0)]
COLORS_HEX = ['#ff0000', '#00ff00', '#0000ff']

import string
import random
def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

"""
>>> id_generator()
'G5G74W'
>>> id_generator(3, "6793YUIO")
'Y3U'
"""

def get_list_img(training_only, testing_only):
    training_imgs_list = []
    for l in training_only:
        for root, dirs, files in os.walk(os.path.join("data/images", l)):
            for f in files:
                if f.endswith(".h5"):
                    training_imgs_list.append(os.path.join(root, f))
    testing_imgs_list = []
    if testing_only:
        for l in testing_only:
            for root, dirs, files in os.walk(os.path.join("data/images", l)):
                for f in files:
                    if f.endswith(".h5"):
                        testing_imgs_list.append(os.path.join(root, f))    
    return training_imgs_list, testing_imgs_list

def fill_contour(img, contour):
    img_cp = img.copy()
    raise NotImplementedError("!")

def show_img_d(img):
    assert len(img.shape) == 2 or img.shape[2] == 1 # todo: ?? maybe yes, channel of np image is the last dim
    if isinstance(img, np.ndarray):
        print 'show_img_d normalize'
        min_ = np.min(img[:])
        max_ = np.max(img[:])
        img = (img - min_) / (max_ - min_)
    show_img(img)

def np_to_pil(img_np):
    """ in-place transform """
    mode = None
    if (len(img_np.shape) == 3 and img_np.shape[0] == 1) or img_np.shape[0] == 3: 
        img_np = np.transpose(img_np,(1, 2, 0))
    print("img.dtype: ", img_np.dtype, "img.shape: ", img_np.shape)
    if len(img_np.shape) == 2 or img_np.shape[2] == 1: # todo: ??
        if len(img_np.shape) != 2: img_np = img_np[:, :, 0]
        if img_np.dtype == np.uint8:
            mode = 'L'
        if img_np.dtype == np.int16:
            mode = 'I;16'
        if img_np.dtype == np.int32:
            mode = 'I'
        elif img_np.dtype == np.float64 or img_np.dtype == np.float32:
            # can only be 0-1 normalized image, for depth image w/ unnormalized, refer to show_img__d
            img_np = img_np * 255
            img_np = img_np.round()
            img_np = img_np.astype("uint8")
            mode = 'L'
    else:
        if img_np.dtype == np.uint8:
            mode = 'RGB'
        else:
            if np.max(img_np) <= 1.0:
                img_np = img_np * 255
            img_np = img_np.round()
            img_np = img_np.astype(np.uint8)
            mode = 'RGB'   
    assert mode is not None, '{} is not supported'.format(img_np.dtype)
    img_pil = Image.fromarray(img_np, mode=mode)
    return img_pil

def show_img(img):
    """ Transform to PIL and then show """
    img_cp = img.copy()
    print("before show: ", np.max(img_cp), np.mean(img_cp), np.min(img_cp))
    if isinstance(img_cp, np.ndarray):
        img_cp = np_to_pil(img_cp)
    img_cp.show()
    print("after show: ", np.max(img_cp), np.mean(img_cp), np.min(img_cp))

def draw_rect_img(img, pt_tl, pt_br, color = "#ff0000"):
    img_cp = img.copy()
    if isinstance(img_cp, np.ndarray):
        img_cp = np_to_pil(img_cp)
    draw = ImageDraw.Draw(img_cp)
    draw.rectangle([(pt_tl[0], pt_tl[1]), (pt_br[0], pt_br[1])], outline=color) # yes, x and then y
    return img_cp

def mask_by_rect(img, pt_tl, pt_br):
    img_cp = img.copy()
    if not isinstance(img_cp, np.ndarray):
        raise NotImplementedError("!")
    img_cp = np.ma.array(img_cp)
    img_cp[pt_tl[1]:pt_br[1], pt_tl[0]:pt_br[0]] = np.ma.masked
    img_cp.mask = ~img_cp.mask
    img_cp.fill_value = 0
    mask = ~img_cp.mask
    img_cp = img_cp.filled()
    return img_cp, mask.astype('uint8')

def mask_by_u(img, ct):
    img_cp = img.copy()
    if not isinstance(img_cp, np.ndarray):
        raise NotImplementedError("!")
    ct = [[c] for c in ct]
    ct = np.asarray(ct, dtype=np.int32)
    img_contour = np.zeros((img.shape[0],img.shape[1],3), np.uint8)
    cv2.drawContours(img_contour, [ct], -1, (255,255,255), thickness=-1)
    img_b_2 = (img_contour[:,:,2] > 0).astype(np.int32)
    # show_img(img_b_2*255)
    img_cp = np.ma.array(img_cp)
    img_cp.mask = img_b_2
    img_cp.mask = ~img_cp.mask
    img_cp.fill_value = 0
    mask = ~img_cp.mask
    img_cp = img_cp.filled()
    # show_img(img_cp)
    # show_img(mask.astype('uint8')*255)
    # ma.mask is true for regions hope to be masked
    # mask = ~img_cp.mask is true(1) for un-masked
    return img_cp, mask.astype('uint8')


def find_jsons():
    flist = []
    for root, dirs, files in os.walk("data/jsons"):
        for f in files:
            flist.append(os.path.join(root, f))
    return flist
def convert_to_image_filenames(json_paths):
    json_filenames = list(map(os.path.basename, json_paths))
    json_filenames_without_ext = list(map(lambda x: x[0],map(os.path.splitext, json_filenames)))
    img_filenames = list(map(lambda x:x, json_filenames_without_ext))
    basepaths = list(map(lambda x:x[0], map(os.path.split, json_paths)))
    full_img_filename = zip(basepaths, json_filenames_without_ext, img_filenames)
    return list(full_img_filename)
def convert_to_basic_name(json_paths):
    json_filenames = list(map(os.path.basename, json_paths))
    json_filenames_without_ext = list(map(lambda x: x[0],map(os.path.splitext, json_filenames)))
    img_filenames = list(map(lambda x:x.replace("_image_normal_Lidar.png", ""), json_filenames_without_ext))
    img_filenames = list(map(lambda x:x+"_depth.h5", img_filenames))
    basepaths = list(map(lambda x:x[0], map(os.path.split, json_paths)))
    return list(zip(basepaths, img_filenames))
def remove_json_prefix(basepath):
    jsons = re.compile("^data/jsons")
    m1 = jsons.search(basepath)
    end = m1.end(0)
    return basepath[end:]

def get_normalized_coordinates(d, obj):
    (x1, y1) = (d["top_left_x"], d["top_left_y"])
    (x2, y2) = (d["bottom_right_x"], d["bottom_right_y"])
    img_width = obj["image_width"]
    img_height = obj["image_height"]
    x1 /= img_width
    x2 /= img_width
    y1 /= img_height
    y2 /= img_height
    return (x1, y1, x2, y2)
def load_h5py(filepath):
    f = h5py.File(filepath, "r")
    shape = f["shape"]
    matrix = f["matrix"]
    img = np.array(matrix)
    img = img.reshape(shape[0], shape[1])
    f.close()
    if np.any(np.isnan(img)):
        return None
    else:
        return img
#expect row, col, color
def standardize(img):
    new_img = np.copy(img)
    color_channels = new_img.shape[-1]
    for i in range(color_channels):
        mean = np.mean(new_img[:,:,i])
        std = np.std(new_img[:,:,i])
        new_img[:,:,i] -= mean
        new_img[:,:,i] /= std
    return new_img

def rotate_pnt(x, y, angle):
    x = x - 0.5
    y = 0.5 - y
    new_x = x * math.cos(math.radians((angle))) + y * math.sin(math.radians((angle)))
    new_y = -x * math.sin(math.radians((angle))) + y * math.cos(math.radians((angle)))
    new_x= new_x + 0.5
    new_y = 0.5 - new_y
    return (new_x, new_y)

def fixresult(obj, epsilon=0.05):
    back_obj = False
    if len(obj["data"]) == 3:
        back_obj = obj["data"][0]
        obj["data"] = obj["data"][1:]
    result_data = list(sorted(obj["data"], key=lambda x:x["top_left_y"]))
    # prev_data = result_data[0]
    sort_required = []
    if abs(obj["data"][0]["top_left_y"] - obj["data"][1]["top_left_y"]) < epsilon:
        result_data = list(sorted(obj["data"], key=lambda x:x["top_left_x"]))
    if back_obj:
        result_data.insert(0, back_obj)
    obj["data"] = result_data

def fixresult3(obj):
    # print(obj)
    def check_intersection(boxA, boxB):
        if boxA["top_left_x"] < boxB["bottom_right_x"] and boxA["bottom_right_x"] > boxB["top_left_x"] and boxA["top_left_y"] < boxB["bottom_right_y"] and boxA["bottom_right_y"] > boxB["top_left_y"]:
            return True
        else:
            return False
    inter_counter = 0
    new_data = []
    for boxA in obj["data"]:
        for boxB in obj["data"]:
            if boxB != boxA:
                if check_intersection(boxA, boxB):
                    inter_counter += 1
                    new_data.append(boxB)
        if inter_counter == 2:
            # print("inter")
            new_data.insert(0, boxA)
            break
        else:
            inter_counter = 0
            new_data = []
    if len(new_data) == 0:
        # print("other")
        min_dist = float('inf')
        min_box = None
        for boxA in obj["data"]:
            boxA_center_x = (boxA["top_left_x"] + boxA["bottom_right_x"])/2
            boxA_center_y = (boxA["top_left_y"] + boxA["bottom_right_y"])/2
            total_dist = 0
            for boxB in obj["data"]:
                if boxB != boxA:
                    boxB_center_x = (boxB["top_left_x"] + boxB["bottom_right_x"])/2
                    boxB_center_y = (boxB["top_left_y"] + boxB["bottom_right_y"])/2
                    AB_dist = math.sqrt((boxA_center_x - boxB_center_x)**2 + (boxA_center_y - boxB_center_y)**2)
                    total_dist += AB_dist
                    new_data.append(boxB)
            if total_dist < min_dist:
                min_dist = total_dist
                mix_box = boxA
                new_data.insert(0, boxA)
                break
            else:
                total_dist = 0
                new_data = []
    elif len(new_data) == 3:
        pass
    else:
        raise RuntimeError("in fixresult3: invalid new data size")
    obj["data"] = new_data

def add_img_w_mask(img, mask_bin, h5file, category, img_file):
    img_arr = np.array(img)
    if category + '_fname' not in h5file:
        strings = h5file.require_dataset(category+'_fname', (0, 1), 'S50', maxshape=(None, 1))
    else:
        strings = h5file[category+'_fname']
    strings.resize(strings.len()+1, axis=0)
    strings[strings.len()-1] = img_file.split('/')[-1].encode("ascii", "ignore")
    if category not in h5file:
        depth_img = h5file.require_dataset(category, (0, IMG_SCALE[0], IMG_SCALE[1], 5), dtype='uint8', maxshape=(None, IMG_SCALE[0], IMG_SCALE[1], 5))
    else:
        depth_img = h5file[category]
    img_arr[img_arr < 2] = 0
    ffi = cffi.FFI()
    ffi.cdef("bool NormalMapFromPointsSetPython(float *image_depth, unsigned char *image_normal, float *mask, int width, int height);")
    c = ffi.dlopen("./normalmap.so")
    depth_map = ffi.new("float[{}]".format(img_arr.shape[0] * img_arr.shape[1]), img_arr.flatten().tolist())
    normal_map = ffi.new("unsigned char[{}]".format(img_arr.shape[0] * img_arr.shape[1]*3), b"\0")
    mask = ffi.new("float[{}]".format(img_arr.shape[0] * img_arr.shape[1]), img_arr.flatten().tolist())
    c.NormalMapFromPointsSetPython(depth_map, normal_map, mask, img_arr.shape[1], img_arr.shape[0])
    new_normal_map = np.zeros((img_arr.shape[0], img_arr.shape[1], 3), dtype="uint8")
    for j in range(img_arr.shape[0]):
        for k in range(img_arr.shape[1]):
            for i in range(3):
                new_normal_map[j][k][2-i] = normal_map[j * img_arr.shape[1] * 3 + k * 3 + i]
    new_normal_map = np.array(Image.fromarray(new_normal_map).resize(IMG_SCALE, resample=Image.BILINEAR))
    img_min = img_arr.min()
    img_max = img_arr.max()
    if img_max > MAX_VAL:
        print("exceeded max_val", img_max)
    if img_min < MIN_VAL:
        print("underflow min_val:", img_min)
    ratio = 255/(MAX_VAL - MIN_VAL)
    mapped_img = (img_arr - MIN_VAL) * ratio
    round_img = mapped_img.round()
    img_arr = round_img.astype("uint8")
    img_arr = Image.fromarray(img_arr).resize(IMG_SCALE, resample=Image.BILINEAR)
    img_arr = np.array(img_arr).reshape((IMG_SCALE[0], IMG_SCALE[1], 1))
    img_arr[img_arr < 2] = 0
    # show_img(new_normal_map)
    mask_bin = Image.fromarray(mask_bin).resize(IMG_SCALE, resample=Image.BILINEAR)
    mask_bin = np.array(mask_bin).astype('uint8').reshape((IMG_SCALE[0], IMG_SCALE[1], 1))
    mask_bin[mask_bin>0] = 255
    final_img = np.concatenate((img_arr, new_normal_map, mask_bin), axis=-1)
    depth_img.resize(depth_img.len()+1, axis=0)
    depth_img[depth_img.len()-1] = final_img
    return final_img

def add_img_w_colored_box(img, h5file, category, boxes, img_file):
    img_arr = np.array(img)
    if category + '_fname' not in h5file:
        strings = h5file.require_dataset(category+'_fname', (0, 1), 'S50', maxshape=(None, 1))
    else:
        strings = h5file[category+'_fname']
    strings.resize(strings.len()+1, axis=0)
    strings[strings.len()-1] = img_file.split('/')[-1].encode("ascii", "ignore")
    if category not in h5file:
        #depth_img = h5file.require_dataset(category, (0, IMG_SCALE[0], IMG_SCALE[1], 1), dtype='float', maxshape=(None, IMG_SCALE[0], IMG_SCALE[1], 1))
        depth_img = h5file.require_dataset(category, (0, IMG_SCALE[0], IMG_SCALE[1], 4), dtype='uint8', maxshape=(None, IMG_SCALE[0], IMG_SCALE[1], 4))
    else:
        depth_img = h5file[category]
    img_arr[img_arr < 2] = 0
    ffi = cffi.FFI()
    ffi.cdef("bool NormalMapFromPointsSetPython(float *image_depth, unsigned char *image_normal, float *mask, int width, int height);")
    c = ffi.dlopen("./normalmap.so")
    depth_map = ffi.new("float[{}]".format(img_arr.shape[0] * img_arr.shape[1]), img_arr.flatten().tolist())
    normal_map = ffi.new("unsigned char[{}]".format(img_arr.shape[0] * img_arr.shape[1]*3), b"\0")
    mask = ffi.new("float[{}]".format(img_arr.shape[0] * img_arr.shape[1]), img_arr.flatten().tolist())
    c.NormalMapFromPointsSetPython(depth_map, normal_map, mask, img_arr.shape[1], img_arr.shape[0])
    new_normal_map = np.zeros((img_arr.shape[0], img_arr.shape[1], 3), dtype="uint8")
    for j in range(img_arr.shape[0]):
        for k in range(img_arr.shape[1]):
            for i in range(3):
                new_normal_map[j][k][2-i] = normal_map[j * img_arr.shape[1] * 3 + k * 3 + i]
    for ind, box in enumerate(boxes):
        pt_tl, pt_br = box
        new_normal_map = draw_rect_img(new_normal_map, pt_tl, pt_br, COLORS_HEX[ind])
    # show_img(new_normal_map)
    new_normal_map = np.array(new_normal_map.resize(IMG_SCALE, resample=Image.BILINEAR))
    img_min = img_arr.min()
    img_max = img_arr.max()
    if img_max > MAX_VAL:
        print("exceeded max_val", img_max)
    if img_min < MIN_VAL:
        print("underflow min_val:", img_min)
    ratio = 255/(MAX_VAL - MIN_VAL)
    mapped_img = (img_arr - MIN_VAL) * ratio
    round_img = mapped_img.round()
    img_arr = round_img.astype("uint8")
    img_arr = Image.fromarray(img_arr).resize(IMG_SCALE, resample=Image.BILINEAR)
    img_arr = np.array(img_arr).reshape((IMG_SCALE[0], IMG_SCALE[1], 1))
    img_arr[img_arr < 2] = 0
    # show_img(new_normal_map)
    final_img = np.concatenate((img_arr, new_normal_map), axis=-1)
    depth_img.resize(depth_img.len()+1, axis=0)
    depth_img[depth_img.len()-1] = final_img
    return final_img

def add_img(img, h5file, category):
    #img = img.resize(IMG_SCALE, resample=Image.BILINEAR)
    img_arr = np.array(img)
    if category not in h5file:
        #depth_img = h5file.require_dataset(category, (0, IMG_SCALE[0], IMG_SCALE[1], 1), dtype='float', maxshape=(None, IMG_SCALE[0], IMG_SCALE[1], 1))
        depth_img = h5file.require_dataset(category, (0, IMG_SCALE[0], IMG_SCALE[1], 4), dtype='uint8', maxshape=(None, IMG_SCALE[0], IMG_SCALE[1], 4))
    else:
        depth_img = h5file[category]
    img_arr[img_arr < 2] = 0
    ffi = cffi.FFI()
    ffi.cdef("bool NormalMapFromPointsSetPython(float *image_depth, unsigned char *image_normal, float *mask, int width, int height);")
    c = ffi.dlopen("./normalmap.so")
    depth_map = ffi.new("float[{}]".format(img_arr.shape[0] * img_arr.shape[1]), img_arr.flatten().tolist())
    normal_map = ffi.new("unsigned char[{}]".format(img_arr.shape[0] * img_arr.shape[1]*3), b"\0")
    mask = ffi.new("float[{}]".format(img_arr.shape[0] * img_arr.shape[1]), img_arr.flatten().tolist())
    c.NormalMapFromPointsSetPython(depth_map, normal_map, mask, img_arr.shape[1], img_arr.shape[0])
    new_normal_map = np.zeros((img_arr.shape[0], img_arr.shape[1], 3), dtype="uint8")
    for j in range(img_arr.shape[0]):
        for k in range(img_arr.shape[1]):
            for i in range(3):
                new_normal_map[j][k][2-i] = normal_map[j * img_arr.shape[1] * 3 + k * 3 + i]
    new_normal_map = np.array(Image.fromarray(new_normal_map).resize(IMG_SCALE, resample=Image.BILINEAR))

    img_min = img_arr.min()
    img_max = img_arr.max()
    if img_max > MAX_VAL:
        print("exceeded max_val", img_max)
    if img_min < MIN_VAL:
        print("underflow min_val:", img_min)
    ratio = 255/(MAX_VAL - MIN_VAL)
    mapped_img = (img_arr - MIN_VAL) * ratio
    round_img = mapped_img.round()
    img_arr = round_img.astype("uint8")
    img_arr = Image.fromarray(img_arr).resize(IMG_SCALE, resample=Image.BILINEAR)
    img_arr = np.array(img_arr).reshape((IMG_SCALE[0], IMG_SCALE[1], 1))
    img_arr[img_arr < 2] = 0
    # show_img(new_normal_map)
    final_img = np.concatenate((img_arr, new_normal_map), axis=-1)
    depth_img.resize(depth_img.len()+1, axis=0)
    depth_img[depth_img.len()-1] = final_img
    return final_img

def add_dormer_xy(click_obj, h5file, category):
    if category+"_outputs" not in h5file:
        outputs = h5file.require_dataset(category+"_outputs", (0, 20), dtype='float', maxshape=(None, 20))
    else:
        outputs = h5file[category+"_outputs"]
    outputs.resize(outputs.len()+1, axis=0)
    obj_data = [0 for i in range(20)]
    for ind, x in enumerate(click_obj["data"]):
        obj_data[2*ind] = x["x"]
        obj_data[2*ind+1] = x["y"]
    outputs[outputs.len()-1] = obj_data

def add_dormer_mask(click_obj, box, h5file, category):
    if category+"_heatmap" not in h5file:
        outputs = h5file.require_dataset(category+"_heatmap", (0, IMG_SCALE[0], IMG_SCALE[1], 1), dtype='float', maxshape=(None, IMG_SCALE[0], IMG_SCALE[1], 1))
    else:
        outputs = h5file[category+"_heatmap"]
    heatmap = np.zeros(IMG_SCALE)
    for x in click_obj["data"]:
        heatmap[int(round(x["y"]*IMG_SCALE[1])), int(round(x["x"]*IMG_SCALE[0]))] = 10
    heatmap = scipy.ndimage.convolve(heatmap, K_GUASSIAN, mode="constant")
    heatmap = np.array(heatmap)
    # mask the heatmap with box
    heatmap, mask = mask_by_rect(heatmap, box[0], box[1])
    outputs.resize(outputs.len()+1, axis=0)
    outputs[outputs.len()-1] = heatmap.reshape(IMG_SCALE[0], IMG_SCALE[1], 1)
    
def add_dormer_mask_u(click_obj, ct, h5file, category):
    if category+"_heatmap" not in h5file:
        outputs = h5file.require_dataset(category+"_heatmap", (0, IMG_SCALE[0], IMG_SCALE[1], 1), dtype='float', maxshape=(None, IMG_SCALE[0], IMG_SCALE[1], 1))
    else:
        outputs = h5file[category+"_heatmap"]
    heatmap = np.zeros(IMG_SCALE)
    for x in click_obj["data"]:
        heatmap[int(np.floor(x["y"]*IMG_SCALE[1])), int(np.floor(x["x"]*IMG_SCALE[0]))] = 10
    heatmap = scipy.ndimage.convolve(heatmap, K_GUASSIAN, mode="constant")
    heatmap = np.array(heatmap)
    # mask the heatmap with box
    heatmap, mask = mask_by_u(heatmap,ct)
    outputs.resize(outputs.len()+1, axis=0)
    outputs[outputs.len()-1] = heatmap.reshape(IMG_SCALE[0], IMG_SCALE[1], 1)

def add_heatmap(click_obj, h5file, category):
    if category+"_heatmap" not in h5file:
        outputs = h5file.require_dataset(category+"_heatmap", (0, IMG_SCALE[0], IMG_SCALE[1], 1), dtype='float', maxshape=(None, IMG_SCALE[0], IMG_SCALE[1], 1))
    else:
        outputs = h5file[category+"_heatmap"]
    outputs.resize(outputs.len()+1, axis=0)
    heatmap = np.zeros(IMG_SCALE)
    for x in click_obj["data"]:
        heatmap[int(round(x["y"]*IMG_SCALE[1])), int(round(x["x"]*IMG_SCALE[0]))] = 10
    heatmap = scipy.ndimage.convolve(heatmap, K_GUASSIAN, mode="constant")
    heatmap = np.array(heatmap)
    outputs[outputs.len()-1] = heatmap.reshape(IMG_SCALE[0], IMG_SCALE[1], 1)

def add_heatmap_footprint(obj, h5file, category):
    """ 
    Produce corner heatmap for footprint, at this moment we combine add_heatmap and add_json
    We exp on i-i shape temporarily. 
    """
    d_out = 4
    if category+"_heatmap" not in h5file:
        outputs = h5file.require_dataset(category+"_heatmap", (0, IMG_SCALE[0], IMG_SCALE[1], d_out), dtype='float', maxshape=(None, IMG_SCALE[0], IMG_SCALE[1], d_out))
    else:
        outputs = h5file[category+"_heatmap"]
    heatmaps = np.zeros((IMG_SCALE[0], IMG_SCALE[1], d_out))
    heatmap_l = [np.zeros(IMG_SCALE) for i in range(d_out)]
    obj_data = obj["data"]
    for ind in range(len(obj_data)):
        y_spike_tl, x_spike_tl = int(round(obj_data[ind]["top_left_y"]*IMG_SCALE[1])), int(round(obj_data[ind]["top_left_x"]*IMG_SCALE[0]))
        y_spike_br, x_spike_br = int(round(obj_data[ind]["bottom_right_y"]*IMG_SCALE[1])), int(round(obj_data[ind]["bottom_right_x"]*IMG_SCALE[0]))
        heatmap_l[0][y_spike_tl, x_spike_tl] = 10
        heatmap_l[1][y_spike_br, x_spike_br] = 10
        heatmap_l[2][y_spike_tl, x_spike_br] = 10
        heatmap_l[3][y_spike_br, x_spike_tl] = 10
    for i in range(d_out):
        heatmap_l[i] = scipy.ndimage.convolve(heatmap_l[i], K_GUASSIAN, mode="constant")
        heatmap_l[i] = np.array(heatmap_l[i])
        heatmaps[:,:,i] = heatmap_l[i]
        # show_img_d(heatmap_l[i])
    outputs.resize(outputs.len()+1, axis=0)
    outputs[outputs.len()-1] = heatmaps
    # show_img_d(heatmap2)

def add_heatmap_flat(obj, h5file, category):
    if category+"_heatmap" not in h5file:
        outputs = h5file.require_dataset(category+"_heatmap", (0, IMG_SCALE[0], IMG_SCALE[1], 1), dtype='float', maxshape=(None, IMG_SCALE[0], IMG_SCALE[1], 1))
    else:
        outputs = h5file[category+"_heatmap"]
    heatmap = np.zeros((IMG_SCALE[0], IMG_SCALE[1]))
    obj_data = obj["data"]
    # print ("obj: ", obj)
    # print ("obj_data: ", obj_data) 
    color = (10, 10, 10)
    if "new" in obj:
        for c in obj_data:
            contours = np.array([ [int(round(p["x"]*IMG_SCALE[0])), int(round(p["y"]*IMG_SCALE[1]))] for p in c])
            cv2.fillPoly(heatmap, pts =[contours], color=color)
    else: 
        if len(obj_data) != 4 and len(obj_data) != 6 and len(obj_data) != 8 \
        and len(obj_data) != 5 and len(obj_data) != 0 and len(obj_data) != 1 and len(obj_data) != 9:
            print len(obj_data)
            return False, None        
        if len(obj_data) == 4 or len(obj_data) == 6:
            contours = np.array([ [int(round(p["x"]*IMG_SCALE[0])), int(round(p["y"]*IMG_SCALE[1]))] for p in obj_data])
            cv2.fillPoly(heatmap, pts =[contours], color=color)
        if len(obj_data) == 5:
            contours = np.array([ [int(round(p["x"]*IMG_SCALE[0])), int(round(p["y"]*IMG_SCALE[1]))] for p in obj_data[:4]])
            cv2.fillPoly(heatmap, pts =[contours], color=color)
        if len(obj_data) == 8 or len(obj_data) == 9:
            contours = np.array([ [int(round(p["x"]*IMG_SCALE[0])), int(round(p["y"]*IMG_SCALE[1]))] for p in obj_data[:4]])
            cv2.fillPoly(heatmap, pts =[contours], color=color)
            contours = np.array([ [int(round(p["x"]*IMG_SCALE[0])), int(round(p["y"]*IMG_SCALE[1]))] for p in obj_data[4:8]])
            cv2.fillPoly(heatmap, pts =[contours], color=color)
    outputs.resize(outputs.len()+1, axis=0)
    outputs[outputs.len()-1] = heatmap.reshape(IMG_SCALE[1], IMG_SCALE[0], 1)
    # exit(0)
    # show_img_d(heatmap)
    return True, heatmap

def find_inners_u(obj_data):
    offset = 0
    len_ = len(obj_data)
    for ind in range(len_):
        y1, x1 = int(round(obj_data[ind]["y"]*IMG_SCALE[1])), int(round(obj_data[ind]["x"]*IMG_SCALE[0]))
        y2, x2 = int(round(obj_data[(ind + 1)%len_]["y"]*IMG_SCALE[1])), int(round(obj_data[(ind+1)%len_]["x"]*IMG_SCALE[0]))
        y3, x3 = int(round(obj_data[(ind + 2)%len_]["y"]*IMG_SCALE[1])), int(round(obj_data[(ind+2)%len_]["x"]*IMG_SCALE[0]))
        y21, x21 = y2-y1, x2-x1
        y32, x32 = y3-y2, x3-x2
        sig = np.cross([x32, y32, 0], [x21, y21, 0])[2]
        if sig < 0:
            offset = ind + 2
            break
    obj_data = obj_data[offset:] + obj_data[:offset]
    return obj_data

def find_inners_l(obj_data):
    offset = 0
    len_ = len(obj_data)
    for ind in range(len_):
        y1, x1 = int(round(obj_data[ind]["y"]*IMG_SCALE[1])), int(round(obj_data[ind]["x"]*IMG_SCALE[0]))
        y2, x2 = int(round(obj_data[(ind + 1)%len_]["y"]*IMG_SCALE[1])), int(round(obj_data[(ind+1)%len_]["x"]*IMG_SCALE[0]))
        y3, x3 = int(round(obj_data[(ind + 2)%len_]["y"]*IMG_SCALE[1])), int(round(obj_data[(ind+2)%len_]["x"]*IMG_SCALE[0]))
        y21, x21 = y2-y1, x2-x1
        y32, x32 = y3-y2, x3-x2
        sig = np.cross([x32, y32, 0], [x21, y21, 0])[2]
        if sig < 0:
            offset = ind + 1
            break
    obj_data = obj_data[offset:] + obj_data[:offset]
    return obj_data

def add_heatmap_iii(obj, h5file, category):
    if category+"_heatmap" not in h5file:
        outputs = h5file.require_dataset(category+"_heatmap", (0, IMG_SCALE[0], IMG_SCALE[1], 4), dtype='float', maxshape=(None, IMG_SCALE[0], IMG_SCALE[1], 4))
    else:
        outputs = h5file[category+"_heatmap"]
    heatmaps = np.zeros((IMG_SCALE[0], IMG_SCALE[1], 4))
    data = obj["data"]
    # dict_data_0_unicode = {str(key):value for key,value in obj["data"][0].items()}

    # if "top_left_x" in dict_data_0_unicode:
    #     pass
    # elif not(len(data) == 12 or len(data) == 4):
    #     print "len(data): ", len(data)
    #     return False, None
    boxes = []
    dict_data_0_unicode = obj["data"]

    if not( "top_left_x" in dict_data_0_unicode or len(data) == 8 or len(data) == 12 or len(data) == 4):
        return False, None
    if "top_left_x" in dict_data_0_unicode:
        for ind, d in enumerate(data):
            (x1, y1, x2, y2) = get_normalized_coordinates(d, obj)
            pt_tl = [x1*IMG_SCALE[1], y1*IMG_SCALE[0]]
            pt_tr = [x2*IMG_SCALE[1], y1*IMG_SCALE[0]]
            pt_br = [x2*IMG_SCALE[1], y2*IMG_SCALE[0]]
            pt_bl = [x1*IMG_SCALE[1], y2*IMG_SCALE[0]]
            boxes.append([pt_tl, pt_tr, pt_br, pt_bl])
    else:
        for ind in range(len(data)/4):
            tl_x = min(data[ind*4+0]["x"], data[ind*4+1]["x"], data[ind*4+2]["x"], data[ind*4+3]["x"])
            tl_y = min(data[ind*4+0]["y"], data[ind*4+1]["y"], data[ind*4+2]["y"], data[ind*4+3]["y"]) 
            br_x = max(data[ind*4+0]["x"], data[ind*4+1]["x"], data[ind*4+2]["x"], data[ind*4+3]["x"])
            br_y = max(data[ind*4+0]["y"], data[ind*4+1]["y"], data[ind*4+2]["y"], data[ind*4+3]["y"]) 
            pt_tl = [tl_x*IMG_SCALE[0],tl_y*IMG_SCALE[1]]
            pt_tr = [br_x*IMG_SCALE[0],tl_y*IMG_SCALE[1]]
            pt_br = [br_x*IMG_SCALE[0],br_y*IMG_SCALE[1]]
            pt_bl = [tl_x*IMG_SCALE[0],br_y*IMG_SCALE[1]]
            boxes.append([pt_tl, pt_tr, pt_br, pt_bl])
    heatmaps_3i = [np.zeros(IMG_SCALE) for i in range(4)]
    for ind, hm in enumerate(heatmaps_3i):
        for box in boxes:
            hm[box[ind][1], box[ind][0]] = 10
    for i in range(4):
        heatmaps_3i[i] = scipy.ndimage.convolve(heatmaps_3i[i], K_GUASSIAN, mode="constant")
        heatmaps_3i[i] = np.array(heatmaps_3i[i])
        heatmaps[:,:,i] = heatmaps_3i[i]
    outputs.resize(outputs.len()+1, axis=0)
    outputs[outputs.len()-1] = heatmaps
    return True, heatmaps


def add_heatmap_u(obj, h5file, category):
    if category+"_heatmap" not in h5file:
        outputs = h5file.require_dataset(category+"_heatmap", (0, IMG_SCALE[0], IMG_SCALE[1], 4), dtype='float', maxshape=(None, IMG_SCALE[0], IMG_SCALE[1], 4))
    else:
        outputs = h5file[category+"_heatmap"]
    heatmaps = np.zeros((IMG_SCALE[0], IMG_SCALE[1], 4))
    obj_data = obj["data"]
    if len(obj_data) != 8:
        print "len(obj_data): ", len(obj_data)
        return False, None   
    #assert(len(obj_data) == 8 and "len not correct")  
    obj_data = find_inners_u(obj_data)
    heatmap_l = [np.zeros(IMG_SCALE) for i in range(4)]
    for ind in range(4):
        y1, x1 = int(round(obj_data[ind]["y"]*IMG_SCALE[1])), int(round(obj_data[ind]["x"]*IMG_SCALE[0]))
        y2, x2 = int(round(obj_data[7 - ind]["y"]*IMG_SCALE[1])), int(round(obj_data[7 - ind]["x"]*IMG_SCALE[0]))
        heatmap_l[ind][y1, x1] = 10
        heatmap_l[ind][y2, x2] = 10
    for i in range(4):
        heatmap_l[i] = scipy.ndimage.convolve(heatmap_l[i], K_GUASSIAN, mode="constant")
        heatmap_l[i] = np.array(heatmap_l[i])
        heatmaps[:,:,i] = heatmap_l[i]
    outputs.resize(outputs.len()+1, axis=0)
    outputs[outputs.len()-1] = heatmaps
    return True, heatmaps

def add_heatmap_comp(obj, h5file, category):
    # if category+"_heatmap" not in h5file:
    #     outputs = h5file.require_dataset(category+"_heatmap", (0, IMG_SCALE[0], IMG_SCALE[1]), dtype='float', maxshape=(None, IMG_SCALE[0], IMG_SCALE[1]))
    # else:
    #     outputs = h5file[category+"_heatmap"]
    heatmap = np.zeros((IMG_SCALE[0], IMG_SCALE[1], 3))
    obj_data = obj["data"]
    if len(obj_data) %4 != 0:
        print "len(obj_data): ", len(obj_data)
        return False, None
    for ind in range(len(obj_data)/4):       
        contours = np.array([ [int(round(p["x"]*IMG_SCALE[0])), int(round(p["y"]*IMG_SCALE[1]))] for p in obj_data[4*ind:4*ind+4]])
        cv2.fillPoly(heatmap, pts =[contours], color=COLORS[ind%len(COLORS)])
    # outputs.resize(outputs.len()+1, axis=0)
    # outputs[outputs.len()-1] = heatmap
    return True, heatmap


def add_heatmap_l(obj, h5file, category):
    if category+"_heatmap" not in h5file:
        outputs = h5file.require_dataset(category+"_heatmap", (0, IMG_SCALE[0], IMG_SCALE[1], 4), dtype='float', maxshape=(None, IMG_SCALE[0], IMG_SCALE[1], 4))
    else:
        outputs = h5file[category+"_heatmap"]
    heatmaps = np.zeros((IMG_SCALE[0], IMG_SCALE[1], 4))
    obj_data = obj["data"]
    if len(obj_data) != 6:
        print "just throw, len(obj_data): ", len(obj_data)
        return False, None
    # otherwise, we could add this example   
    obj_data = find_inners_l(obj_data)
    heatmap_l = [np.zeros(IMG_SCALE) for i in range(4)]
    y0, x0 = int(round(obj_data[0]["y"]*IMG_SCALE[1])), int(round(obj_data[0]["x"]*IMG_SCALE[0]))
    heatmap_l[0][y0, x0] = 10
    for ind in range(1, 3):
        y1, x1 = int(round(obj_data[ind]["y"]*IMG_SCALE[1])), int(round(obj_data[ind]["x"]*IMG_SCALE[0]))
        y2, x2 = int(round(obj_data[6 - ind]["y"]*IMG_SCALE[1])), int(round(obj_data[6 - ind]["x"]*IMG_SCALE[0]))
        heatmap_l[ind][y1, x1] = 10
        heatmap_l[ind][y2, x2] = 10
    y3, x3 = int(round(obj_data[3]["y"]*IMG_SCALE[1])), int(round(obj_data[3]["x"]*IMG_SCALE[0]))
    heatmap_l[3][y3, x3] = 10

    for i in range(4):
        heatmap_l[i] = scipy.ndimage.convolve(heatmap_l[i], K_GUASSIAN, mode="constant")
        heatmap_l[i] = np.array(heatmap_l[i])
        heatmaps[:,:,i] = heatmap_l[i]
    outputs.resize(outputs.len()+1, axis=0)
    outputs[outputs.len()-1] = heatmaps
    return True, heatmaps

def draw_polygon_index(obj):
    obj_data = obj["data"]
    heatmap = np.zeros((IMG_SCALE[0] * 2, IMG_SCALE[1] * 2, 3))
    heatmap_pil = np_to_pil(heatmap)
    draw = ImageDraw.Draw(heatmap_pil)
    font = ImageFont.truetype("Arial.ttf", size=1)
    
    if "new" in obj:
        ind = 0
        for c in obj_data:
            for p in c:
                x, y = int(round(p["x"]*IMG_SCALE[0] * 2)), int(round(p["y"]*IMG_SCALE[1] * 2))
                draw.text((x, y),str(ind),(255,255,255))
                ind += 1

    else:
        for ind, p in enumerate(obj_data):
            x, y = int(round(p["x"]*IMG_SCALE[0] * 2)), int(round(p["y"]*IMG_SCALE[1] * 2))
            draw.text((x, y),str(ind),(255,255,255))
    # show_img(heatmap_pil)
    return heatmap_pil


def get_heatmap(img):
    # the least effect i know: normalize to 0-1
    cmap = plt.get_cmap('jet')
    rgba_img = cmap(img)
    rgb_img = rgba_img[:,:,:3]
    return rgb_img

def overlap_img_heatmap(method, obj, img, img_file):
    if os.path.exists("temp.h5"):
        os.unlink("temp.h5")
    h5file = h5py.File("temp.h5", 'w')
    res, img_heat = method(obj, h5file, "temp")
    print res
    if not res:
        heatmap = draw_polygon_index(obj)
    else:
        # if (len(img_heat.shape) == 3 and img_heat.shape[2] != 1):
        #     img_heat = img_heat[:,:,1]        
        # heatmap = get_heatmap(img_heat)
        print img_heat.shape
        heatmap = img_heat.copy()
        heatmap = heatmap * 255
        heatmap = heatmap.round()
        heatmap = heatmap.astype("uint8")        
        heatmap = Image.fromarray(heatmap)
        # heatmap = heatmap.resize((IMG_SCALE[0] * 2,IMG_SCALE[1] * 2), resample=Image.BILINEAR)
        heatmap = heatmap.resize((IMG_SCALE[0] * 2,IMG_SCALE[1] * 2), resample=Image.BILINEAR)
        # show_img(heatmap)
        # exit(0)

    img4d = add_img(img, h5file, "temp1")
    ori = img4d[:,:,1:4]
    ori = Image.fromarray(ori, 'RGB')
    ori = ori.resize((IMG_SCALE[0] * 2,IMG_SCALE[1] * 2), resample=Image.BILINEAR)
    out = Image.blend(ori, heatmap, 0.5)
    out.save(os.path.join("overlap_debug/" + img_file.split('/')[-2] + '_' + img_file.split('/')[-1] + ".png"))
    heatmap2 = draw_polygon_index(obj)
    out2 = Image.blend(ori, heatmap2, 0.5)
    out2.save(os.path.join("overlap_debug/" + img_file.split('/')[-2] + '_' + img_file.split('/')[-1] + "_pixel.png"))
    os.system("rm temp.h5")

def write_obj(filename, faces, vertices):
    with open(filename, "w+") as fout:
        for v in vertices:
            fout.write("v " + str(v[0]) + " " + str(v[1]) + " " + str(v[2]) + "\n")
        for f in faces:
            fout.write("f " + str(f[0]+1) + " " + str(f[1]+1) + " " + str(f[2]+1) + "\n")

def write_img_to_obj_color(filename, img_d):
    max_ = -1e10
    min_ = 1e10
    blue = [0, 0, 255]
    green = [0, 255, 0]
    yellow = [255, 255, 0]
    red = [255, 0, 0]
    for j in xrange(img_d.shape[0]):
        for i in xrange(img_d.shape[1]):
            if img_d[j][i] > 1e-8:
                max_ = max(max_, img_d[j][i])
                min_ = min(min_, img_d[j][i])

    diff = max_ - min_
    color = [0, 0, 0]
    with open(filename, "w+") as fout:
        for j in xrange(img_d.shape[0]):
            for i in xrange(img_d.shape[1]):
                if img_d[j][i] < min_ + diff * 0.25:
                    color = green
                elif img_d[j][i] < min_ + diff * 0.5:
                    color = blue
                elif img_d[j][i] < min_ + diff * 0.75:
                    color = yellow
                else:
                    color = red
                fout.write("v " + str(i) + " " + str(j) + " " + str(img_d[j][i]) + " " + 
                   str(color[0]) + " " + str(color[1]) + " " + str(color[2]) + " " + "\n") 

def read_obj_pos(filename):
    f = open(filename, "r") 
    if not f:
        print "Cannot open a file: " + filename
        return None

    vertices = []
    faces = []
    v_pattern = re.compile(r"v\s+(\d+[.]*\d*)\s+(\d+[.]*\d*)\s+(\d+[.]*\d*)")
    f_pattern = re.compile(r"f\s+(\d+[.]*\d*)\s+(\d+[.]*\d*)\s+(\d+[.]*\d*)")
    for line in f:
        if line[0] == "v":
            m = v_pattern.search(line)
            if m:
                vertices.append([float(m.group(1)), float(m.group(2)), float(m.group(3))])
        if line[0] == "f":
            m = f_pattern.search(line)
            if m:
                faces.append([int(m.group(1))-1, int(m.group(2))-1, int(m.group(3))-1])

    f.close()
    return (vertices, faces)

def read_obj(filename):
    f = open(filename, "r") 
    if not f:
        print "Cannot open a file: " + filename
        return None

    vertices = []
    faces = []
    v_pattern = re.compile(r"v\s+([+-]?\d+[.]*\d*)\s+([+-]?\d+[.]*\d*)\s+([+-]?\d+[.]*\d*)")
    f_pattern = re.compile(r"f\s+(\d+[.]*\d*)\s+(\d+[.]*\d*)\s+(\d+[.]*\d*)")
    for line in f:
        if line[0] == "v":
            m = v_pattern.search(line)
            if m:
                vertices.append([float(m.group(1)), float(m.group(2)), float(m.group(3))])
        if line[0] == "f":
            m = f_pattern.search(line)
            if m:
                faces.append([int(m.group(1))-1, int(m.group(2))-1, int(m.group(3))-1])

    f.close()
    return (vertices, faces)