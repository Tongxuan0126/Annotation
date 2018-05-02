import numpy as py
from util import *
import glob
import cffi
from PIL import Image, ImageDraw
import h5py

def computeRGB(img_d):
    img_arr = img_d.copy()
    img_rgb = np.zeros((img_d.shape[0], img_d.shape[1], 3), dtype='uint8')
    ffi = cffi.FFI()
    # ffi.cdef("bool NormalMapFromPointsSetPython(float *image_depth, unsigned char *image_normal, float *mask, int width, int height);")
    # c = ffi.dlopen("./normalmap.so")
    ffi.cdef("bool cffi_ComputeNormalImgF(float *image_depth, double kPixelsBetwPts, int width, int height, unsigned char *image_normal);")
    c = ffi.dlopen("./normalmap_new.so")
    depth_map = ffi.new("float[{}]".format(img_arr.shape[0] * img_arr.shape[1]), img_arr.flatten().tolist())
    normal_map = ffi.new("unsigned char[{}]".format(img_arr.shape[0] * img_arr.shape[1]*3), img_rgb.flatten().tolist())
    # mask = ffi.new("float[{}]".format(img_arr.shape[0] * img_arr.shape[1]), img_arr.flatten().tolist())
    # c.NormalMapFromPointsSetPython(depth_map, normal_map, mask, img_arr.shape[1], img_arr.shape[0])
    c.cffi_ComputeNormalImgF(depth_map, 1, img_arr.shape[1], img_arr.shape[0], normal_map)
    new_normal_map = np.zeros((img_arr.shape[0], img_arr.shape[1], 3), dtype="uint8")
    sum_ = 0
    for j in range(img_arr.shape[0]):
        for k in range(img_arr.shape[1]):
            sum_ += normal_map[j * img_arr.shape[1] * 3 + k * 3 + 0]
            for i in range(3):
                new_normal_map[j][k][2-i] = normal_map[j * img_arr.shape[1] * 3 + k * 3 + i]
    new_normal_map = Image.fromarray(new_normal_map).resize((img_d.shape[0], img_d.shape[1]), resample=Image.BILINEAR)
    return new_normal_map

def process_h5():
    files = list(glob.glob(os.path.join(dir_h5, "*.h5")))
    for filepath in files:
        img_d = load_h5py(filepath)
        f_output = os.path.join(dir_input, filepath.split("/")[-1].split(".")[0] + ".png")
        img_n = computeRGB(img_d)
        img_n.save(f_output)

if __name__ == '__main__':
    dir_h5 = '/Users/huayizeng/Desktop/proj1/NPR/data_further_label/l_flat_test'
    dir_input = '/Users/huayizeng/Desktop/proj1/NPR/data_further_label/l_flat_test'
    process_h5()

