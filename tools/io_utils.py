import os
import json
import cv2
import numpy as np
import pylas
from osgeo import gdal

gdal.UseExceptions()

# read dsm prediction config file
def read_config(config_file):
    with open(config_file, "r") as f:
        config = json.load(f)
    return config

def mkdir_if_not_exist(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

# read dsm prediction specific format file
# # 3 -> view num
# # 0 path_to_file -> kv format
# # 1 ...
# # 2 ...
def read_from_txt(txt_file):
    with open(txt_file, "r") as f:
        text = f.read().splitlines()
    
    num = int(text[0])
    content = text[1:]
    if num != len(content):
        raise ValueError(f"read_from_txt: read {txt_file}, should get {num} lines, but get {len(content)} lines.")
    
    res = dict()
    for line in content:
        kv = line.split(" ", 1)
        res[int(kv[0])] = kv[1]
    
    return res

# read read dsm prediction specific format file
# # 1 -> pair num
# # 2 -> pair first part
# # 2 0 99.99 1 99.99 second part
def read_pair_from_txt(txt_file):
    with open(txt_file, "r") as f:
        text = f.read().splitlines()

    num = int(text[0])
    content = text[1:]
    if num != len(content) / 2:
        raise ValueError(f"read_pair_from_txt: read {txt_file}, should get {num} pair(s), but get {len(content) / 2} pair(s).")

    view_idx_list = []
    for i in range(num):
        ref_idx = [content[i * 2]]

        src_content = content[i * 2 + 1].split(" ")
        src_num = int(src_content[0])
        if src_num != len(src_content[1:]) / 2:
            raise ValueError(
                f"read_pair_from_txt: read {txt_file}, should get {src_num} source view(s), but get {len(src_content[1:]) / 2} source view(s)."
            )

        for j in range(src_num):
            src_idx = src_content[j * 2 + 1]
            ref_idx.append(src_idx)

        view_idx_list.append(ref_idx)

    return view_idx_list

# txt file to np array
def read_np_array_from_txt(txt_file):
    with open(txt_file, "r") as f:
        text = f.read().splitlines()

    res = np.array(text, float)
    return res

# get gdal's image size
def get_image_size(image_path):
    image = gdal.Open(image_path)
    if image is None:
        raise Exception(f"gdal open {image_path} failed")
    
    width = image.RasterXSize
    height = image.RasterYSize

    del image
    return width, height

# gdal read image(gray)
def read_image(image_path, x_start, y_start, x_size, y_size):
    data = gdal.Open(image_path)
    if data is None:
        raise Exception("GDAL RasterIO Error: Opening" + image_path + " failed!")
    
    data = data.ReadAsArray(x_start, y_start, x_size, y_size)

    # process to augment contrast
    if len(data.shape) == 2: # [H, W]
        small_thre = np.percentile(data.reshape(-1, 1), 2)
        data[data < small_thre] = small_thre
        
        big_thre = np.percentile(data.reshape(-1, 1), 98)
        data[data > big_thre] = big_thre

        image = 255 * (data - small_thre) / (big_thre - small_thre)
        image = image.astype(np.uint8)
        res = np.stack([image, image, image], axis = 0) # gray to rgb, fit network inputs
    else:
        res = data

    del data
    return res # [3, H, W]

def save_image(path, image):
    cv2.imwrite(path, image)

def save_rpc(path, rpc):
    addition0 = ['LINE_OFF:', 'SAMP_OFF:', 'LAT_OFF:', 'LON_OFF:', 'HEIGHT_OFF:', 'LINE_SCALE:', 'SAMP_SCALE:',
                    'LAT_SCALE:', 'LON_SCALE:', 'HEIGHT_SCALE:', 'LINE_NUM_COEFF_1:', 'LINE_NUM_COEFF_2:',
                    'LINE_NUM_COEFF_3:', 'LINE_NUM_COEFF_4:', 'LINE_NUM_COEFF_5:', 'LINE_NUM_COEFF_6:',
                    'LINE_NUM_COEFF_7:', 'LINE_NUM_COEFF_8:', 'LINE_NUM_COEFF_9:', 'LINE_NUM_COEFF_10:',
                    'LINE_NUM_COEFF_11:', 'LINE_NUM_COEFF_12:', 'LINE_NUM_COEFF_13:', 'LINE_NUM_COEFF_14:',
                    'LINE_NUM_COEFF_15:', 'LINE_NUM_COEFF_16:', 'LINE_NUM_COEFF_17:', 'LINE_NUM_COEFF_18:',
                    'LINE_NUM_COEFF_19:', 'LINE_NUM_COEFF_20:', 'LINE_DEN_COEFF_1:', 'LINE_DEN_COEFF_2:',
                    'LINE_DEN_COEFF_3:', 'LINE_DEN_COEFF_4:', 'LINE_DEN_COEFF_5:', 'LINE_DEN_COEFF_6:',
                    'LINE_DEN_COEFF_7:', 'LINE_DEN_COEFF_8:', 'LINE_DEN_COEFF_9:', 'LINE_DEN_COEFF_10:',
                    'LINE_DEN_COEFF_11:', 'LINE_DEN_COEFF_12:', 'LINE_DEN_COEFF_13:', 'LINE_DEN_COEFF_14:',
                    'LINE_DEN_COEFF_15:', 'LINE_DEN_COEFF_16:', 'LINE_DEN_COEFF_17:', 'LINE_DEN_COEFF_18:',
                    'LINE_DEN_COEFF_19:', 'LINE_DEN_COEFF_20:', 'SAMP_NUM_COEFF_1:', 'SAMP_NUM_COEFF_2:',
                    'SAMP_NUM_COEFF_3:', 'SAMP_NUM_COEFF_4:', 'SAMP_NUM_COEFF_5:', 'SAMP_NUM_COEFF_6:',
                    'SAMP_NUM_COEFF_7:', 'SAMP_NUM_COEFF_8:', 'SAMP_NUM_COEFF_9:', 'SAMP_NUM_COEFF_10:',
                    'SAMP_NUM_COEFF_11:', 'SAMP_NUM_COEFF_12:', 'SAMP_NUM_COEFF_13:', 'SAMP_NUM_COEFF_14:',
                    'SAMP_NUM_COEFF_15:', 'SAMP_NUM_COEFF_16:', 'SAMP_NUM_COEFF_17:', 'SAMP_NUM_COEFF_18:',
                    'SAMP_NUM_COEFF_19:', 'SAMP_NUM_COEFF_20:', 'SAMP_DEN_COEFF_1:', 'SAMP_DEN_COEFF_2:',
                    'SAMP_DEN_COEFF_3:', 'SAMP_DEN_COEFF_4:', 'SAMP_DEN_COEFF_5:', 'SAMP_DEN_COEFF_6:',
                    'SAMP_DEN_COEFF_7:', 'SAMP_DEN_COEFF_8:', 'SAMP_DEN_COEFF_9:', 'SAMP_DEN_COEFF_10:',
                    'SAMP_DEN_COEFF_11:', 'SAMP_DEN_COEFF_12:', 'SAMP_DEN_COEFF_13:', 'SAMP_DEN_COEFF_14:',
                    'SAMP_DEN_COEFF_15:', 'SAMP_DEN_COEFF_16:', 'SAMP_DEN_COEFF_17:', 'SAMP_DEN_COEFF_18:',
                    'SAMP_DEN_COEFF_19:', 'SAMP_DEN_COEFF_20:', 'LAT_NUM_COEFF_1:', 'LAT_NUM_COEFF_2:',
                    'LAT_NUM_COEFF_3:', 'LAT_NUM_COEFF_4:', 'LAT_NUM_COEFF_5:', 'LAT_NUM_COEFF_6:',
                    'LAT_NUM_COEFF_7:', 'LAT_NUM_COEFF_8:', 'LAT_NUM_COEFF_9:', 'LAT_NUM_COEFF_10:',
                    'LAT_NUM_COEFF_11:', 'LAT_NUM_COEFF_12:', 'LAT_NUM_COEFF_13:', 'LAT_NUM_COEFF_14:',
                    'LAT_NUM_COEFF_15:', 'LAT_NUM_COEFF_16:', 'LAT_NUM_COEFF_17:', 'LAT_NUM_COEFF_18:',
                    'LAT_NUM_COEFF_19:', 'LAT_NUM_COEFF_20:', 'LAT_DEN_COEFF_1:', 'LAT_DEN_COEFF_2:',
                    'LAT_DEN_COEFF_3:', 'LAT_DEN_COEFF_4:', 'LAT_DEN_COEFF_5:', 'LAT_DEN_COEFF_6:',
                    'LAT_DEN_COEFF_7:', 'LAT_DEN_COEFF_8:', 'LAT_DEN_COEFF_9:', 'LAT_DEN_COEFF_10:',
                    'LAT_DEN_COEFF_11:', 'LAT_DEN_COEFF_12:', 'LAT_DEN_COEFF_13:', 'LAT_DEN_COEFF_14:',
                    'LAT_DEN_COEFF_15:', 'LAT_DEN_COEFF_16:', 'LAT_DEN_COEFF_17:', 'LAT_DEN_COEFF_18:',
                    'LAT_DEN_COEFF_19:', 'LAT_DEN_COEFF_20:', 'LON_NUM_COEFF_1:', 'LON_NUM_COEFF_2:',
                    'LON_NUM_COEFF_3:', 'LON_NUM_COEFF_4:', 'LON_NUM_COEFF_5:', 'LON_NUM_COEFF_6:',
                    'LON_NUM_COEFF_7:', 'LON_NUM_COEFF_8:', 'LON_NUM_COEFF_9:', 'LON_NUM_COEFF_10:',
                    'LON_NUM_COEFF_11:', 'LON_NUM_COEFF_12:', 'LON_NUM_COEFF_13:', 'LON_NUM_COEFF_14:',
                    'LON_NUM_COEFF_15:', 'LON_NUM_COEFF_16:', 'LON_NUM_COEFF_17:', 'LON_NUM_COEFF_18:',
                    'LON_NUM_COEFF_19:', 'LON_NUM_COEFF_20:', 'LON_DEN_COEFF_1:', 'LON_DEN_COEFF_2:',
                    'LON_DEN_COEFF_3:', 'LON_DEN_COEFF_4:', 'LON_DEN_COEFF_5:', 'LON_DEN_COEFF_6:',
                    'LON_DEN_COEFF_7:', 'LON_DEN_COEFF_8:', 'LON_DEN_COEFF_9:', 'LON_DEN_COEFF_10:',
                    'LON_DEN_COEFF_11:', 'LON_DEN_COEFF_12:', 'LON_DEN_COEFF_13:', 'LON_DEN_COEFF_14:',
                    'LON_DEN_COEFF_15:', 'LON_DEN_COEFF_16:', 'LON_DEN_COEFF_17:', 'LON_DEN_COEFF_18:',
                    'LON_DEN_COEFF_19:', 'LON_DEN_COEFF_20:']
    addition1 = ['pixels', 'pixels', 'degrees', 'degrees', 'meters', 'pixels', 'pixels', 'degrees', 'degrees',
                    'meters']

    text = ""

    text += addition0[0] + " " + str(rpc.LINE_OFF) + " " + addition1[0] + "\n"
    text += addition0[1] + " " + str(rpc.SAMP_OFF) + " " + addition1[1] + "\n"
    text += addition0[2] + " " + str(rpc.LAT_OFF) + " " + addition1[2] + "\n"
    text += addition0[3] + " " + str(rpc.LON_OFF) + " " + addition1[3] + "\n"
    text += addition0[4] + " " + str(rpc.HEIGHT_OFF) + " " + addition1[4] + "\n"
    text += addition0[5] + " " + str(rpc.LINE_SCALE) + " " + addition1[5] + "\n"
    text += addition0[6] + " " + str(rpc.SAMP_SCALE) + " " + addition1[6] + "\n"
    text += addition0[7] + " " + str(rpc.LAT_SCALE) + " " + addition1[7] + "\n"
    text += addition0[8] + " " + str(rpc.LON_SCALE) + " " + addition1[8] + "\n"
    text += addition0[9] + " " + str(rpc.HEIGHT_SCALE) + " " + addition1[9] + "\n"

    for i in range(10, 30):
        text += addition0[i] + " " + str(rpc.LNUM[i - 10]) + "\n"
    for i in range(30, 50):
        text += addition0[i] + " " + str(rpc.LDEM[i - 30]) + "\n"
    for i in range(50, 70):
        text += addition0[i] + " " + str(rpc.SNUM[i - 50]) + "\n"
    for i in range(70, 90):
        text += addition0[i] + " " + str(rpc.SDEM[i - 70]) + "\n"
    for i in range(90, 110):
        text += addition0[i] + " " + str(rpc.LATNUM[i - 90]) + "\n"
    for i in range(110, 130):
        text += addition0[i] + " " + str(rpc.LATDEM[i - 110]) + "\n"
    for i in range(130, 150):
        text += addition0[i] + " " + str(rpc.LONNUM[i - 130]) + "\n"
    for i in range(150, 170):
        text += addition0[i] + " " + str(rpc.LONDEM[i - 150]) + "\n"

    f = open(path, "w")
    f.write(text)
    f.close()

def write_point_cloud(path, points):
    pcd = pylas.create()

    pcd.x = points[:, 0]
    pcd.y = points[:, 1]
    pcd.z = points[:, 2]

    pcd.write(path) 

def read_point_cloud(path):
    las = pylas.read(path)
    points = np.stack([las.x, las.y, las.z], axis = -1)
    return points

# nband: gray -> 1, rgb -> 3
# proj: WKT String
# geoTrans: affine transform, help pixel(col, line) -> geo(x, y)
# invalid_value: nonsens data value
def raster_create(raster_path, width, height, nband, proj, geoTrans, invalid_value, dtype="Byte"):
    driver = gdal.GetDriverByName("GTiff")

    if dtype == "Byte":
        raster_type = gdal.GDT_Byte
    elif dtype == "Float32":
        raster_type = gdal.GDT_Float32
    else:
        raise Exception("{} not supported yet.".format(dtype))
    
    dataset = driver.Create(raster_path, width, height, nband, raster_type)
    if invalid_value is not None:
        for i in range(nband):
            band = dataset.GetRasterBand(i + 1)
            band.SetNoDataValue(invalid_value)
    dataset.SetGeoTransform(geoTrans)
    dataset.SetProjection(proj)

    # write into dsm file
    text_ = str(geoTrans[1]) + "\n" + str(geoTrans[2]) + "\n" + str(geoTrans[4]) + "\n" + str(geoTrans[5]) + "\n"
    text_ += str(geoTrans[0] + float(geoTrans[1]) / 2) + "\n" + str(geoTrans[3] + float(geoTrans[5]) / 2)
    tfw_path = raster_path.replace(".tif", ".tfw")
    with open(tfw_path, "w") as f:
        f.write(text_)

    del driver, dataset

def write_dsm(out_path, xlu, ylu, data):
    dataset = gdal.Open(out_path, gdal.GF_Write)
    if dataset is None:
        raise Exception("GDAL RasterIO Error: Opening" + out_path + " failed!")

    if data is None:
        return

    if len(data.shape) == 3:
        im_bands = data.shape[0]
    else:
        im_bands = 1

    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(data, xlu, ylu)
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i + 1).WriteArray(data[i], xlu, ylu)
    del dataset
