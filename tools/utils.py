import json
import os
import torch
import numpy as np
import torchvision.utils as vutils
import cv2
from osgeo import gdal
import pylas
from dataset.rpc_model import RPCModel

# torch.no_grad warpper for functions
def make_nograd_func(func): # TODO what means
    def wrapper(*f_args, **f_kwargs):
        with torch.no_grad():
            ret = func(*f_args, **f_kwargs)
        return ret

    return wrapper

# convert a function into recursive style to handle nested dict/list/tuple variables
def make_recursive_func(func): # TODO what means
    def wrapper(vars):
        if isinstance(vars, list):
            return [wrapper(x) for x in vars]
        elif isinstance(vars, tuple):
            return tuple([wrapper(x) for x in vars])
        elif isinstance(vars, dict):
            return {k: wrapper(v) for k, v in vars.items()}
        else:
            return func(vars)

    return wrapper

# a wrapper to compute metrics for each image individually
def compute_metrics_for_each_image(metric_func):
    def wrapper(depth_est, depth_gt, mask, *args):
        batch_size = depth_gt.shape[0]
        results = []
        # compute result one by one
        for idx in range(batch_size):
            ret = metric_func(depth_est[idx], depth_gt[idx], mask[idx], *args)
            results.append(ret)
        return torch.stack(results).mean()

    return wrapper

@make_recursive_func
def tocuda(vars): # TODO what means
    if isinstance(vars, torch.Tensor):
        return vars.cuda()
    elif isinstance(vars, str):
        return vars
    else:
        raise NotImplementedError("invalid input type {} for tensor2numpy".format(type(vars)))
    
# NOTE: please do not use this to build up training loss
@make_nograd_func
@compute_metrics_for_each_image
def AbsDepthError_metrics(depth_est, depth_gt, mask, thres=10.0):
    depth_est, depth_gt = depth_est[mask], depth_gt[mask]
    diff = (depth_est - depth_gt).abs()
    mask2 = (diff < thres)
    result = diff[mask2]
    #result = torch.sum(result)/torch.sum(mask2*mask1)
    #return torch.mean((depth_est - depth_gt).abs())
    return torch.mean(result)

@make_nograd_func
@compute_metrics_for_each_image
def Threshold_metrics(depth_est, depth_gt, mask, thres):
    assert isinstance(thres, (int, float))
    depth_est, depth_gt = depth_est[mask], depth_gt[mask]
    errors = torch.abs(depth_est - depth_gt)
    err_mask = errors < thres
    return torch.mean(err_mask.float())

@make_nograd_func
@compute_metrics_for_each_image
def MAE_metrics(depth_est, depth_gt, mask):
    depth_est, depth_gt = depth_est[mask], depth_gt[mask]
    if depth_est.numel() == 0:
        return torch.tensor(0.0, device=depth_est.device)
    
    abs_diff = (depth_est - depth_gt).abs()
    return torch.mean(abs_diff)

@make_nograd_func
@compute_metrics_for_each_image
def RMSE_metrics(depth_est, depth_gt, mask):
    depth_est, depth_gt = depth_est[mask], depth_gt[mask]
    if depth_est.numel() == 0:
        return torch.tensor(0.0, device=depth_est.device)
    
    square_diff = (depth_est - depth_gt) ** 2
    return torch.sqrt(torch.mean(square_diff))

@make_nograd_func
@compute_metrics_for_each_image
def Completeness_metrics(depth_est, depth_gt, mask, conf_thres=0.8):
    valid_mask = (depth_est > conf_thres) # depth_est is photometric_confidence to meet '@compute_metrics_for_each_image' format requirement

    if mask.sum() == 0:
        return torch.tensor(0.0, device=depth_est.device)
    return torch.mean(valid_mask[mask].float())

@make_recursive_func
def tensor2float(vars):
    if isinstance(vars, float):
        return vars
    elif isinstance(vars, torch.Tensor):
        return vars.data.item()
    else:
        raise NotImplementedError("invalid input type {} for tensor2float".format(type(vars)))

@make_recursive_func
def tensor2numpy(vars):
    if isinstance(vars, np.ndarray):
        return vars
    elif isinstance(vars, torch.Tensor):
        return vars.detach().cpu().numpy().copy()
    else:
        raise NotImplementedError("invalid input type {} for tensor2numpy".format(type(vars)))

def save_scalars(logger, mode, scalar_dict, global_step):
    scalar_dict = tensor2float(scalar_dict)
    for key, value in scalar_dict.items():
        if not isinstance(value, (list, tuple)):
            name = '{}/{}'.format(mode, key)
            logger.add_scalar(name, value, global_step)
        else:
            for idx in range(len(value)):
                name = '{}/{}_{}'.format(mode, key, idx)
                logger.add_scalar(name, value[idx], global_step)


def save_images(logger, mode, images_dict, global_step):
    images_dict = tensor2numpy(images_dict)

    def preprocess(name, img):
        if not (len(img.shape) == 3 or len(img.shape) == 4):
            raise NotImplementedError("invalid img shape {}:{} in save_images".format(name, img.shape))
        if len(img.shape) == 3:
            img = img[:, np.newaxis, :, :]
        img = torch.from_numpy(img[:1])
        return vutils.make_grid(img, padding=0, nrow=1, normalize=True, scale_each=True)

    for key, value in images_dict.items():
        if not isinstance(value, (list, tuple)):
            name = '{}/{}'.format(mode, key)
            logger.add_image(name, preprocess(name, value), global_step)
        else:
            for idx in range(len(value)):
                name = '{}/{}_{}'.format(mode, key, idx)
                logger.add_image(name, preprocess(name, value[idx]), global_step)

class DictAverageMeter(object):
    def __init__(self):
        self.data = {}
        self.count = 0

    def update(self, new_input):
        self.count += 1
        if len(self.data) == 0:
            for k, v in new_input.items():
                if not isinstance(v, float):
                    raise NotImplementedError("invalid data {}: {}".format(k, type(v)))
                self.data[k] = v
        else:
            for k, v in new_input.items():
                if not isinstance(v, float):
                    raise NotImplementedError("invalid data {}: {}".format(k, type(v)))
                self.data[k] += v

    def mean(self):
        return {k: v / self.count for k, v in self.data.items()}

def visualize_depth(depth, mask=None, min_val=None, max_val=None):
    if mask is not None:
        depth = np.where(mask > 0.5, depth, 0)
        
    if min_val is None:
        min_val = np.percentile(depth[depth > 0], 5) if np.any(depth > 0) else 0
    if max_val is None:
        max_val = np.percentile(depth[depth > 0], 95) if np.any(depth > 0) else 1

    depth_norm = (depth - min_val) / (max_val - min_val + 1e-6)
    depth_norm = np.clip(depth_norm, 0, 1)
    depth_int = (depth_norm * 255).astype(np.uint8)
    
    return cv2.applyColorMap(depth_int, cv2.COLORMAP_MAGMA)

def unnormalize_image(img_tensor):
    img = tensor2numpy(img_tensor) # [3, H, W]
    img = np.transpose(img, (1, 2, 0)) # [H, W, 3]
    
    img_min, img_max = img.min(), img.max()
    img = (img - img_min) / (img_max - img_min + 1e-6)
    return (img * 255).astype(np.uint8)

def read_config(config_file):
    with open(config_file, "r") as f:
        config = json.load(f)
    return config

def mkdir_if_not_exist(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

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
        view_idx_list.append(ref_idx)

        src_content = content[i * 2 + 1].split(" ")
        src_num = int(src_content[0])
        if src_num != len(src_content[1:]) / 2:
            raise ValueError(
                f"read_pair_from_txt: read {txt_file}, should get {src_num} source view(s), but get {len(src_content[1:]) / 2} source view(s)."
            )

        for j in range(src_num):
            src_idx = src_content[j * 2 + 1]
            view_idx_list.append(src_idx)

    return view_idx_list

def read_np_array_from_txt(txt_file):
    with open(txt_file, "r") as f:
        text = f.read().splitlines()

    res = np.array(text, float)
    return res

def get_image_size(image_path):
    image = gdal.Open(image_path)
    if image is None:
        raise Exception(f"gdal open {image_path} failed")
    
    width = image.RasterXSize
    height = image.RasterYSize

    del image
    return width, height

def read_image(image_path, x_start, y_start, x_size, y_size):
    data = gdal.Open(image_path)
    if data is None:
        raise Exception("GDAL RasterIO Error: Opening" + image_path + " failed!")
    
    data = data.ReadAsArray(x_start, y_start, x_size, y_size)

    if len(data.shape) == 2: # TODO: what?
        # small value adjust
        small_thre = np.percentile(data.reshape(-1, 1), 2) # TODO: what
        data[data < small_thre] = small_thre
        
        # big value adjust
        big_thre = np.percentile(data.reshape(-1, 1), 98) # TODO: what
        data[data < big_thre] = big_thre

        image = 255 * (data - small_thre) / (big_thre - small_thre)
        image = image.astype(np.uint8)
        res = np.stack([image, image, image], axis = 0)

    del data
    return res 

def save_image(path, image):
    import cv2
    cv2.imwrite(path, image)

def save_rpc(self, path):
    addition0 = ['LINE_OFF:', 'SAMP_OFF:', 'LAT_OFF:', 'LONG_OFF:', 'HEIGHT_OFF:', 'LINE_SCALE:', 'SAMP_SCALE:',
                    'LAT_SCALE:', 'LONG_SCALE:', 'HEIGHT_SCALE:', 'LINE_NUM_COEFF_1:', 'LINE_NUM_COEFF_2:',
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
                    'LAT_DEN_COEFF_19:', 'LAT_DEN_COEFF_20:', 'LONG_NUM_COEFF_1:', 'LONG_NUM_COEFF_2:',
                    'LONG_NUM_COEFF_3:', 'LONG_NUM_COEFF_4:', 'LONG_NUM_COEFF_5:', 'LONG_NUM_COEFF_6:',
                    'LONG_NUM_COEFF_7:', 'LONG_NUM_COEFF_8:', 'LONG_NUM_COEFF_9:', 'LONG_NUM_COEFF_10:',
                    'LONG_NUM_COEFF_11:', 'LONG_NUM_COEFF_12:', 'LONG_NUM_COEFF_13:', 'LONG_NUM_COEFF_14:',
                    'LONG_NUM_COEFF_15:', 'LONG_NUM_COEFF_16:', 'LONG_NUM_COEFF_17:', 'LONG_NUM_COEFF_18:',
                    'LONG_NUM_COEFF_19:', 'LONG_NUM_COEFF_20:', 'LONG_DEN_COEFF_1:', 'LONG_DEN_COEFF_2:',
                    'LONG_DEN_COEFF_3:', 'LONG_DEN_COEFF_4:', 'LONG_DEN_COEFF_5:', 'LONG_DEN_COEFF_6:',
                    'LONG_DEN_COEFF_7:', 'LONG_DEN_COEFF_8:', 'LONG_DEN_COEFF_9:', 'LONG_DEN_COEFF_10:',
                    'LONG_DEN_COEFF_11:', 'LONG_DEN_COEFF_12:', 'LONG_DEN_COEFF_13:', 'LONG_DEN_COEFF_14:',
                    'LONG_DEN_COEFF_15:', 'LONG_DEN_COEFF_16:', 'LONG_DEN_COEFF_17:', 'LONG_DEN_COEFF_18:',
                    'LONG_DEN_COEFF_19:', 'LONG_DEN_COEFF_20:']
    addition1 = ['pixels', 'pixels', 'degrees', 'degrees', 'meters', 'pixels', 'pixels', 'degrees', 'degrees',
                    'meters']

    text = ""

    text += addition0[0] + " " + str(self.LINE_OFF) + " " + addition1[0] + "\n"
    text += addition0[1] + " " + str(self.SAMP_OFF) + " " + addition1[1] + "\n"
    text += addition0[2] + " " + str(self.LAT_OFF) + " " + addition1[2] + "\n"
    text += addition0[3] + " " + str(self.LONG_OFF) + " " + addition1[3] + "\n"
    text += addition0[4] + " " + str(self.HEIGHT_OFF) + " " + addition1[4] + "\n"
    text += addition0[5] + " " + str(self.LINE_SCALE) + " " + addition1[5] + "\n"
    text += addition0[6] + " " + str(self.SAMP_SCALE) + " " + addition1[6] + "\n"
    text += addition0[7] + " " + str(self.LAT_SCALE) + " " + addition1[7] + "\n"
    text += addition0[8] + " " + str(self.LONG_SCALE) + " " + addition1[8] + "\n"
    text += addition0[9] + " " + str(self.HEIGHT_SCALE) + " " + addition1[9] + "\n"

    for i in range(10, 30):
        text += addition0[i] + " " + str(self.LNUM[i - 10]) + "\n"
    for i in range(30, 50):
        text += addition0[i] + " " + str(self.LDEM[i - 30]) + "\n"
    for i in range(50, 70):
        text += addition0[i] + " " + str(self.SNUM[i - 50]) + "\n"
    for i in range(70, 90):
        text += addition0[i] + " " + str(self.SDEM[i - 70]) + "\n"
    for i in range(90, 110):
        text += addition0[i] + " " + str(self.LATNUM[i - 90]) + "\n"
    for i in range(110, 130):
        text += addition0[i] + " " + str(self.LATDEM[i - 110]) + "\n"
    for i in range(130, 150):
        text += addition0[i] + " " + str(self.LONNUM[i - 130]) + "\n"
    for i in range(150, 170):
        text += addition0[i] + " " + str(self.LONDEM[i - 150]) + "\n"

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

def filter_depth(
        depths,
        rpcs,
        p_ratio,
        d_ratio,
        geo_consitency_thre,
        prob = None,
        cofidence_ratio = 0.0
):
    ref_depth = depths[0]
    ref_rpc = rpcs[0]
    view_num = depths.shape[0]

    # ref view prob map, build photo mask
    if prob is not None:
        ref_prob = prob
        photo_mask = ref_prob > cofidence_ratio
    else:
        photo_mask = np.ones_like(ref_depth, bool)

    # build geometric mask
    geo_mask_sum = 0
    depth_map_reproj = []
    for idx in range(1, view_num):
        src_depth = depths[idx]
        src_rpc = rpcs[idx]

        geo_mask_tmp, ref_depth_reproj_tmp = reproj_and_check(ref_depth, ref_rpc, src_depth, src_rpc, p_ratio, d_ratio)

        geo_mask_sum += geo_mask_tmp.astype(np.int32)
        depth_map_reproj.append(ref_depth_reproj_tmp)
    
    # average
    geo_mask = geo_mask_sum >= geo_consitency_thre
    depth_map_est = (sum(depth_map_reproj) + ref_depth) / (geo_mask_sum + 1) # TODO: why
    final_mask = np.logical_and(photo_mask, geo_mask)

    return final_mask, depth_map_est
        
def reproj_and_check(ref_depth, ref_rpc, src_depth, src_rpc, p_ratio, d_ratio):
    ref_rpc_model = RPCModel(ref_rpc)
    src_rpc_model = RPCModel(src_rpc)

    width, height = ref_depth.shape[1], ref_depth.shape[0]
    ref_x, ref_y = np.meshgrid(np.arange(0, width), np.arange(0, height))
    ref_x, ref_y = ref_x.reshape([-1]), ref_y.reshape([-1])

    # depth sample
    lat, lon = ref_rpc_model.photo2obj(ref_x.astype(float), ref_y.astype(float), ref_depth([-1]))
    src_x, src_y = src_rpc_model.obj2photo(lat, lon, ref_depth.reshape([-1]))
    src_x, src_y = src_x.reshape([height, width]), src_y.reshape([height, width])
    src_depth_sample = cv2.remap(src_depth, src_x.astype(np.float32), src_y.astype(np.float32), interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=-999)

    # projection and back-projection
    lat, lon = src_rpc_model.photo2obj(src_x.astype(float), src_y.astype(float), src_depth_sample.reshape(-1))
    reproj_x, reproj_y = ref_rpc_model.obj2photo(lat, lon, src_depth_sample(-1))

    # check x,y error
    xy_diff = np.sqrt((reproj_x - ref_x) ** 2 + (reproj_y - ref_y) ** 2)
    depth_diff = np.abs(src_depth_sample - ref_depth)

    # build mask
    mask = np.logical_and(xy_diff < p_ratio, depth_diff < d_ratio)
    depth_diff[~mask] = 0 # fliter high-error points

    return mask, depth_diff

def raster_create(raster_path, width, height, nband, proj, geoTrans, invalid_value, dtype="Byte"): # TODO: read
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
    
def build_dsm(points, ul_e, ul_n, xunit, yunit, e_size, n_size): # TODO: read
    dsm = proj_to_grid(points, ul_e, ul_n, xunit, yunit, e_size, n_size)
    dsm = cv2.medianBlur(dsm.astype(np.float32), 3)

    return dsm

def proj_to_grid(points, xoff, yoff, xresolution, yresolution, xsize, ysize): # TODO: read
    row = np.floor((yoff - points[:, 1]) / xresolution).astype(dtype=int)
    col = np.floor((points[:, 0] - xoff) / yresolution).astype(dtype=int)

    points_group_idx = row * xsize + col
    points_val = points[:, 2]

    # remove points that lie out of the dsm boundary
    mask = ((row >= 0) * (col >= 0) * (row < ysize) * (col < xsize)) > 0

    # print("mask num:", np.sum(mask.astype(int)))

    points_group_idx = points_group_idx[mask]
    points_val = points_val[mask]

    # create a place holder for all pixels in the dsm
    group_idx = np.arange(xsize * ysize).astype(dtype=int)
    group_val = np.empty(xsize * ysize)
    group_val.fill(np.nan)

    # concatenate place holders with the real valuies, then aggregate
    group_idx = np.concatenate((group_idx, points_group_idx))
    group_val = np.concatenate((group_val, points_val))

    dsm = npg.aggregate(group_idx, group_val, func='nanmax', fill_value=np.nan)
    dsm = dsm.reshape((ysize, xsize))

    # try to fill very small holes
    dsm_new = dsm.copy()
    nan_places = np.argwhere(np.isnan(dsm_new))
    for i in range(nan_places.shape[0]):
        row = nan_places[i, 0]
        col = nan_places[i, 1]
        neighbors = []
        for j in range(row-1, row+2):
            for k in range(col-1, col+2):
                if j >= 0 and j < dsm_new.shape[0] and k >=0 and k < dsm_new.shape[1]:
                    val = dsm_new[j, k]
                    if not np.isnan(val):
                        neighbors.append(val)

        if neighbors:
            dsm[row, col] = np.median(neighbors)

    return dsm

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