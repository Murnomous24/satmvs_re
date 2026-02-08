import json
import os
import torch
import numpy as np
import torchvision.utils as vutils
import cv2
from osgeo import gdal

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