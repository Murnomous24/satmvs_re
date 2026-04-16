import torch
import numpy as np
import torchvision.utils as vutils
import cv2

# torch.no_grad warpper for functions
def make_nograd_func(func):
    def wrapper(*f_args, **f_kwargs):
        with torch.no_grad():
            ret = func(*f_args, **f_kwargs)
        return ret

    return wrapper

# convert a function into recursive style to handle nested dict/list/tuple variables
def make_recursive_func(func):
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
        results = [r for r in results if not torch.isnan(r)] # filter nan
        if len(results) == 0:
             return torch.tensor(0.0, device=depth_gt.device)
        return torch.stack(results).mean()

    return wrapper

@make_recursive_func
def tocuda(vars):
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
def MAE_metrics(depth_est, depth_gt, mask, thres=10.0):
    depth_est, depth_gt = depth_est[mask], depth_gt[mask]
    diff = (depth_est - depth_gt).abs()
    mask2 = (diff < thres)
    result = diff[mask2]
    return torch.mean(result)

@make_nograd_func
@compute_metrics_for_each_image
def RMSE_metrics(depth_est, depth_gt, mask, thres=10.0):
    depth_est, depth_gt = depth_est[mask], depth_gt[mask]
    diff = (depth_est - depth_gt).abs()
    mask2 = (diff < thres)
    result = diff[mask2] ** 2
    return torch.sqrt(torch.mean(result))

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

    def first_map(img):
        # Keep behavior aligned with previous implementation: log first sample only.
        if img.ndim == 4:
            return img[0]
        if img.ndim == 3:
            return img[0]
        raise NotImplementedError("invalid img dim {} in save_images".format(img.ndim))

    def get_valid_mask(h, w):
        mask_data = images_dict.get("mask", None)
        if mask_data is None:
            return np.ones((h, w), dtype=bool)

        mask_map = first_map(mask_data)
        if mask_map.ndim == 3 and mask_map.shape[0] == 1:
            mask_map = mask_map[0]
        if mask_map.shape != (h, w):
            return np.ones((h, w), dtype=bool)
        return mask_map > 0.5

    def colorize_depth(depth):
        h, w = depth.shape
        valid = np.isfinite(depth) & get_valid_mask(h, w)
        color = np.zeros((h, w, 3), dtype=np.uint8)
        color[:, :] = np.array([20, 20, 20], dtype=np.uint8)

        if np.any(valid):
            valid_values = depth[valid]
            vmin, vmax = np.percentile(valid_values, [2.0, 98.0])
            if vmax <= vmin:
                vmin = float(valid_values.min())
                vmax = float(valid_values.max()) + 1e-6

            depth_norm = np.clip((depth - vmin) / (vmax - vmin + 1e-6), 0.0, 1.0)
            depth_u8 = (depth_norm * 255.0).astype(np.uint8)
            depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
            depth_color = cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB)
            color[valid] = depth_color[valid]

        return torch.from_numpy(np.transpose(color, (2, 0, 1))).float() / 255.0

    def colorize_errormap(error_map):
        h, w = error_map.shape
        valid = np.isfinite(error_map) & get_valid_mask(h, w)
        mapped = np.zeros_like(error_map, dtype=np.float32)

        m1 = error_map < 1.0
        mapped[m1] = (error_map[m1] / 1.0) * 96.0

        m2 = (error_map >= 1.0) & (error_map < 2.5)
        mapped[m2] = 96.0 + ((error_map[m2] - 1.0) / 1.5) * (160.0 - 96.0)

        m3 = (error_map >= 2.5) & (error_map < 7.5)
        mapped[m3] = 160.0 + ((error_map[m3] - 2.5) / 5.0) * (224.0 - 160.0)

        m4 = error_map >= 7.5
        mapped[m4] = np.clip(224.0 + ((error_map[m4] - 7.5) / 2.5) * 31.0, 224.0, 255.0)

        mapped_u8 = mapped.astype(np.uint8)
        color = cv2.applyColorMap(mapped_u8, cv2.COLORMAP_JET)
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        color[~valid] = 0
        return torch.from_numpy(np.transpose(color, (2, 0, 1))).float() / 255.0

    def colorize_mask(mask_map):
        valid = mask_map > 0.5
        h, w = mask_map.shape
        color = np.zeros((h, w, 3), dtype=np.uint8)
        # Green: valid pixels kept for supervision; Red: pruned pixels.
        color[valid] = np.array([0, 220, 0], dtype=np.uint8)
        color[~valid] = np.array([220, 30, 30], dtype=np.uint8)
        return torch.from_numpy(np.transpose(color, (2, 0, 1))).float() / 255.0

    def preprocess(name, img):
        if not (len(img.shape) == 3 or len(img.shape) == 4):
            raise NotImplementedError("invalid img shape {}:{} in save_images".format(name, img.shape))

        vis_key = name.split("/")[-1].lower()
        img_map = first_map(img)

        if img_map.ndim == 2 and ("depth_est" in vis_key or "depth_gt" in vis_key):
            return colorize_depth(img_map)

        if img_map.ndim == 2 and ("errormap" in vis_key or "error_map" in vis_key):
            return colorize_errormap(img_map)

        if img_map.ndim == 2 and vis_key == "mask":
            return colorize_mask(img_map)

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
        self.weight_sum = 0.0

    def update(self, new_input, weight=1.0):
        weight = float(weight)
        if weight <= 0:
            return

        if len(self.data) == 0:
            for k, v in new_input.items():
                if not isinstance(v, float):
                    raise NotImplementedError("invalid data {}: {}".format(k, type(v)))
                self.data[k] = v * weight
        else:
            for k, v in new_input.items():
                if not isinstance(v, float):
                    raise NotImplementedError("invalid data {}: {}".format(k, type(v)))
                self.data[k] += v * weight

        self.weight_sum += weight

    def mean(self):
        if self.weight_sum <= 0:
            return {}
        return {k: v / self.weight_sum for k, v in self.data.items()}


def unnormalize_image(img_tensor):
    img = tensor2numpy(img_tensor) # [3, H, W]
    img = np.transpose(img, (1, 2, 0)) # [H, W, 3]
    
    img_min, img_max = img.min(), img.max()
    img = (img - img_min) / (img_max - img_min + 1e-6)
    return (img * 255).astype(np.uint8)
