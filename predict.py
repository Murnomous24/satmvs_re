import argparse
import os
import time
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import cv2
import matplotlib.pyplot as plt
from dataset.data_io import save_pfm

from tools.utils import *
from dataset import find_dataset
from networks.casmvs import *

parser = argparse.ArgumentParser(description = "satmvs_re predicting file")
# arguments option
parser.add_argument('--model', default = 'casmvs', help = 'select model', choices = ['casmvs'])
parser.add_argument('--geo_model', default = 'pinhole', help = 'select dataset format', choices = ['pinhole', 'rpc'])
# dataset
parser.add_argument('--dataset_root', default = None, help = 'dataset root')
parser.add_argument('--loadckpt', default = None, help = 'specific checkpoint file for prediction')
parser.add_argument('--logdir', default = None, help = 'the folder save training logs')
parser.add_argument('--resume', default = False, help = 'continue to train model in old checkpoint')
parser.add_argument('--batch_size', type = int, default = 1, help = "batch size")
# mvs setting
parser.add_argument('--view_num', type = int, default = 3, help = 'number of input view')
parser.add_argument('--ref_view', type = int, default = 2, help = 'index of reference view')
# cascade setting
parser.add_argument('--ndepths', type = str, default = "64,32,8", help = "number of depths")
parser.add_argument('--min_interval', type = float, default = 2.5, help = "min interval of each depth plane")
parser.add_argument('--depth_inter_ratio', type = str, default = "4,2,1", help = "depth interval ratio") # TODO: what
parser.add_argument('--cr_base_chs', type = str, default = "8,8,8", help = "cost volume regularization base channels")
# others setting
parser.add_argument('--gpu_id', type = str, default = "0")

# parse arguments and check
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

# model compatibility check
assert args.geo_model in args.loadckpt, Exception("geo model and checkpoint do not match")
assert args.model in args.loadckpt, Exception("network and checkpoint do not match")
assert os.path.isfile(args.loadckpt), Exception("--loadckpt must point a .ckpt file")

# get dataset path
if args.dataset_root is None:
    raise ValueError("--dataset_root must be specified to locate data.")
pred_path = os.path.join(args.dataset_root, 'test') # TODO: test equals predict
if not os.path.exists(pred_path):
    raise ValueError("predict_path do not exist")

# setup log dir
if args.loadckpt is None:
    raise ValueError("predict need 'loadckpt', but get none")
if os.path.isfile(args.loadckpt):
    cur_log_dir = os.path.dirname(args.loadckpt)
else:
    cur_log_dir = args.loadckpt
pred_log_dir = os.path.join(cur_log_dir, 'predict')
if not os.path.isdir(pred_log_dir):
        os.mkdir(pred_log_dir)
print(f"log directory: {pred_log_dir}")

# dataset and dataloader
mvsdataset = find_dataset(args.geo_model)
pred_dataset = mvsdataset(
    pred_path,
    "test", # TODO: modify pred dataset mode
    args.view_num,
    ref_view = args.ref_view
)
pred_loader = DataLoader(
    pred_dataset,
    args.batch_size,
    shuffle = False,
    num_workers = 0,
    drop_last = False
)

# mvs model
model = None
if args.model == "casmvs":
    model = CascadeMVSNet(
        geo_model = args.geo_model,
        refine = False,
        min_interval = args.min_interval,
        ndepths = [int(depth) for depth in args.ndepths.split(",") if depth],
        depth_intervals_ratio = [float(interval) for interval in args.depth_inter_ratio.split(",") if interval],
        cr_base_chs = [int(ch) for ch in args.cr_base_chs.split(",") if ch]
    )
    print(f"use CascadeMVSNet model")
else:
    raise Exception(f"{args.model} has no implementation")

# move model to gpu
model = nn.DataParallel(model)
model.cuda()

# load checkpoint
print(f"load checkpoint from {args.loadckpt}")
loadckpt = args.loadckpt
if os.path.isdir(loadckpt):
    saved_models = [file for file in os.listdir(loadckpt) if file.endswith(".ckpt")]
    saved_models = sorted(saved_models, key=lambda x: int(x.split('_')[1].split(".")[0]))
    loadckpt = os.path.join(loadckpt, saved_models[-1])

print(f"load checkpoint from {loadckpt} for resume training")
state_dict = torch.load(loadckpt)
model.load_state_dict(state_dict['model'])

@make_nograd_func
def predict_batch(model, sample):
    model.eval()

    # run multi-stage mvs pipeline
    sample_cuda = tocuda(sample)
    outputs = model(
        sample_cuda["images"],
        sample_cuda["cameras_para"],
        sample_cuda["depth_values"]
    )

    # get metrices
    num_stage = len([int(depth) for depth in args.ndepths.split(",") if depth])
    depth_est = outputs[f"stage{num_stage}"]["depth"]
    photometric_confidence = outputs[f"stage{num_stage}"]["photometric_confidence"]

    # wrap outputs
    image_outputs = {
        "depth_est": depth_est,
        "photometric_confidence": photometric_confidence
    }
    
    return image_outputs

def predict():
    for batch_idx, sample in enumerate(pred_loader):
        start_time = time.perf_counter()
        b_idx = str(sample['view_idx'][0])
        b_name = str(sample['view_name'][0])

        image_outputs = predict_batch(model, sample)
        torch.cuda.synchronize()
        batch_time = time.perf_counter() - start_time
        print(f'Iter {batch_idx}/{len(pred_loader)}, name {b_name}, time = {batch_time}')

        # modify results's format
        depth_est = np.squeeze(tensor2numpy(image_outputs["depth_est"]))
        prob = np.float32(np.squeeze(tensor2numpy(image_outputs["photometric_confidence"])))

        # save depth and confidence in PFM format
        depth_dir = os.path.join(pred_log_dir, "depth_est", b_idx)
        conf_dir = os.path.join(pred_log_dir, "confidence", b_idx)
        os.makedirs(depth_dir, exist_ok=True)
        os.makedirs(conf_dir, exist_ok=True)
        depth_path = os.path.join(depth_dir, f"{b_name}.pfm")
        conf_path = os.path.join(conf_dir, f"{b_name}.pfm")
        save_pfm(depth_path, depth_est.astype(np.float32))
        save_pfm(conf_path, prob.astype(np.float32))

        print(f'Iter {batch_idx}/{len(pred_loader)}, Saved: {depth_path} , {conf_path}, time = {time.perf_counter() - start_time:.3f}s')

def predict(
        output_path,
        depth_range,
        view_num,
        cfg,
        geo_model = "rpc" # for dsm output, default 'rpc'
):
    # dataset and dataloader
    mvsdataset = find_dataset(geo_model)
    pred_dataset = mvsdataset(
        output_path,
        "pred",
        view_num,
        depth_range,
        cfg
    )
    pred_loader = DataLoader(
        pred_dataset,
        cfg.batch_size,
        shuffle = False,
        num_workers = 0,
        drop_last = False
    )
    
    # model
    model = None
    if cfg.model == "casmvs":
        model = CascadeMVSNet(
            geo_model = cfg.geo_model,
            refine = False,
            min_interval = depth_range[2], # TODO: too hard-code..
            ndepths = [int(depth) for depth in cfg.ndepths.split(",") if depth],
            depth_intervals_ratio = [float(interval) for interval in cfg.depth_inter_ratio.split(",") if interval],
            cr_base_chs = [int(ch) for ch in cfg.cr_base_chs.split(",") if ch]
        )
        print(f"use CascadeMVSNet model")
    else:
        raise Exception(f"{args.model} has no implementation")
    
    model = nn.DataParallel(model)
    model.cuda()

    # load checkpoint
    loadckpt = cfg.loadckpt
    if os.path.isdir(loadckpt):
        saved_models = [file for file in os.listdir(loadckpt) if file.endswith(".ckpt")]
        saved_models = sorted(saved_models, key=lambda x: int(x.split('_')[1].split(".")[0]))
        loadckpt = os.path.join(loadckpt, saved_models[-1])
    state_dict = torch.load(loadckpt)
    model.load_state_dict(state_dict=['model'])

    # create folder
    output_folder = os.path.join(output_path, 'mvs')
    mkdir_if_not_exist(output_folder)

    # model inference
    for batch_idx, sample in enumerate(pred_loader):
        b_idx = str(sample['view_idx'][0])
        b_name = str(sample['view_name'][0])

        # build result
        image_outputs = predict_batch(model, sample)
        depth_est = np.squeeze(tensor2numpy(image_outputs["depth_est"]))
        prob = np.float32(np.squeeze(tensor2numpy(image_outputs["photometric_confidence"])))

        # create sub-folder
        output_folder_view = output_folder + f"{b_idx}"
        output_folder_view_depth_map = output_folder_view + f"depth"
        output_folder_view_depth_map_color = output_folder_view + f"depth/color"
        output_folder_view_prob = output_folder_view + f"prob"
        output_folder_view_prob_color = output_folder_view + f"prob/color"
        mkdir_if_not_exist(output_folder_view)
        mkdir_if_not_exist(output_folder_view_depth_map)
        mkdir_if_not_exist(output_folder_view_depth_map_color)
        mkdir_if_not_exist(output_folder_view_prob)
        mkdir_if_not_exist(output_folder_view_prob_color)

        # save result
        save_pfm(output_folder_view_depth_map, depth_est)
        plt.imsave(output_folder_view_depth_map_color + f"{b_name}.png", depth_est, format='png')
        save_pfm(output_folder_view_prob, prob)
        plt.imsave(output_folder_view_prob_color + f"{b_name}.png", prob, format='png')

        del image_outputs


if __name__ == '__main__':
    predict()