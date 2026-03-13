import argparse
import os
import time
import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import torch.nn as nn
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from dataset.data_io import save_pfm

from tools.utils import *
from dataset import find_dataset
from networks.casmvs import *
from networks.loss import casmvs_loss

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
parser.add_argument('--eta', action='store_true', help='use eta in cost volume')
parser.add_argument('--attn_temp', type = float, default = 2.0, help = 'attention temperature for eta')
parser.add_argument('--dlossw', type = str, default = "0.5,1.0,2.0", help = 'depth loss weight for each stage (for eval metrics)')
parser.add_argument('--summary_freq', type = int, default = 50, help = 'tensorboard summary frequency in prediction')
# others setting
parser.add_argument('--gpu_id', type = str, default = "0")

@make_nograd_func
def predict_batch(model, sample, ndepths, dlossw=None):
    model.eval()

    # run multi-stage mvs pipeline
    sample_cuda = tocuda(sample)
    outputs = model(
        sample_cuda["images"],
        sample_cuda["cameras_para"],
        sample_cuda["depth_values"]
    )

    # get metrices
    # num_stage = len([int(depth) for depth in ndepths.split(",") if depth])
    # depth_est = outputs[f"stage{num_stage}"]["depth"]
    # photometric_confidence = outputs[f"stage{num_stage}"]["photometric_confidence"]

    # wrap outputs
    # image_outputs = {
    #    "depth_est": depth_est,
    #    "photometric_confidence": photometric_confidence
    # }
    
    # return image_outputs

    # get metrices
    num_stage = len([int(depth) for depth in ndepths.split(",") if depth])
    depth_est = outputs[f"stage{num_stage}"]["depth"]
    photometric_confidence = outputs[f"stage{num_stage}"]["photometric_confidence"]

    scalar_outputs = {}
    if "depth" in sample_cuda and "mask" in sample_cuda:
        depth_gt_ms = sample_cuda["depth"]
        mask_ms = sample_cuda["mask"]
        depth_final_gt = depth_gt_ms[f"stage{num_stage}"]
        mask_final = mask_ms[f"stage{num_stage}"]

        # Keep metric set consistent with train.py (test_batch detailed_summary=True).
        loss, depth_loss = casmvs_loss(
            outputs,
            depth_gt_ms,
            mask_ms,
            dlossw = [float(weight) for weight in dlossw.split(",") if weight] if dlossw is not None else None
        )
        scalar_outputs["loss"] = loss
        scalar_outputs["depth_loss"] = depth_loss
        scalar_outputs["abs_depth_error"] = AbsDepthError_metrics(depth_est, depth_final_gt, mask_final > 0.5, 250.0)
        
        scalar_outputs["mae"] = MAE_metrics(depth_est, depth_final_gt, mask_final > 0.5)
        scalar_outputs["rmse"] = RMSE_metrics(depth_est, depth_final_gt, mask_final > 0.5)
        scalar_outputs["threshold_1.0m_acc"] = Threshold_metrics(depth_est, depth_final_gt, mask_final > 0.5, 1.0)
        scalar_outputs["threshold_2.5m_acc"] = Threshold_metrics(depth_est, depth_final_gt, mask_final > 0.5, 2.5)
        scalar_outputs["threshold_7.5m_acc"] = Threshold_metrics(depth_est, depth_final_gt, mask_final > 0.5, 7.5)
        scalar_outputs["completeness"] = Completeness_metrics(photometric_confidence, depth_final_gt, mask_final > 0.5)

    # wrap outputs
    image_outputs = {
        "depth_est": depth_est,
        "photometric_confidence": photometric_confidence
    }

    if "depth" in sample_cuda and "mask" in sample_cuda:
        image_outputs["depth_gt"] = depth_final_gt
        image_outputs["ref_image"] = sample_cuda["images"][:, 0]
        image_outputs["mask"] = mask_final
        image_outputs["errormap"] = (depth_est - depth_final_gt).abs()
    
    return image_outputs, tensor2float(scalar_outputs)


def predict_cli(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    print(
        "Predict CLI params: "
        f"model={args.model}, geo_model={args.geo_model}, view_num={args.view_num}, ref_view={args.ref_view}, "
        f"batch_size={args.batch_size}, ndepths={args.ndepths}, min_interval={args.min_interval}, "
        f"depth_inter_ratio={args.depth_inter_ratio}, cr_base_chs={args.cr_base_chs}, gpu_id={args.gpu_id}"
    )

    # model compatibility check
    if args.loadckpt is None:
        raise ValueError("--loadckpt must be specified")
    # ckpt name check
    # assert args.geo_model in args.loadckpt, Exception("geo model and checkpoint do not match")
    # assert args.model in args.loadckpt, Exception("network and checkpoint do not match")
    assert os.path.isfile(args.loadckpt), Exception("--loadckpt must point a .ckpt file")

    # get dataset path
    if args.dataset_root is None:
        raise ValueError("--dataset_root must be specified to locate data.")
    pred_path = os.path.join(args.dataset_root, 'test') # TODO: test equals predict
    if not os.path.exists(pred_path):
        raise ValueError("predict_path do not exist")
    print(f"predict path: {pred_path}")

    # setup log dir
    if os.path.isfile(args.loadckpt):
        cur_log_dir = os.path.dirname(args.loadckpt)
    else:
        cur_log_dir = args.loadckpt
    pred_log_dir = os.path.join(cur_log_dir, 'predict')
    os.makedirs(pred_log_dir, exist_ok=True)
    print(f"log directory: {pred_log_dir}")

    tb_log_dir = os.path.join(pred_log_dir, "tensorboard")
    os.makedirs(tb_log_dir, exist_ok=True)
    logger = SummaryWriter(tb_log_dir)
    print(f"tensorboard directory: {tb_log_dir}")

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
        num_workers = 4,
        pin_memory = True,
        drop_last = False
    )

    # mvs model
    if args.model == "casmvs":
        model = CascadeMVSNet(
            geo_model = args.geo_model,
            refine = False,
            min_interval = args.min_interval,
            ndepths = [int(depth) for depth in args.ndepths.split(",") if depth],
            depth_intervals_ratio = [float(interval) for interval in args.depth_inter_ratio.split(",") if interval],
            cr_base_chs = [int(ch) for ch in args.cr_base_chs.split(",") if ch],
            eta = args.eta,
            attn_temp = args.attn_temp
        )
        print("use CascadeMVSNet model")
    else:
        raise Exception(f"{args.model} has no implementation")

    # move model to gpu
    model = nn.DataParallel(model)
    model.cuda()

    # load checkpoint
    loadckpt = args.loadckpt
    if os.path.isdir(loadckpt):
        saved_models = [file for file in os.listdir(loadckpt) if file.endswith(".ckpt")]
        saved_models = sorted(saved_models, key=lambda x: int(x.split('_')[1].split(".")[0]))
        loadckpt = os.path.join(loadckpt, saved_models[-1])

    print(f"load checkpoint from {loadckpt} for resume training")
    state_dict = torch.load(loadckpt)
    model.load_state_dict(state_dict['model'])

    avg_test_scalars = DictAverageMeter()
    for batch_idx, sample in enumerate(tqdm(pred_loader, desc="Predict", unit="batch")):
        start_time = time.perf_counter()
        b_idx = str(sample['view_idx'][0])
        b_name = str(sample['view_name'][0])

        image_outputs, scalar_outputs = predict_batch(model, sample, args.ndepths, args.dlossw)
        if len(scalar_outputs) > 0:
            avg_test_scalars.update(scalar_outputs)

        global_step = batch_idx
        if global_step % args.summary_freq == 0:
            if len(scalar_outputs) > 0:
                save_scalars(logger, 'predict', scalar_outputs, global_step)
            save_images(logger, 'predict', image_outputs, global_step)

        torch.cuda.synchronize()
        batch_time = time.perf_counter() - start_time
        print(f'Iter {batch_idx}/{len(pred_loader)}, name {b_name}, time = {batch_time}, metrics = {scalar_outputs}')

        # modify results's format
        depth_est_batch = tensor2numpy(image_outputs["depth_est"])
        prob_batch = tensor2numpy(image_outputs["photometric_confidence"])
        
        # Iterate over batch
        batch_size = depth_est_batch.shape[0]
        for i in range(batch_size):
            depth_est = np.squeeze(depth_est_batch[i])
            prob = np.float32(np.squeeze(prob_batch[i]))
            
            curr_b_idx = str(sample['view_idx'][i]) if isinstance(sample['view_idx'], list) or isinstance(sample['view_idx'], torch.Tensor) else str(sample['view_idx'][i].item())
            curr_b_name = str(sample['view_name'][i])

            # save depth and confidence in PFM format
            depth_dir = os.path.join(pred_log_dir, "depth_est", curr_b_idx)
            conf_dir = os.path.join(pred_log_dir, "confidence", curr_b_idx)
            depth_color_dir = os.path.join(depth_dir, "color")
            conf_color_dir = os.path.join(conf_dir, "color")
            os.makedirs(depth_dir, exist_ok=True)
            os.makedirs(conf_dir, exist_ok=True)
            os.makedirs(depth_color_dir, exist_ok=True)
            os.makedirs(conf_color_dir, exist_ok=True)
            
            depth_path = os.path.join(depth_dir, f"{curr_b_name}.pfm")
            conf_path = os.path.join(conf_dir, f"{curr_b_name}.pfm")
            depth_color_path = os.path.join(depth_color_dir, f"{curr_b_name}.png")
            conf_color_path = os.path.join(conf_color_dir, f"{curr_b_name}.png")
            
            save_pfm(depth_path, depth_est.astype(np.float32))
            save_pfm(conf_path, prob.astype(np.float32))
            plt.imsave(depth_color_path, depth_est, format="png")
            plt.imsave(conf_color_path, prob, format="png")

            print(f'Iter {batch_idx}/{len(pred_loader)} (Batch {i}/{batch_size}), Saved: {depth_path}')

    logger.close()
    print(f"avg_test_scalars: {avg_test_scalars.mean()}")
    # write prediction metrics
    record_path = os.path.join(pred_log_dir, 'predict_record.txt')
    current_avg_metrics = avg_test_scalars.mean()
    with open(record_path, "a+", encoding="utf-8") as f:
        f.write(f"Predict Metrics: {current_avg_metrics}\n")

def predict(
        output_path,
        depth_range,
        view_num,
        cfg,
        geo_model = None # for dsm output
):
    if geo_model is None:
        geo_model = cfg.geo_model
    if cfg.geo_model != geo_model:
        raise ValueError("geo_model mismatch between cfg and predict() argument")
    if cfg.loadckpt is None:
        raise ValueError("--loadckpt must be specified for prediction")
    
    # dataset and dataloader
    mvsdataset = find_dataset(geo_model)
    pred_dataset = mvsdataset( # TODO: ref_view?
        output_path,
        "pred",
        view_num,
        depth_range = depth_range
    )
    pred_loader = DataLoader(
        pred_dataset,
        cfg.batch_size,
        shuffle = False,
        num_workers = 4,
        pin_memory = True,
        drop_last = False
    )
    
    # model
    model = None
    if cfg.model == "casmvs":
        model = CascadeMVSNet(
            geo_model = geo_model,
            refine = False,
            min_interval = depth_range[2], # TODO: too hard-code..
            ndepths = [int(depth) for depth in cfg.ndepths.split(",") if depth],
            depth_intervals_ratio = [float(interval) for interval in cfg.depth_inter_ratio.split(",") if interval],
            cr_base_chs = [int(ch) for ch in cfg.cr_base_chs.split(",") if ch],
            eta = cfg.eta,
            attn_temp = cfg.attn_temp
        )
        print(f"use CascadeMVSNet model")
    else:
        raise Exception(f"{cfg.model} has no implementation")
    
    model = nn.DataParallel(model)
    model.cuda()

    # load checkpoint
    loadckpt = cfg.loadckpt
    if os.path.isdir(loadckpt):
        saved_models = [file for file in os.listdir(loadckpt) if file.endswith(".ckpt")]
        saved_models = sorted(saved_models, key=lambda x: int(x.split('_')[1].split(".")[0]))
        loadckpt = os.path.join(loadckpt, saved_models[-1])
    state_dict = torch.load(loadckpt)
    model.load_state_dict(state_dict['model'])

    # create folder
    output_folder = os.path.join(output_path, 'mvs')
    mkdir_if_not_exist(output_folder)

    # model inference
    avg_test_scalars = DictAverageMeter()
    for batch_idx, sample in enumerate(tqdm(pred_loader, desc="Predict dsm", unit="batch")):
        start_time = time.time()
        b_idx = str(sample['view_idx'][0])
        b_name = str(sample['view_name'][0])

        # build result
        image_outputs, scalar_outputs = predict_batch(model, sample, cfg.ndepths, getattr(cfg, "dlossw", None))
        if len(scalar_outputs) > 0:
            avg_test_scalars.update(scalar_outputs)
        
        depth_est = np.squeeze(tensor2numpy(image_outputs["depth_est"]))
        prob = np.float32(np.squeeze(tensor2numpy(image_outputs["photometric_confidence"])))

        # create sub-folder
        output_folder_view = os.path.join(output_folder, b_idx)
        output_folder_view_depth_map = os.path.join(output_folder_view, "depth_est")
        output_folder_view_depth_map_color = os.path.join(output_folder_view_depth_map, "color")
        output_folder_view_prob = os.path.join(output_folder_view, "confidence")
        output_folder_view_prob_color = os.path.join(output_folder_view_prob, "color")
        mkdir_if_not_exist(output_folder_view)
        mkdir_if_not_exist(output_folder_view_depth_map)
        mkdir_if_not_exist(output_folder_view_depth_map_color)
        mkdir_if_not_exist(output_folder_view_prob)
        mkdir_if_not_exist(output_folder_view_prob_color)

        # save result
        save_pfm(os.path.join(output_folder_view_depth_map, f"{b_name}.pfm"), depth_est.astype(np.float32))
        plt.imsave(os.path.join(output_folder_view_depth_map_color, f"{b_name}.png"), depth_est, format="png")
        save_pfm(os.path.join(output_folder_view_prob, f"{b_name}.pfm"), prob.astype(np.float32))
        plt.imsave(os.path.join(output_folder_view_prob_color, f"{b_name}.png"), prob, format="png")

        print("Iter {}/{}, {}, time = {:.3f}, metrics = {}".format(batch_idx, len(pred_loader), b_name, time.time() - start_time, scalar_outputs))

        del image_outputs
    
    print(f"avg_test_scalars: {avg_test_scalars.mean()}")
    # write prediction metrics
    record_path = os.path.join(output_path, 'predict_record.txt')
    current_avg_metrics = avg_test_scalars.mean()
    with open(record_path, "a+", encoding="utf-8") as f:
        f.write(f"Predict Metrics: {current_avg_metrics}\n")


if __name__ == '__main__':
    predict_cli(parser.parse_args())