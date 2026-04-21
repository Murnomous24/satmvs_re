import argparse
import datetime
import os
import time
import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm

from tools.utils import *
from tools.train_resume_utils import (
    StatefulRandomSampler,
    capture_rng_states,
    require_ckpt_keys,
    restore_rng_states,
    seed_everything,
    seed_worker,
)
from dataset import find_dataset
from networks.casmvs import *
from networks.loss import *


parser = argparse.ArgumentParser(description = "satmvs_re training file")
# arguments option
parser.add_argument('--mode', default = 'test', help = 'train/test', choices = ['train', 'test'])
parser.add_argument('--model', default = 'casmvs', help = 'select model', choices = ['casmvs'])
parser.add_argument('--geo_model', default = 'pinhole', help = 'select dataset format', choices = ['pinhole', 'rpc'])
# dataset
parser.add_argument('--dataset_root', default = None, help = 'dataset root')
parser.add_argument('--train_loadckpt', default = None, help = 'specific checkpoint file for resume training')
parser.add_argument('--test_loadckpt', default = None, help = 'specific checkpoint file for test')
parser.add_argument('--logdir', default = None, help = 'the folder save training logs')
parser.add_argument('--resume', action = 'store_true', help = 'continue to train model in old checkpoint')
parser.add_argument('--batch_size', type = int, default = 1, help = "batch size")
# mvs setting
parser.add_argument('--view_num', type = int, default = 3, help = 'number of input view')
parser.add_argument('--ref_view', type = int, default = 2, help = 'index of reference view')
parser.add_argument('--aux_channel', type = str, default = "gray", choices = ["gray", "gabor", "dwt"], help = "auxiliary input channel mode")
# cascade setting
parser.add_argument('--ndepths', type = str, default = "64,32,8", help = "number of depths")
parser.add_argument('--min_interval', type = float, default = 2.5, help = "min interval of each depth plane")
parser.add_argument('--depth_inter_ratio', type = str, default = "4,2,1", help = "depth interval ratio") # TODO: what
parser.add_argument('--loss_type', type = str, default = "casmvs", choices = ["casmvs", "entropy", "casmvs_entropy", "casmvs_dds", "entropy_dds"], help = 'training loss type')
parser.add_argument('--dlossw', type = str, default = "0.5,1.0,2.0", help = 'depth loss weight for each stage')
parser.add_argument('--elossw', type = str, default = "0.5,1.0,2.0", help = 'entropy loss weight for each stage; defaults to dlossw when omitted')
parser.add_argument('--entropy_weight', type = float, default = 0.1, help = 'global entropy loss weight in casmvs_entropy mode')
parser.add_argument('--ddslw', type = str, default = "0.5,1.0,2.0", help = 'dds loss weight for each stage; defaults to dlossw when omitted')
parser.add_argument('--dds_weight', type = float, default = 0.05, help = 'global dds loss weight in casmvs_dds mode')
parser.add_argument('--dds_num_bins', type = int, default = 32, help = 'number of bins for soft-histogram dds loss')
parser.add_argument('--dds_sigma', type = float, default = 0.0, help = 'gaussian sigma for soft-histogram dds loss; <=0 means auto')
parser.add_argument('--cr_base_chs', type = str, default = "8,8,8", help = "cost volume regularization base channels")
parser.add_argument('--eta', action='store_true', help='use eta in cost volume')
parser.add_argument('--attn_temp', type = float, default = 1.0, help = 'attention temperature for eta')
parser.add_argument('--group_cor_dim', type = str, default = "8,4,4", help = 'group correlation dim for each stage when eta is enabled')
# training setting
parser.add_argument('--epochs', type = int, default = 30, help = 'number of epochs of training')
parser.add_argument('--lr', type = float, default = 0.001, help = 'learning rate')
parser.add_argument('--lrepochs', type = str, default = '10,12,14:2', help = 'epoch index to downscale lr and downscale rate')
parser.add_argument('--wd', type = float, default = 0.0, help = 'weight decay')
parser.add_argument('--summary_freq', type = int, default = 50, help = 'train/test tensorboard summary frequency (unit: iteration steps)')
parser.add_argument('--save_freq', type = int, default = 1, help = 'save checkpoint frequency')
parser.add_argument('--progress_mode', type = str, default = "tqdm", choices = ["tqdm", "log"], help = "progress display mode")
parser.add_argument('--progress_log_freq', type = int, default = 10, help = "log frequency in log progress mode (batches)")
parser.add_argument('--num_workers', type = int, default = 4, help = "dataloader worker count for prediction")
# others setting
parser.add_argument('--seed', type = int, default = 42, metavar = 'S', help = 'answer of universe')
parser.add_argument('--gpu_id', type = str, default = "0")

# parse arguments and check
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
if args.progress_log_freq <= 0:
    raise ValueError("--progress_log_freq must be > 0")
seed_everything(args.seed)

def parse_stage_ints(value_str, value_name):
    values = [int(value) for value in value_str.split(",") if value]
    expected_len = len([int(depth) for depth in args.ndepths.split(",") if depth])
    if len(values) != expected_len:
        raise ValueError(f"{value_name} length must match number of stages, get {len(values)} vs {expected_len}")
    return values

# get dataset path
if args.dataset_root is None:
    raise ValueError("--dataset_root must be specified to locate data.")
train_path = os.path.join(args.dataset_root, 'train')
test_path = os.path.join(args.dataset_root, 'test')
if not os.path.exists(test_path):
    print(f"test_path: {test_path} not found, use train_path: {train_path} to replace")
    test_path = train_path

# check mode, setup log dir and build logger
current_time_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
if args.mode == 'train': # train mode
    if args.resume:
        if args.train_loadckpt is None:
            raise ValueError("resume training need 'train_loadckpt', but get none")
        
        if os.path.isfile(args.train_loadckpt):
            cur_log_dir = os.path.dirname(args.train_loadckpt)
        else:
            cur_log_dir = args.train_loadckpt
    else:
        cur_log_dir = os.path.join(args.logdir, args.model, args.geo_model, current_time_str)
    
    if not os.path.exists(cur_log_dir): # in train mode, we may make empty dir
        os.makedirs(cur_log_dir)
    logger = SummaryWriter(cur_log_dir)
else: # test mode
    if args.test_loadckpt is None:
        raise ValueError("test need 'test_loadckpt', but get none")
    cur_log_dir = os.path.dirname(args.test_loadckpt) if os.path.isfile(args.test_loadckpt) else args.test_loadckpt
print(f"log directory: {cur_log_dir}")

# dataset and dataloader
mvsdataset = find_dataset(args.geo_model)
train_dataset = mvsdataset(
    train_path,
    "train",
    args.view_num,
    ref_view = args.ref_view,
    aux_mode = args.aux_channel
)
test_dataset = mvsdataset(
    test_path,
    "test",
    args.view_num,
    ref_view = args.ref_view,
    aux_mode = args.aux_channel
)
train_loader_generator = torch.Generator()
train_loader_generator.manual_seed(args.seed)
train_sampler = StatefulRandomSampler(train_dataset, seed = args.seed)
train_loader = DataLoader(
    train_dataset,
    args.batch_size,
    shuffle = False,
    sampler = train_sampler,
    num_workers = args.num_workers,
    pin_memory = True,
    drop_last = True,
    worker_init_fn = seed_worker,
    generator = train_loader_generator
)
test_loader = DataLoader(
    test_dataset,
    args.batch_size,
    shuffle = False,
    num_workers = args.num_workers,
    pin_memory = True,
    drop_last = False,
    worker_init_fn = seed_worker
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
        cr_base_chs = [int(ch) for ch in args.cr_base_chs.split(",") if ch],
        eta = args.eta,
        attn_temp = args.attn_temp,
        group_cor_dim = parse_stage_ints(args.group_cor_dim, "group_cor_dim"),
        aux_mode = args.aux_channel
    )
    print(f"use CascadeMVSNet model")
else:
    raise Exception(f"{args.model} has no implementation")

# move model to gpu
if args.mode in ["train", "test"]:
    model = nn.DataParallel(model)
model.cuda()

# set model loss & optimizer
if args.loss_type == "casmvs":
    model_loss = casmvs_loss
elif args.loss_type == "entropy":
    model_loss = entropy_loss
elif args.loss_type == "casmvs_entropy":
    model_loss = casmvs_entropy_loss
elif args.loss_type == "casmvs_dds":
    model_loss = casmvs_dds_loss
elif args.loss_type == "entropy_dds":
    model_loss = entropy_dds_loss
else:
    raise ValueError(f"unsupported loss_type: {args.loss_type}")
optimizer = optim.RMSprop(
    [{'params': model.parameters(), 'initial_lr': args.lr}],
    lr = args.lr,
    alpha = 0.9,
    weight_decay = args.wd
)

def parse_loss_weights(weight_str, weight_name):
    if weight_str is None:
        return None
    weights = [float(weight) for weight in weight_str.split(",") if weight]
    expected_len = len([int(depth) for depth in args.ndepths.split(",") if depth])
    if len(weights) != expected_len:
        raise ValueError(f"{weight_name} length must match number of stages, get {len(weights)} vs {expected_len}")
    return weights

def get_loss_config():
    return {
        "loss_type": args.loss_type,
        "dlossw": parse_loss_weights(args.dlossw, "dlossw"),
        "elossw": parse_loss_weights(args.elossw, "elossw"),
        "entropy_weight": args.entropy_weight,
        "ddslw": parse_loss_weights(args.ddslw, "ddslw"),
        "dds_weight": args.dds_weight,
        "dds_num_bins": args.dds_num_bins,
        "dds_sigma": None if args.dds_sigma <= 0 else args.dds_sigma,
    }

def get_model_info(model):
    model_core = model.module if isinstance(model, nn.DataParallel) else model
    total_params = sum(parameter.numel() for parameter in model_core.parameters())
    trainable_params = sum(
        parameter.numel() for parameter in model_core.parameters() if parameter.requires_grad
    )
    return {
        "model_name": model_core.__class__.__name__,
        "total_params": total_params,
        "trainable_params": trainable_params,
    }

def format_run_header():
    loss_config = get_loss_config()
    model_info = get_model_info(model)
    optimizer_name = optimizer.__class__.__name__
    current_lr = optimizer.param_groups[0]["lr"] if len(optimizer.param_groups) > 0 else None

    lines = [
        "================ Run Start ================",
        f"time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"logdir: {cur_log_dir}",
        f"mode: {args.mode}",
        f"model: {args.model}",
        f"geo_model: {args.geo_model}",
        f"dataset_root: {args.dataset_root}",
        f"train_path: {train_path}",
        f"test_path: {test_path}",
        f"resume: {args.resume}",
        f"checkpoint: {args.train_loadckpt if args.resume else args.test_loadckpt if args.mode == 'test' else None}",
        "",
        "model_info:",
        f"  model_name: {model_info['model_name']}",
        f"  total_params: {model_info['total_params']}",
        f"  trainable_params: {model_info['trainable_params']}",
        "",
        "loss_info:",
        f"  loss_type: {loss_config['loss_type']}",
        f"  dlossw: {loss_config['dlossw']}",
        f"  elossw: {loss_config['elossw']}",
        f"  entropy_weight: {loss_config['entropy_weight']}",
        f"  ddslw: {loss_config['ddslw']}",
        f"  dds_weight: {loss_config['dds_weight']}",
        f"  dds_num_bins: {loss_config['dds_num_bins']}",
        f"  dds_sigma: {loss_config['dds_sigma']}",
        "",
        "optimizer_info:",
        f"  optimizer: {optimizer_name}",
        f"  lr: {current_lr}",
        f"  wd: {args.wd}",
        f"  lrepochs: {args.lrepochs}",
        "",
        "train_setup:",
        f"  batch_size: {args.batch_size}",
        f"  view_num: {args.view_num}",
        f"  ref_view: {args.ref_view}",
        f"  aux_channel: {args.aux_channel}",
        f"  ndepths: {[int(depth) for depth in args.ndepths.split(',') if depth]}",
        f"  min_interval: {args.min_interval}",
        f"  depth_inter_ratio: {[float(interval) for interval in args.depth_inter_ratio.split(',') if interval]}",
        f"  cr_base_chs: {[int(ch) for ch in args.cr_base_chs.split(',') if ch]}",
        f"  eta: {args.eta}",
        f"  attn_temp: {args.attn_temp}",
        f"  group_cor_dim: {parse_stage_ints(args.group_cor_dim, 'group_cor_dim')}",
        f"  epochs: {args.epochs}",
        f"  seed: {args.seed}",
        f"  gpu_id: {args.gpu_id}",
        "===========================================",
        "",
    ]
    return "\n".join(lines)

# load checkpoint if needed
start_epoch = 1
global_step = 0
resume_step_in_epoch = 0
resume_scheduler_state = None
resume_sampler_state = None
if(args.mode == "train" and args.resume):
    print(f"load checkpoint from {args.train_loadckpt}")
    loadckpt = args.train_loadckpt
    if os.path.isdir(loadckpt):
        saved_models = [file for file in os.listdir(loadckpt) if file.endswith(".ckpt")]
        if not saved_models:
            raise FileNotFoundError(f"no checkpoint file found in directory: {loadckpt}")
        saved_models = sorted(saved_models, key=lambda x: int(x.split('_')[1].split(".")[0]))
        loadckpt = os.path.join(loadckpt, saved_models[-1])
    elif not os.path.isfile(loadckpt):
        raise FileNotFoundError(f"checkpoint path does not exist: {loadckpt}")

    print(f"load checkpoint from {loadckpt} for resume training")
    state_dict = torch.load(loadckpt, map_location = "cpu")
    require_ckpt_keys(
        state_dict,
        [
            "epoch",
            "global_step",
            "step_in_epoch",
            "model",
            "optimizer",
            "scheduler",
            "sampler_state",
            "rng_torch",
            "rng_cuda",
            "rng_numpy",
            "rng_python",
            "data_generator_state"
        ],
        loadckpt
    )
    model.load_state_dict(state_dict['model'])
    optimizer.load_state_dict(state_dict['optimizer'])
    resume_step_in_epoch = int(state_dict['step_in_epoch'])
    if resume_step_in_epoch > 0:
        start_epoch = int(state_dict['epoch'])
    else:
        start_epoch = int(state_dict['epoch']) + 1
    global_step = state_dict['global_step']
    resume_scheduler_state = state_dict['scheduler']
    resume_sampler_state = state_dict['sampler_state']
    train_loader_generator.set_state(state_dict["data_generator_state"])
    restore_rng_states(state_dict)
elif args.mode == "test":
    assert args.test_loadckpt is not None, f"test need 'test_loadckpt', but get none"
    print(f"load checkpoint from {args.test_loadckpt}")
    
    loadckpt = args.test_loadckpt
    if os.path.isdir(loadckpt):
        saved_models = [file for file in os.listdir(loadckpt) if file.endswith(".ckpt")]
        if not saved_models:
            raise FileNotFoundError(f"no checkpoint file found in directory: {loadckpt}")
        saved_models = sorted(saved_models, key=lambda x: int(x.split('_')[1].split(".")[0]))
        loadckpt = os.path.join(loadckpt, saved_models[-1])
    elif not os.path.isfile(loadckpt):
        raise FileNotFoundError(f"checkpoint path does not exist: {loadckpt}")
    print(f"load checkpoint from {loadckpt} for test")
    state_dict = torch.load(loadckpt, map_location = "cpu")
    model.load_state_dict(state_dict['model'])

# train / test on one batch
def train_batch(sample, detailed_summary = False):
    model.train()
    optimizer.zero_grad()

    # run multi-stage mvs pipeline
    sample_cuda = tocuda(sample)
    depth_gt_ms = sample_cuda["depth"]
    mask_ms = sample_cuda["mask"]
    outputs = model(
        sample_cuda["images"],
        sample_cuda["cameras_para"],
        sample_cuda["depth_values"]
    )

    # get loss and backward
    loss_config = get_loss_config()
    loss_outputs = model_loss(
        outputs,
        depth_gt_ms,
        mask_ms,
        dlossw = loss_config["dlossw"],
        elossw = loss_config["elossw"],
        entropy_weight = loss_config["entropy_weight"],
        ddslw = loss_config["ddslw"],
        dds_weight = loss_config["dds_weight"],
        dds_num_bins = loss_config["dds_num_bins"],
        dds_sigma = loss_config["dds_sigma"],
        depth_values = sample_cuda["depth_values"]
    )
    scalar_outputs = build_loss_scalar_outputs(
        loss_outputs,
        loss_config["loss_type"],
        entropy_weight = loss_config["entropy_weight"],
        dds_weight = loss_config["dds_weight"]
    )
    loss = scalar_outputs["loss"]
    loss.backward()
    # TODO: each batch or each loop?
    optimizer.step()
    # # Step scheduler per batch to match iteration-level LR decay behavior.
    # lr_scheduler.step()

    # get metrices
    num_stage = len([int(depth) for depth in args.ndepths.split(",") if depth])
    depth_est = outputs[f"stage{num_stage}"]["depth"]
    photometric_confidence = outputs[f"stage{num_stage}"]["photometric_confidence"]
    depth_final_gt = depth_gt_ms[f"stage{num_stage}"]
    mask_final = mask_ms[f"stage{num_stage}"]
    
    image_outputs = {
        "depth_est": depth_est,
        "depth_gt": depth_final_gt,
        "ref_image": sample_cuda["images"][:, 0],
        "mask": mask_final
    }

    # summary if needed
    if detailed_summary:
        image_outputs["errormap"] = (depth_est - depth_final_gt).abs()
        scalar_outputs["abs_depth_error"] = AbsDepthError_metrics(depth_est, depth_final_gt, mask_final > 0.5, 250.0)
        scalar_outputs["mae"] = MAE_metrics(depth_est, depth_final_gt, mask_final > 0.5)
        scalar_outputs["rmse"] = RMSE_metrics(depth_est, depth_final_gt, mask_final > 0.5)
        scalar_outputs["threshold_1.0m_acc"] = Threshold_metrics(depth_est, depth_final_gt, mask_final > 0.5, 1.0)
        scalar_outputs["threshold_2.5m_acc"] = Threshold_metrics(depth_est, depth_final_gt, mask_final > 0.5, 2.5)
        scalar_outputs["threshold_7.5m_acc"] = Threshold_metrics(depth_est, depth_final_gt, mask_final > 0.5, 7.5)
        scalar_outputs["completeness"] = Completeness_metrics(photometric_confidence, depth_final_gt, mask_final > 0.5)
    
    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs

@make_nograd_func
def test_batch(sample, detailed_summary = False):
    model.eval()

    # run multi-stage mvs pipeline
    sample_cuda = tocuda(sample)
    depth_gt_ms = sample_cuda["depth"]
    mask_ms = sample_cuda["mask"]
    outputs = model(
        sample_cuda["images"],
        sample_cuda["cameras_para"],
        sample_cuda["depth_values"]
    )

    # get metrices
    num_stage = len([int(depth) for depth in args.ndepths.split(",") if depth])
    depth_final_gt = depth_gt_ms["stage{}".format(num_stage)]
    mask_final = mask_ms["stage{}".format(num_stage)]
    depth_est = outputs[f"stage{num_stage}"]["depth"]
    photometric_confidence = outputs[f"stage{num_stage}"]["photometric_confidence"]

    # get loss
    loss_config = get_loss_config()
    loss_outputs = model_loss(
        outputs,
        depth_gt_ms,
        mask_ms,
        dlossw = loss_config["dlossw"],
        elossw = loss_config["elossw"],
        entropy_weight = loss_config["entropy_weight"],
        ddslw = loss_config["ddslw"],
        dds_weight = loss_config["dds_weight"],
        dds_num_bins = loss_config["dds_num_bins"],
        dds_sigma = loss_config["dds_sigma"],
        depth_values = sample_cuda["depth_values"]
    )
    scalar_outputs = build_loss_scalar_outputs(
        loss_outputs,
        loss_config["loss_type"],
        entropy_weight = loss_config["entropy_weight"],
        dds_weight = loss_config["dds_weight"]
    )
    loss = scalar_outputs["loss"]

    # wrap outputs
    image_outputs = {
        "depth_est": depth_est,
        "photometric_confidence": photometric_confidence,
        "depth_gt": depth_final_gt,
        "ref_image": sample_cuda["images"][:, 0],
        "mask": mask_final
    }

    # summary if needed
    if detailed_summary:
        image_outputs["errormap"] = (depth_est - depth_final_gt).abs()
        scalar_outputs["abs_depth_error"] = AbsDepthError_metrics(depth_est, depth_final_gt, mask_final > 0.5, 250.0)
        scalar_outputs["mae"] = MAE_metrics(depth_est, depth_final_gt, mask_final > 0.5)
        scalar_outputs["rmse"] = RMSE_metrics(depth_est, depth_final_gt, mask_final > 0.5)
        scalar_outputs["threshold_1.0m_acc"] = Threshold_metrics(depth_est, depth_final_gt, mask_final > 0.5, 1.0)
        scalar_outputs["threshold_2.5m_acc"] = Threshold_metrics(depth_est, depth_final_gt, mask_final > 0.5, 2.5)
        scalar_outputs["threshold_7.5m_acc"] = Threshold_metrics(depth_est, depth_final_gt, mask_final > 0.5, 7.5)
        scalar_outputs["completeness"] = Completeness_metrics(photometric_confidence, depth_final_gt, mask_final > 0.5)
    
    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs

def train():
    milestones = [int(idx) for idx in args.lrepochs.split(':')[0].split(',')] # [10, 12, 14]
    lr_gamma = 1 / float(args.lrepochs.split(':')[1]) # 2
    lr_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones,
        lr_gamma,
        last_epoch = -1
    )
    if args.resume:
        lr_scheduler.load_state_dict(resume_scheduler_state)
        train_sampler.load_state_dict(resume_sampler_state)

    current_global_step = global_step
    record_path = os.path.join(cur_log_dir, 'train_record.txt')
    if not os.path.exists(record_path) or os.path.getsize(record_path) == 0:
        with open(record_path, "a+", encoding="utf-8") as f:
            f.write(format_run_header())

    # Include args.epochs itself so --epochs=N runs N epochs (1..N by default).
    for epoch_idx in range(start_epoch, args.epochs + 1):
        print(f"epoch {epoch_idx}")
        train_sampler.set_epoch(epoch_idx)

        step_in_epoch_start = 0
        if args.resume and epoch_idx == start_epoch and resume_step_in_epoch > 0:
            step_in_epoch_start = resume_step_in_epoch
            print(f"resume from epoch {epoch_idx}, step_in_epoch {step_in_epoch_start}")
        train_sampler.set_start_step(step_in_epoch_start, args.batch_size)

        # train
        use_tqdm = args.progress_mode == "tqdm"
        train_total_batch = max(len(train_loader) - step_in_epoch_start, 0)
        train_enum = enumerate(train_loader, start = step_in_epoch_start)
        train_iter = tqdm(train_enum, total = train_total_batch, desc = f"Train {epoch_idx}/{args.epochs}", unit = "batch") if use_tqdm else train_enum
        for batch_idx, sample in train_iter:
            start_time = time.perf_counter()
            current_global_step += 1
            bool_summary = current_global_step % args.summary_freq == 0
            
            loss, scalar_outputs, image_outputs = train_batch(
                sample,
                detailed_summary = bool_summary
            )
            
            torch.cuda.synchronize()
            batch_time = time.perf_counter() - start_time

            if bool_summary:
                save_scalars(logger, 'train', scalar_outputs, current_global_step)
                save_images(logger, 'train', image_outputs, current_global_step)
            if use_tqdm:
                train_iter.set_postfix(loss = f"{loss:.4f}", time = f"{batch_time:.3f}s")
                tqdm.write(f'epoch {epoch_idx}/{args.epochs}, iter {batch_idx}/{len(train_loader)}, train loss = {loss}, time = {batch_time}, train_result = {scalar_outputs}')
            else:
                is_last = (batch_idx + 1) == len(train_loader)
                if (batch_idx + 1) % args.progress_log_freq == 0 or batch_idx == 0 or is_last:
                    print(
                        f"[Train][{epoch_idx}/{args.epochs}] "
                        f"iter {batch_idx + 1}/{len(train_loader)}, "
                        f"loss={loss:.4f}, time={batch_time:.3f}s, metrics={scalar_outputs}"
                    )
            del scalar_outputs, image_outputs
    
        # test
        avg_test_scalars = DictAverageMeter()
        test_iter = tqdm(enumerate(test_loader), total = len(test_loader), desc = f"Test {epoch_idx}/{args.epochs}", unit = "batch") if use_tqdm else enumerate(test_loader)
        for batch_idx, sample in test_iter:
            start_time = time.perf_counter()
            eval_global_step = current_global_step + batch_idx + 1
            bool_summary = eval_global_step % args.summary_freq == 0

            loss, scalar_outputs, image_outputs = test_batch(
                sample,
                detailed_summary = True
            )

            torch.cuda.synchronize()
            batch_time = time.perf_counter() - start_time

            if bool_summary:
                save_scalars(logger, 'test', scalar_outputs, eval_global_step)
                save_images(logger, 'test', image_outputs, eval_global_step)
            # Use sample-count weighting so epoch metrics are invariant to batch size.
            batch_size_cur = int(sample["images"].shape[0])
            avg_test_scalars.update(scalar_outputs, weight = batch_size_cur)
            
            if use_tqdm:
                test_iter.set_postfix(loss = f"{loss:.4f}", time = f"{batch_time:.3f}s")
                tqdm.write(f'epoch {epoch_idx}/{args.epochs}, iter {batch_idx}/{len(test_loader)}, test loss = {loss}, time = {batch_time}, test_result = {scalar_outputs}')
            else:
                is_last = (batch_idx + 1) == len(test_loader)
                if (batch_idx + 1) % args.progress_log_freq == 0 or batch_idx == 0 or is_last:
                    print(
                        f"[Test][{epoch_idx}/{args.epochs}] "
                        f"iter {batch_idx + 1}/{len(test_loader)}, "
                        f"loss={loss:.4f}, time={batch_time:.3f}s, metrics={scalar_outputs}"
                    )
            del scalar_outputs, image_outputs
        save_scalars(logger, 'fulltest', avg_test_scalars.mean(), current_global_step)
        print(f"avg_test_scalars: {avg_test_scalars.mean()}")

        # write log
        current_avg_metrics = avg_test_scalars.mean()
        with open(record_path, "a+", encoding="utf-8") as f:
            f.write(f"Epoch {epoch_idx} [{args.loss_type}]: {current_avg_metrics}\n")

        # Keep epoch-level semantics for lrepoch milestones.
        lr_scheduler.step()

        # save checkpoint
        if (epoch_idx + 1) % args.save_freq == 0:
            rng_states = capture_rng_states()
            torch.save(
                {
                    'epoch': epoch_idx,
                    'global_step': current_global_step,
                    'step_in_epoch': 0,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': lr_scheduler.state_dict(),
                    'sampler_state': train_sampler.state_dict(),
                    'rng_torch': rng_states['rng_torch'],
                    'rng_cuda': rng_states['rng_cuda'],
                    'rng_numpy': rng_states['rng_numpy'],
                    'rng_python': rng_states['rng_python'],
                    'data_generator_state': train_loader_generator.get_state()
                },
                f"{cur_log_dir}/model_{epoch_idx:0>6}.ckpt"
            )

# main
if __name__ == '__main__':
    if args.mode == "train":
        train()
