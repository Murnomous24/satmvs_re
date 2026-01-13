import argparse
import datetime
import os
import time
import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

from tools.utils import *
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
parser.add_argument('--resume', default = False, help = 'continue to train model in old checkpoint')
parser.add_argument('--batch_size', type = int, default = 1, help = "batch size")
# mvs setting
parser.add_argument('--view_num', type = int, default = 3, help = 'number of input view')
parser.add_argument('--ref_view', type = int, default = 2, help = 'index of reference view')
# cascade setting
parser.add_argument('--ndepths', type = str, default = "64,32,8", help = "number of depths")
parser.add_argument('--min_interval', type = float, default = 2.5, help = "min interval of each depth plane")
parser.add_argument('--depth_inter_ratio', type = str, default = "4,2,1", help = "depth interval ratio") # TODO: what
parser.add_argument('--dlossw', type = str, default = "0.5,1.0,2.0", help = 'depth loss weight for each stage')
parser.add_argument('--cr_base_chs', type = str, default = "8,8,8", help = "cost volume regularization base channels")
# training setting
parser.add_argument('--epochs', type = int, default = 30, help = 'number of epochs of training')
parser.add_argument('--lr', type = float, default = 0.001, help = 'learning rate')
parser.add_argument('--lrepochs', type = str, default = '10,12,14:2', help = 'epoch index to downscale lr and downscale rate')
parser.add_argument('--wd', type = float, default = 0.0, help = 'weight decay')
parser.add_argument('--summary_freq', type = int, default = 50, help = 'print and save log frequency')
parser.add_argument('--save_freq', type = int, default = 1, help = 'save checkpoint frequency')
# others setting
parser.add_argument('--seed', type = int, default = 42, metavar = 'S', help = 'answer of universe')
parser.add_argument('--gpu_id', type = str, default = "0")

# parse arguments and check
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

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
    ref_view = args.ref_view
)
test_dataset = mvsdataset(
    test_path,
    "test",
    args.view_num,
    ref_view = args.ref_view
)
train_loader = DataLoader(
    train_dataset,
    args.batch_size,
    shuffle = True,
    num_workers = 0,
    drop_last = True
)
test_loader = DataLoader(
    test_dataset,
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
if args.mode in ["train", "test"]:
    model = nn.DataParallel(model)
model.cuda()

# set model loss & optimizer
model_loss = casmvs_loss
optimizer = optim.RMSprop(
    [{'params': model.parameters(), 'initial_lr': args.lr}],
    lr = args.lr,
    alpha = 0.9,
    weight_decay = args.wd
)

# load checkpoint if needed
start_epoch = 1
if(args.mode == "train" and args.resume):
    print(f"load checkpoint from {args.train_loadckpt}")
    loadckpt = args.train_loadckpt
    if os.path.isdir(loadckpt):
        saved_models = [file for file in os.listdir(loadckpt) if file.endswith(".ckpt")]
        saved_models = sorted(saved_models, key=lambda x: int(x.split('_')[1].split(".")[0]))
        loadckpt = os.path.join(loadckpt, saved_models[-1])

    print(f"load checkpoint from {loadckpt} for resume training")
    state_dict = torch.load(loadckpt)
    model.load_state_dict(state_dict['model'])
    optimizer.load_state_dict(state_dict['optimizer'])
    start_epoch = state_dict['epoch'] + 1
elif args.mode == "test":
    assert args.test_loadckpt is not None, f"test need 'test_loadckpt', but get none"
    print(f"load checkpoint from {args.test_loadckpt}")
    
    loadckpt = args.test_loadckpt
    print(f"load checkpoint from {loadckpt} for test")
    state_dict = torch.load(loadckpt)
    model.load_state_dict(state_dict['model'])

# before training, set seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# train / test on one batch
def train_batch(sample, lr_scheduler, detailed_summary = False):
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
    loss, depth_loss = model_loss(
        outputs,
        depth_gt_ms,
        mask_ms,
        dlossw = [float(weight) for weight in args.dlossw.split(",") if weight]
    )
    loss.backward()
    # TODO: each batch or each loop?
    optimizer.step()
    lr_scheduler.step()


    # get metrices
    num_stage = len([int(depth) for depth in args.ndepths.split(",") if depth])
    depth_est = outputs[f"stage{num_stage}"]["depth"]
    depth_final_gt = depth_gt_ms[f"stage{num_stage}"]
    mask_final = mask_ms[f"stage{num_stage}"]
    scalar_outputs = {
        "loss": loss,
        "depth_loss": depth_loss
    }
    image_outputs = {
        "depth_est": depth_est,
        "depth_gt": sample["depth"]["stage1"], # TODO: why stage1
        "ref_image": sample["images"][:, 0],
        "mask": sample["mask"]["stage1"]  # TODO: why stage1
    }

    # summary if needed
    if detailed_summary:
        image_outputs["errormap"] = (depth_est - depth_final_gt).abs() * mask_final
        scalar_outputs["abs_depth_error"] = AbsDepthError_metrics(depth_est, depth_final_gt, mask_final > 0.5, 250.0)
        scalar_outputs["threshold_1.0m_error"] = Threshold_metrics(depth_est, depth_final_gt, mask_final > 0.5, 1.0)
        scalar_outputs["threshold_2.5m_error"] = Threshold_metrics(depth_est, depth_final_gt, mask_final > 0.5, 2.5)
        scalar_outputs["threshold_7.5m_error"] = Threshold_metrics(depth_est, depth_final_gt, mask_final > 0.5, 7.5)
    
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
    loss, depth_loss = model_loss(
        outputs,
        depth_gt_ms,
        mask_ms,
        dlossw = [float(weight) for weight in args.dlossw.split(",") if weight]
    )

    # wrap outputs
    scalar_outputs = {
        "loss": loss,
        "depth_loss": depth_loss
    }
    image_outputs = {
        "depth_est": depth_est,
        "photometric_confidence": photometric_confidence,
        "depth_gt": sample["depth"]["stage1"],  # TODO: why stage1
        "ref_image": sample["images"][:, 0],
        "mask": sample["mask"]["stage1"]  # TODO: why stage1
    }

    # summary if needed
    if detailed_summary:
        image_outputs["errormap"] = (depth_est - depth_final_gt).abs() * mask_final
        scalar_outputs["abs_depth_error"] = AbsDepthError_metrics(depth_est, depth_final_gt, mask_final > 0.5, 250.0)
        scalar_outputs["threshold_1.0m_error"] = Threshold_metrics(depth_est, depth_final_gt, mask_final > 0.5, 1.0)
        scalar_outputs["threshold_2.5m_error"] = Threshold_metrics(depth_est, depth_final_gt, mask_final > 0.5, 2.5)
        scalar_outputs["threshold_7.5m_error"] = Threshold_metrics(depth_est, depth_final_gt, mask_final > 0.5, 7.5)
    
    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs

def train():
    milestones = [int(idx) for idx in args.lrepochs.split(':')[0].split(',')] # [10, 12, 14]
    lr_gamma = 1 / float(args.lrepochs.split(':')[1]) # 2
    lr_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones,
        lr_gamma,
        last_epoch = start_epoch - 1
    )

    for epoch_idx in range(start_epoch, args.epochs):
        print(f"epoch {epoch_idx}")

        global_step = len(train_loader) * epoch_idx # batch * epoch TODO right ?

        # train
        for batch_idx, sample in enumerate(train_loader):
            start_time = time.time()
            global_step = len(train_loader) * epoch_idx + batch_idx
            bool_summary = global_step % args.summary_freq == 0
            
            loss, scalar_outputs, image_outputs = train_batch(
                sample,
                lr_scheduler,
                detailed_summary = bool_summary
            )
            
            if bool_summary:
                save_scalars(logger, 'train', scalar_outputs, global_step)
                save_images(logger, 'train', image_outputs, global_step)
            print(f'epoch {epoch_idx}/{args.epochs}, iter {batch_idx}/{len(train_loader)}, train loss = {loss}, time = {time.time() - start_time}, train_result = {scalar_outputs}')
            del scalar_outputs, image_outputs
    
        # test
        avg_test_scalars = DictAverageMeter()
        for batch_idx, sample in enumerate(test_loader):
            start_time = time.time()
            global_step = len(train_loader) * epoch_idx + batch_idx
            bool_summary = global_step % args.summary_freq == 0

            loss, scalar_outputs, image_outputs = test_batch(
                sample,
                detailed_summary = bool_summary
            )

            if bool_summary:
                save_scalars(logger, 'test', scalar_outputs, global_step)
                save_images(logger, 'test', image_outputs, global_step)
            avg_test_scalars.update(scalar_outputs) # update scalars
            
            print(f'epoch {epoch_idx}/{args.epochs}, iter {batch_idx}/{len(train_loader)}, test loss = {loss}, time = {time.time() - start_time}, train_result = {scalar_outputs}')
            del scalar_outputs, image_outputs
        save_scalars(logger, 'fulltest', avg_test_scalars.mean(), global_step)
        print(f"avg_test_scalars: {avg_test_scalars.mean()}")

        # write log
        record_path = os.path.join(cur_log_dir, 'train_record.txt')
        current_avg_metrics = avg_test_scalars.mean()
        with open(record_path, "a+", encoding="utf-8") as f:
            f.write(f"Epoch {epoch_idx}: {current_avg_metrics}\n")

        # save checkpoint
        if (epoch_idx + 1) % args.save_freq == 0:
            torch.save(
                {
                    'epoch': epoch_idx,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                },
                f"{cur_log_dir}/model_{epoch_idx:0>6}.ckpt"
            )

# main
if __name__ == '__main__':
    if args.mode == "train":
        train()
