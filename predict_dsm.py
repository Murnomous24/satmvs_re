import os
import argparse
from tools.utils import read_config, mkdir_if_not_exist, read_from_txt, read_pair_from_txt, read_np_array_from_txt
from dsm_pipeline import Pipeline

parser = argparse.ArgumentParser(description='satmvs_re predicting dsm file')

# infomation for input data
parser.add_argument("--config_file", default="dsm_config/config.json")
parser.add_argument("--info_root", default="dsm_infos/whu_tlc")

# model
parser.add_argument('--model', default = 'casmvs', help = 'select model', choices = ['casmvs'])
parser.add_argument('--geo_model', default = 'rpc', help = 'select dataset format', choices = ['pinhole', 'rpc'])
parser.add_argument('--loadckpt', default = '/home/murph_dl/Paper_Re/train_log/26_1_31_23_42/model_000005.ckpt', help = 'specific checkpoint file for prediction')

# load data parameters
parser.add_argument('--resize_scale', type=float, default=1, help='output scale for depth and image (W and H)')
parser.add_argument('--sample_scale', type=float, default=1, help='Downsample scale for building cost volume (W and H)')
parser.add_argument('--interval_scale', type=float, default=1, help='the number of depth values')
parser.add_argument('--batch_size', type=int, default=1, help='predict batch size')
parser.add_argument('--adaptive_scaling', type=bool, default=True,
                    help='Let image size to fit the network, including scaling and cropping')

# Cascade parameters
parser.add_argument('--share_cr', action='store_true', help='whether share the cost volume regularization')
parser.add_argument('--ndepths', type=str, default="64,32,8", help='ndepths')
parser.add_argument('--depth_inter_ratio', type=str, default="4,2,1", help='depth_intervals_ratio')
parser.add_argument('--cr_base_chs', type=str, default="8,8,8", help='cost regularization base channels')

# others setting
parser.add_argument('--gpu_id', type = str, default = "0")

# output
parser.add_argument("--workspace", type=str, default="./dsm_results")

# parse arguments and check
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

def read_project(project_info_file):
    with open(project_info_file, "r") as f:
        project_str = f.read()
    return project_str

def read_pair(images_info_file, cameras_info_file, pairs_info_file):
    image_paths = read_from_txt(images_info_file)
    camera_paths = read_from_txt(cameras_info_file)
    pair_info = read_pair_from_txt(pairs_info_file)

    image_pairs = [
        [image_paths[int(view_idx)] for view_idx in pair_info[idx]] for idx in range(len(pair_info))
    ]
    camera_pairs = [
        [camera_paths[int(view_idx)] for view_idx in pair_info[idx]] for idx in range(len(pair_info))
    ]
    idx_pairs = [[int(view_idx) for view_idx in pair_info[idx]] for idx in range(len(pair_info))]

    return image_pairs, camera_pairs, idx_pairs

def read_border(border_info_file):
    return read_np_array_from_txt(border_info_file)

def read_depth_range(depth_range_info_file):
    return read_np_array_from_txt(depth_range_info_file)

if __name__ == "__main__":
    config = read_config(args.config_file)
    workspace = args.workspace
    mkdir_if_not_exist(workspace)
    print(f"Workspace: {workspace}")

    # per-scene
    scenes = sorted(os.listdir(args.info_root))
    for scene in scenes:
        scene_path = f"{args.info_root}/{scene}"
        print(f"Scene start: {scene}")
        
        # get file path
        project_info_file = f"{scene_path}/WGS 1984 UTM Zone  8N.prj"
        images_info_file = f"{scene_path}/images_info.txt"
        cameras_info_file = f"{scene_path}/cameras_info.txt"
        pairs_info_file = f"{scene_path}/pair.txt"
        border_info_file = f"{scene_path}/border.txt"
        depth_range_file = f"{scene_path}/range.txt"

        # read file content
        project_str = read_project(project_info_file)

        image_pair_list, camera_pair_list, index_pair_list = read_pair(
            images_info_file,
            cameras_info_file,
            pairs_info_file
        )

        border = read_border(border_info_file)
        depth_range = read_depth_range(depth_range_file)
        print(f"Scene metadata loaded: {scene}")

        # make output folder
        pair_workspace = f"{workspace}/{scene}"
        mkdir_if_not_exist(pair_workspace)
    
        # build dsm
        for image_path, camera_path, view_idx in zip(image_pair_list, camera_pair_list, index_pair_list):
            # make dsm output folder
            out_name = ""
            for idx in view_idx:
                out_name += str(idx) + "_"
            out_name = out_name[:-1]

            output_path = os.path.join(pair_workspace, out_name)
            mkdir_if_not_exist(output_path)
            print(f"Pair start: {scene}/{out_name}")

            # run predict and build dsm output
            dsm_pipeline = Pipeline(
                config,
                image_path,
                camera_path,
                project_str,
                border,
                depth_range,
                output_path,
                args
            )
            dsm_pipeline.run()
            print(f"Pair done: {scene}/{out_name}")

        print(f"Scene done: {scene}")

        