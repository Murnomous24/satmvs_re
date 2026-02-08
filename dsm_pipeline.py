import copy
import os
import numpy as np
from predict import predict
from tools.utils import get_image_size
from dataset.rpc_model import RPCModel
from tools.utils_utm_projection import Projection
from tools.utils_geo import calculate_block_properties

class Pipeline:
    def __init__(
            self,
            config,
            image_path,
            camera_path,
            project_str,
            border,
            depth_range,
            output_path,
            args
    ):
        self.args = args
        
        # config
        self.run_crop_img = config["run_crop_img"]
        self.run_mvs = config["run_mvs"]
        self.run_generate_points = config["run_generate_points"]
        self.run_generate_dsm = config["run_generate_dsm"]

        # block setting
        self.config_block_x_size = config["block_size_x"]
        self.config_block_y_size = config["block_size_y"]
        self.x_overlap = 1 - config["overlap_x"] # TODO: whattt
        self.y_overlap = 1 - config["overlap_y"] # TODO: whattt
        self.para = config["para"] # TODO: whatt
        self.invalid = config["invalid_value"] # TODO: whatt

        # TODO: what
        self.p_thred = config["position_threshold"] # TODO: whatt
        self.d_thred = config["depth_threshold"] # TODO: whatt
        self.geo_num = config["geometric_num"] # TODO: whatt

        # data
        self.image_path = image_path
        self.rpc_path = camera_path
        self.depth_range = depth_range
        self.view_num = len(self.image_path)
        self.image_size = [get_image_size(path) for path in self.image_path]

        # rpc config
        self.rpcs = []
        idx = 0
        for path in self.rpc_path:
            rpc = RPCModel() # TODO: rpc class implement
            rpc.load_rpc_from_file(path)
            rpc.check(self.image_size[idx][0], self.image_size[idx][1], 100, 30) # TODO: ?
            self.rpcs.append(copy.deepcopy(rpc))
            idx += 1

            if depth_range[0] == 0 and depth_range[1] == 0 and idx == 1: # get depth range from rpc
                depth_range[0], depth_range[1] = rpc.get_height_max_min()

        # projection
        self.projection = Projection(project_str) # TODO: project

        # dsm config
        self.x_unit, self.y_unit = border[4], border[5]
        self.border = border[:4]
        self.dsm_x_size = self.border[2]
        self.dsm_y_size = self.border[3]

        # block config
        # no blocking
        if self.config_block_x_size >= self.dsm_x_size or self.config_block_y_size >= self.dsm_y_size:
            self.jump_crop = [1]

            self.block_x_size, self.block_y_size = self.image_size[0][0], self.image_size[0][1]
            self.image_x_center = np.array([[int(self.block_x_size / 2)] for idx in range(self.view_num)], int) 
            self.image_y_center = np.array([[int(self.block_y_size / 2)] for idx in range(self.view_num)], int)

            self.x_grid_start = 0
            self.y_grid_start = 0

            self.x_dsm_start = [self.x_grid_start * self.x_unit + self.border[0]]
            self.y_dsm_start = [self.y_grid_start * self.y_unit + self.border[1]]
            self.x_dsm_end = [self.dsm_x_size * self.x_unit + self.border[0]]
            self.y_dsm_end = [-self.dsm_y_size * self.y_unit + self.border[1]] # why minus?

            self.block_num = 1
            # TODO change config_block_size ?
        # block
        else:
             self.jump_crop = False
             self.block_x_center = np.zeros(1)
             self.block_y_center = np.zeros(1) 

             self.x_grid_start = np.zeros(1) 
             self.y_grid_start = np.zeros(1)

             self.x_dsm_start = np.zeros(1)
             self.y_dsm_start = np.zeros(1)
             self.x_dsm_end = np.zeros(1)
             self.y_dsm_end = np.zeros(1)

             self.block_num = 0

             # calculate these properties
             calculate_block_properties()

        # output path
        self.output_path = output_path
        self.output_image_path = os.path.join(self.output_path, "image")
        self.output_rpc_path = os.path.join(self.output_path, "rpc")
        self.output_height_path = os.path.join(self.output_path, "height")
        self.output_points_path = os.path.join(self.output_path, "points")
        self.output_dsm_path = os.path.join(self.output_path, "dsm")

        # output per-view folder
        self.output_image_paths = [os.path.join(self.output_image_path, f"{idx}") for idx in range(self.view_num)]
        self.output_rpc_paths = [os.path.join(self.output_rpc_path, f"{idx}") for idx in range(self.view_num)]
        self.output_height_paths = [os.path.join(self.output_height_path, f"{idx}") for idx in range(self.view_num)]

    # crop fullsize image into pieces
    def crop_image(self, block_idx):
        if self.jump_crop:
            return
    
        for view_idx in range(self.view_num):
            # read image block
            image_x_start = self.image_x_center[view_idx][block_idx] - int(self.block_x_size / 2)
            image_y_start = self.image_y_center[view_idx][block_idx] - int(self.block_y_size / 2)
            image = read_image( # TODO: read block from fullsize image
                self.image_path[view_idx],
                int(image_x_start),
                int(image_y_start),
                int(self.block_x_size),
                int(self.block_y_size)
            ) # TODO: shape?
            image = image.transpose([1, 2, 0])

            # read rpc
            full_rpc = self.rpcs[view_idx]
            block_rpc = copy.deepcopy(full_rpc)
            block_rpc.SAMP_OFF -= int(image_x_start)
            block_rpc.LINE_OFF -= int(image_y_start)

            # save image, rpc
            out_name = "block{:0>4d}".format(block_idx)
            image_out_path = os.path.join(self.output_image_paths[view_idx], f"{out_name}.png")
            rpc_out_path = os.path.join(self.output_rpc_paths[view_idx], f"{out_name}.rpc")
            save_image(image_out_path, image) # TODO: save block from fullsize image
            save_rpc(rpc_out_path, block_rpc) # TODO: save rpc information

    def run(self):
        self.create_output_folder()
        
        # crop fullsize image to model input size
        if self.run_crop_img:
            for idx in range(self.block_num):
                self.crop_image(idx)
        
        # mvs pipeline, get the height map
        if self.run_mvs:
            predict(
                self.output_path,
                self.depth_range,
                self.view_num,
                self.args
            )
        
        # build point cloud output
        if self.run_generate_points:
            points = []
            for idx in range(self.block_num):
                point = self.generate_points(idx) # TODO: generate point clouds from depth map
                points.append(point)

            points = np.concatenate(points, axis = 0)
            points_output_path = os.path.join(self.output_points_path, "points.las")
            write_point_cloud(points_output_path, points) # TODO: save point clouds

        # build dsm output
        if self.run_generate_dsm:
            self.generate_dsm() # TODO: generate dsm map