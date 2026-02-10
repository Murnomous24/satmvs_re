import copy
import os
import numpy as np
from predict import predict
from tools.utils import get_image_size, read_image, save_image, save_rpc, write_point_cloud, filter_depth, raster_create, read_point_cloud, build_dsm, write_dsm
from dataset.rpc_model import RPCModel
from dataset.data_io import load_pfm, load_rpc_as_array
from tools.utils_utm_projection import Projection

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
        self.x_overlap = 1 - config["overlap_x"] # neighbor block overlap
        self.y_overlap = 1 - config["overlap_y"]
        self.para = config["para"] # block size adjust
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
        self.projection = Projection(project_str)

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
            self.block_x_center = np.array([[int(self.block_x_size / 2)] for idx in range(self.view_num)], int) 
            self.block_y_center = np.array([[int(self.block_y_size / 2)] for idx in range(self.view_num)], int)

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
             self.jump_crop = np.ones(1)
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
             self.calculate_block_properties()

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

    # blocking properties calculation
    def calculate_block_properties(self):
        block_num_x = self.dsm_x_size / (self.block_x_size)
        block_num_y = self.dsm_y_size / (self.block_y_size)

        # block number adjust
        if abs(block_num_x - int(block_num_x)) < 0.0001:
            block_num_x = int(block_num_x)
        else:
            block_num_x = int(block_num_x + 1)
        if abs(block_num_y - int(block_num_y)) < 0.0001:
            block_num_y = int(block_num_y)
        else:
            block_num_y = int(block_num_y + 1)

        # build grid start and end(NOTE: all of them are array!)
        self.x_grid_start = (np.arange(block_num_x) * block_x_size * x_overlap).astype(int)
        x_grid_end = self.x_grid_start + block_x_size
        self.y_grid_start = (np.arange(block_num_y) * block_y_size * y_overlap).astype(int)
        y_grid_end = self.y_grid_start + block_y_size

        # adjust grid avoid exceed border limit
        x_over_index = x_grid_end > self.dsm_x_size
        self.x_grid_start[x_over_index] = self.dsm_x_size - block_x_size
        x_grid_end = self.x_grid_start + block_x_size

        y_over_index = y_grid_end > self.dsm_y_size
        self.y_grid_start[y_over_index] = self.dsm_y_size - block_y_size
        y_grid_end = self.y_grid_start + block_y_size

        # build meshgrid, shape like [block_num_x, block_num_y]
        self.x_grid_start, self.y_grid_start = np.meshgrid(self.x_grid_start, self.y_grid_start) 
        x_grid_end, y_grid_end = np.meshgrid(x_grid_end, y_grid_end)
        
        # build one-dim vector, shape like [block_num_x * block_num_y]
        self.x_grid_start = self.x_grid_start.reshape(-1)
        self.y_grid_start = self.y_grid_start.reshape(-1)
        x_grid_end = x_grid_end.reshape(-1)
        y_grid_end = y_grid_end.reshape(-1)
        
        # grid -> dsm
        self.x_dsm_start = self.x_grid_start * self.x_unit + self.border[0]
        self.y_dsm_start = -self.y_grid_start * self.y_unit + self.border[1] # TODO: why minus, coordinate setup, y axis face to negative
        self.x_dsm_end = x_grid_end * self.x_unit + self.border[0]
        self.y_dsm_end = -y_grid_end * self.y_unit + self.border[1]
        
        # build per-block corner coordinate
        point_upper_left = np.stack((self.x_dsm_start, self.y_dsm_start), axis = -1)
        point_upper_right = np.stack((self.x_dsm_end, self.y_dsm_start), axis = -1)
        point_bottom_left = np.stack((self.x_dsm_start, self.y_dsm_end), axis = -1)
        point_bottom_right = np.stack((self.x_dsm_end, self.y_dsm_end), axis = -1)
        rects = np.stack((point_upper_left, point_bottom_right, point_upper_right, point_bottom_left)) # TODO: order?

        # plane rects -> geometry rects
        geo_rects = self.projection.project(rects, reverse = True) # [N, N, 2]
        lon_min = np.min(geo_rects[:, :, 0], axis = 0)
        lon_max = np.max(geo_rects[:, :, 0], axis = 0)
        lat_min = np.min(geo_rects[:, :, 1], axis = 0)
        lat_max = np.max(geo_rects[:, :, 1], axis = 0)
        height_temp = np.zeros(lon_min.shape, dtype = np.float64)
        
        # build oct bounding box
        block_x_start_pixel_list = []
        block_y_start_pixel_list = []
        block_x_end_pixel_list = []
        block_y_end_pixel_list = []
        block_x_size_list = []
        block_y_size_list = []
        for rpc in self.rpcs:
            h_min = height_temp + self.depth_range[0]
            h_max = height_temp + self.depth_range[1]

            oct_1 = np.stack((lat_min, lon_min, h_min), axis = -1)
            oct_2 = np.stack((lat_min, lon_max, h_min), axis = -1)
            oct_3 = np.stack((lat_max, lon_min, h_min), axis = -1)
            oct_4 = np.stack((lat_max, lon_max, h_min), axis = -1)
            oct_5 = np.stack((lat_min, lon_min, h_max), axis = -1)
            oct_6 = np.stack((lat_min, lon_max, h_max), axis = -1)
            oct_7 = np.stack((lat_max, lon_min, h_max), axis = -1)
            oct_8 = np.stack((lat_max, lon_max, h_max), axis = -1)
            octs = np.stack((oct_1, oct_2, oct_3, oct_4, oct_5, oct_6, oct_7, oct_8))
            octs = octs.reshape(-1, 3) # [N, 3]

            # rpc project to pixel
            samp, line = rpc.obj2photo(octs[:, 0], octs[:, 1], octs[:, 2])
            samp = samp.reshape((8, -1)) # get bounding box's points projection
            line = line.reshape((8, -1))
            
            # build block's pixel properties
            block_x_start_pixel = np.min(samp, axis = 0)
            block_y_start_pixel = np.min(line, axis = 0)
            block_x_end_pixel = np.max(samp, axis = 0)
            block_y_end_pixel = np.max(line, axis = 0)

            block_x_start_pixel_list.append(block_x_start_pixel)
            block_y_start_pixel_list.append(block_y_start_pixel)
            block_x_end_pixel_list.append(block_x_end_pixel)
            block_y_end_pixel_list.append(block_y_end_pixel)

            # block's size should align to others, cause network input's size must be const
            block_x_size = np.max(block_x_end_pixel - block_x_start_pixel)
            block_y_size = np.max(block_y_end_pixel - block_y_start_pixel)
            # celing
            block_x_size = int(block_x_size / self.para + 1) * self.para
            block_y_size = int(block_y_size / self.para + 1) * self.para
            
            block_x_size_list.append(block_x_size)
            block_y_size_list.append(block_y_size)
        
        block_x_start_pixel_list = np.array(block_x_start_pixel_list)
        block_y_start_pixel_list = np.array(block_y_start_pixel_list)
        block_x_end_pixel_list = np.array(block_x_end_pixel_list)
        block_y_end_pixel_list = np.array(block_y_end_pixel_list)

        # block center
        self.block_x_center = (block_x_start_pixel_list + block_x_end_pixel_list) / 2
        self.block_y_center = (block_y_start_pixel_list + block_y_end_pixel_list) / 2
        self.block_x_center = self.block_x_center.astype(int)
        self.block_y_center = self.block_y_center.astype(int)

        # block size & block range adjust
        block_x_size = np.max(block_x_size) # TODO, here?
        block_y_size = np.max(block_y_size)
        block_num = block_x_size.shape[0]

        self.block_adjust()
    
    # block adjustment(avoiding out of bound)
    # TODO: modify!!!!
    def block_adjust(self):
        # block rects
        self.block_x_start = (self.block_x_center - self.block_x_size / 2).astype(int)
        self.block_y_start = (self.block_y_center - self.block_y_size / 2).astype(int)
        self.block_x_end = (self.block_x_center + self.block_x_size / 2).astype(int)
        self.block_y_end = (self.block_y_center + self.block_y_size / 2).astype(int)

        temp_jump_list = []
        for idx in range(self.view_num):
            jump = np.ones(self.block_num, dtype = int)

            # start (remove condition)
            # upper-left out range
            x_start_remove_0 = (self.block_x_start[idx] <= -self.x_overlap_size * block_x_size)
            y_start_remove_0 = self.block_y_start[idx] <= -self.y_overlap_size * block_y_size
            # bottom-right out range
            x_start_remove_1 = self.block_x_start[idx] >= self.image_sizes[idx][0] - 1
            y_start_remove_1 = self.block_y_start[idx] >= self.image_sizes[idx][1] - 1

            # end (remove condtion)
            # upper-left out range
            x_end_remove_0 = self.block_x_end[idx] < 0
            y_end_remove_0 = self.block_y_end[idx] < 0
            # bottom-right out range
            x_end_remove_1 = self.block_x_end[idx] >= self.image_sizes[idx][0] - 1 + self.x_overlap_size * self.block_x_size
            y_end_remove_1 = self.block_y_end[idx] >= self.image_sizes[idx][1] - 1 + self.y_overlap_size * self.block_y_size

            # start (adjust)
            # upper-left
            x_start_adjust_0 = (self.block_x_start[idx] < 0) & (self.block_x_start[idx] > -self.x_overlap_size * self.block_x_size)
            y_start_adjust_0 = (self.block_y_start[idx] < 0) & (self.block_y_start[idx] > -self.y_overlap_size * self.block_y_size)
            # bottom-right
            x_end_adjust_0 = (self.block_x_end[idx] > self.image_sizes[idx][0] - 1) & (self.block_x_end[idx] < self.image_sizes[idx][0] - 1 + self.x_overlap_size * self.block_x_size)
            y_end_adjust_0 = (self.block_y_end[idx] > self.image_sizes[idx][1] - 1) & (self.block_y_end[idx] < self.image_sizes[idx][1] - 1 + self.y_overlap_size * self.block_y_size)

            # jump flag (remove part)
            jump[x_start_remove_0] = 0
            jump[y_start_remove_0] = 0
            jump[x_start_remove_1] = 0
            jump[y_start_remove_1] = 0
            jump[x_end_remove_0] = 0
            jump[y_end_remove_0] = 0
            jump[x_end_remove_1] = 0
            jump[y_end_remove_1] = 0

            # adjust
            self.block_x_start[idx][x_start_adjust_0] = 0
            self.block_y_start[idx][y_start_adjust_0] = 0
            self.block_x_end[idx][x_start_adjust_0] = self.block_x_size
            self.block_y_end[idx][y_start_adjust_0] = self.block_y_size

            self.block_x_start[idx][x_end_adjust_0] = self.image_sizes[idx][0] - 1 - self.block_x_size
            self.block_y_start[idx][y_end_adjust_0] = self.image_sizes[idx][1] - 1 - self.block_y_size
            self.block_x_end[idx][x_end_adjust_0] = self.image_sizes[idx][0] - 1
            self.block_y_end[idx][y_end_adjust_0] = self.image_sizes[idx][0] - 1

            self.block_x_center = (self.block_x_start + self.block_x_end) / 2
            self.block_y_center = (self.block_y_start + self.block_y_end) / 2
            temp_jump_list.append(jump)
        
        for temp_jump in temp_jump_list:
            self.jump_crop = self.jump_crop * temp_jump
    
    # crop fullsize image into pieces
    def crop_image(self, block_idx):
        if self.jump_crop[block_idx]:
            return
    
        for view_idx in range(self.view_num):
            # read image block
            block_x_start = self.block_x_center[view_idx][block_idx] - int(self.block_x_size / 2)
            block_y_start = self.block_y_center[view_idx][block_idx] - int(self.block_y_size / 2)
            image = read_image( # TODO: read block from fullsize image
                self.image_path[view_idx],
                int(block_x_start),
                int(block_y_start),
                int(self.block_x_size),
                int(self.block_y_size)
            ) # TODO: shape?
            image = image.transpose([1, 2, 0])

            # read rpc
            full_rpc = self.rpcs[view_idx]
            block_rpc = copy.deepcopy(full_rpc)
            block_rpc.SAMP_OFF -= int(block_x_start) # TODO: rpc offset
            block_rpc.LINE_OFF -= int(block_y_start)

            # save image, rpc
            out_name = "block{:0>4d}".format(block_idx)
            image_out_path = os.path.join(self.output_image_paths[view_idx], f"{out_name}.png")
            rpc_out_path = os.path.join(self.output_rpc_paths[view_idx], f"{out_name}.rpc")
            save_image(image_out_path, image) # TODO: save block from fullsize image
            save_rpc(rpc_out_path, block_rpc) # TODO: save rpc information

    def create_output_folder(self):
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)

        if not os.path.exists(self.output_image_path):
            os.mkdir(self.output_image_path)
        if not os.path.exists(self.output_rpc_path):
            os.mkdir(self.output_rpc_path)
        if not os.path.exists(self.output_height_path):
            os.mkdir(self.output_height_path)
        if not os.path.exists(self.output_points_path):
            os.mkdir(self.output_points_path)
        if not os.path.exists(self.output_dsm_path):
            os.mkdir(self.output_dsm_path)

        for v in range(self.view_num):
            if not os.path.exists(self.output_image_paths[v]):
                os.mkdir(self.output_image_paths[v])

            if not os.path.exists(self.output_rpc_paths[v]):
                os.mkdir(self.output_rpc_paths[v])

            if not os.path.exists(self.output_height_paths[v]):
                os.mkdir(self.output_height_paths[v])
    
    def generate_points(self, block_idx):
        if self.jump_crop[block_idx] == 0:
            return
        
        # output folder & file
        out_name = "block{:0>4d}".format(block_idx)
        points_path = os.path.join(self.output_points_path, out_name + ".las")
        if os.path.exists(points_path):
            os.remove(points_path)
    
        heights = []
        rpcs = []

        for idx in self.view_num:
            height_map_path = os.path.join(self.output_height_paths[idx], f"{out_name}.pfm") # TODO
            height_map = load_pfm(height_map_path)
            heights.append(height_map)

            rpc_path = os.path.join(self.output_rpc_paths[idx], f"{out_name}.rpc") # TODO
            rpc, _, _ = load_rpc_as_array(rpc_path)
            rpcs.append(rpc)
        
        heights = np.stack(heights, axis = 0)
        rpcs = np.stack(rpcs, axis = 0)
        mask, heights_est = filter_depth(heights, rpcs, p_ratio = self.p_thred, d_ratio = self.d_thred, geo_consitency_thre = self.geo_num, prob = None, cofidence_ratio = 0.2) # TODO, check arguments

        heights_est = heights_est.reshape(-1)
        mask = mask.reshape(-1)

        x = np.arange(0.0, self.block_x_size, 1.0)
        y = np.arange(0.0, self.block_y_size, 1.0)
        x, y = np.meshgrid(x, y)
        x = x.reshape(-1)
        y = y.reshape(-1)

        x = x[mask]
        y = y[mask]
        heights_final = heights_est[mask]
        
        ref_rpc = RPCModel(rpcs[0])
        lat, lon = ref_rpc.photo2obj(x, y, heights_final)
        geopoints = np.stack([lat, lon], axis = -1)
        projpoints = self.projection.project(geopoints, reverse = False)
        points = np.stack((projpoints[:, 0], projpoints[:, 1], projpoints[:, 2]), axis = -1)

        return points

    def generate_dsm(self):
        dsm_path = os.path.join(self.output_dsm_path, "dsm.tif")
        if os.path.exists(dsm_path):
            os.remove(dsm_path)

        geo_trans = [
            self.border[0] - float(self.x_unit) / 2, self.x_unit, 0,
            self.border[1] - float(self.y_unit) / 2, 0, -self.y_unit,
        ] # TODO: what
        raster_create(
            dsm_path, int(self.dsm_x_size), int(self.dsm_y_size),
            1, self.projection.spatial_reference.ExportToWkt(), geo_trans,
            self.invalid, dtype = "Float32"
        )

        points_path = os.path.join(self.output_points_path, "points.las")
        points = read_point_cloud(points_path)
        
        dsm = build_dsm(
            points, self.border[0], self.border[1],
            self.x_unit, self.y_unit, int(self.dsm_x_size),
            int(self.dsm_y_size)
        )
        write_dsm(dsm_path, 0, 0, dsm)


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
            write_point_cloud(points_output_path, points)

        # build dsm output
        if self.run_generate_dsm:
            self.generate_dsm() # TODO: generate dsm map