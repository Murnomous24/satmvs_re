import numpy as np
from utils_utm_projection import Projection
from dataset.rpc_model import RPCModel

# block adjustment(avoiding out of bound)
# TODO: modify!!!!
def block_adjust(
        block_num: int,
        view_num: int,
        image_sizes: list,
        block_x_center: np.ndarray,
        block_y_center: np.ndarray,
        block_x_size: int,
        block_y_size: int,
        x_overlap_size: float,
        y_overlap_size: float,
        projection: Projection,
        jump_flag_list: list,
    ):
    # block rects
    block_x_start = (block_x_center - block_x_size / 2).astype(int)
    block_y_start = (block_y_center - block_y_size / 2).astype(int)
    block_x_end = (block_x_center + block_x_size / 2).astype(int)
    block_y_end = (block_y_center + block_y_size / 2).astype(int)

    temp_jump_list = []
    for idx in range(view_num):
        jump = np.ones(block_num, dtype = int)

        # start (remove condition)
        # upper-left out range
        x_start_remove_0 = (block_x_start[idx] <= -x_overlap_size * block_x_size)
        y_start_remove_0 = block_y_start[idx] <= -y_overlap_size * block_y_size
        # bottom-right out range
        x_start_remove_1 = block_x_start[idx] >= image_sizes[idx][0] - 1
        y_start_remove_1 = block_y_start[idx] >= image_sizes[idx][1] - 1

        # end (remove condtion)
        # upper-left out range
        x_end_remove_0 = block_x_end[idx] < 0
        y_end_remove_0 = block_y_end[idx] < 0
        # bottom-right out range
        x_end_remove_1 = block_x_end[idx] >= image_sizes[idx][0] - 1 + x_overlap_size * block_x_size
        y_end_remove_1 = block_y_end[idx] >= image_sizes[idx][1] - 1 + y_overlap_size * block_y_size

        # start (adjust)
        # upper-left
        x_start_adjust_0 = (block_x_start[idx] < 0) & (block_x_start[idx] > -x_overlap_size * block_x_size)
        y_start_adjust_0 = (block_y_start[idx] < 0) & (block_y_start[idx] > -y_overlap_size * block_y_size)
        # bottom-right
        x_end_adjust_0 = (block_x_end[idx] > image_sizes[idx][0] - 1) & (block_x_end[idx] < image_sizes[idx][0] - 1 + x_overlap_size * block_x_size)
        y_end_adjust_0 = (block_y_end[idx] > image_sizes[idx][1] - 1) & (block_y_end[idx] < image_sizes[idx][1] - 1 + y_overlap_size * block_y_size)

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
        block_x_start[idx][x_start_adjust_0] = 0
        block_y_start[idx][y_start_adjust_0] = 0
        block_x_end[idx][x_start_adjust_0] = block_x_size
        block_y_end[idx][y_start_adjust_0] = block_y_size

        block_x_start[idx][x_end_adjust_0] = image_sizes[idx][0] - 1 - block_x_size
        block_y_start[idx][y_end_adjust_0] = image_sizes[idx][1] - 1 - block_y_size
        block_x_end[idx][x_end_adjust_0] = image_sizes[idx][0] - 1
        block_y_end[idx][y_end_adjust_0] = image_sizes[idx][0] - 1

        block_x_center = (block_x_start + block_x_end) / 2
        block_y_center = (block_y_start + block_y_end) / 2
        temp_jump_list.append(jump)
    
    for temp_jump in temp_jump_list:
        jump_flag = jump_flag * temp_jump

# blocking properties calculation
def calculate_block_properties(
        dsm_x_size: int,
        dsm_y_size: int,
        x_overlap: float,
        y_overlap: float,
        x_unit: float,
        y_unit: float,
        border: list,
        rpcs: list[RPCModel],
        projection: Projection,
        depth_range: list,
        para: int
    ):
    block_num_x = dsm_x_size / (block_x_size)
    block_num_y = dsm_y_size / (block_y_size)

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
    x_grid_start = (np.arange(block_num_x) * block_x_size * x_overlap).astype(int)
    x_grid_end = x_grid_start + block_x_size
    y_grid_start = (np.arange(block_num_y) * block_y_size * y_overlap).astype(int)
    y_grid_end = y_grid_start + block_y_size

    # adjust grid avoid exceed border limit
    x_over_index = x_grid_end > dsm_x_size
    x_grid_start[x_over_index] = dsm_x_size - block_x_size
    x_grid_end = x_grid_start + block_x_size

    y_over_index = y_grid_end > dsm_y_size
    y_grid_start[y_over_index] = dsm_y_size - block_y_size
    y_grid_end = y_grid_start + block_y_size

    # build meshgrid, shape like [block_num_x, block_num_y]
    x_grid_start, y_grid_start = np.meshgrid(x_grid_start, y_grid_start) 
    x_grid_end, y_grid_end = np.meshgrid(x_grid_end, y_grid_end)
    
    # build one-dim vector, shape like [block_num_x * block_num_y]
    x_grid_start = x_grid_start.reshape(-1)
    y_grid_start = y_grid_start.reshape(-1)
    x_grid_end = x_grid_end.reshape(-1)
    y_grid_end = y_grid_end.reshape(-1)
    
    # grid -> dsm
    x_dsm_start = x_grid_start * x_unit + border[0]
    y_dsm_start = -y_grid_start * y_unit + border[1] # TODO: why minus, coordinate setup, y axis face to negative
    x_dsm_end = x_grid_end * x_unit + border[0]
    y_dsm_end = -y_grid_end * y_unit + border[1]
    
    # build per-block corner coordinate
    point_upper_left = np.stack((x_dsm_start, y_dsm_start), axis = -1)
    point_upper_right = np.stack((x_dsm_end, y_dsm_start), axis = -1)
    point_bottom_left = np.stack((x_dsm_start, y_dsm_end), axis = -1)
    point_bottom_right = np.stack((x_dsm_end, y_dsm_end), axis = -1)
    rects = np.stack((point_upper_left, point_bottom_right, point_upper_right, point_bottom_left)) # TODO: order?

    # plane rects -> geometry rects
    geo_rects = projection.project(rects, reverse = True) # [N, N, 2]
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
    for rpc in rpcs:
        h_min = height_temp + depth_range[0]
        h_max = height_temp + depth_range[1]

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
        block_x_size = int(block_x_size / para + 1) * para
        block_y_size = int(block_y_size / para + 1) * para
        
        block_x_size_list.append(block_x_size)
        block_y_size_list.append(block_y_size)
    
    block_x_start_pixel_list = np.array(block_x_start_pixel_list)
    block_y_start_pixel_list = np.array(block_y_start_pixel_list)
    block_x_end_pixel_list = np.array(block_x_end_pixel_list)
    block_y_end_pixel_list = np.array(block_y_end_pixel_list)

    # block center
    block_x_center = (block_x_start_pixel_list + block_x_end_pixel_list) / 2
    block_y_center = (block_y_start_pixel_list + block_y_end_pixel_list) / 2
    block_x_center = block_x_center.astype(int)
    block_y_center = block_y_center.astype(int)

    # block size & block range adjust
    block_x_size = np.max(block_x_size) # TODO, here?
    block_y_size = np.max(block_y_size)
    block_num = block_x_size.shape[0]

    block_adjust(#TODO)