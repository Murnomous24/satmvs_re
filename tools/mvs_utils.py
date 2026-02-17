import cv2
import numpy as np
import numpy_groupies as npg
from dataset.rpc_model import RPCModel

# filter depth map by proj and reproj's error
def filter_depth(
        depths,
        rpcs,
        p_ratio,
        d_ratio,
        geo_consitency_thre,
        prob = None,
        cofidence_ratio = 0.0
):
    ref_depth = depths[0]
    ref_rpc = rpcs[0]
    view_num = depths.shape[0]

    # ref view prob map, build photo mask
    if prob is not None:
        ref_prob = prob
        photo_mask = ref_prob > cofidence_ratio
    else:
        photo_mask = np.ones_like(ref_depth, bool)

    # build geometric mask
    geo_mask_sum = 0
    depth_map_reproj = []
    for idx in range(1, view_num):
        src_depth = depths[idx]
        src_rpc = rpcs[idx]

        geo_mask_tmp, ref_depth_reproj_tmp = reproj_and_check(ref_depth, ref_rpc, src_depth, src_rpc, p_ratio, d_ratio)

        geo_mask_sum += geo_mask_tmp.astype(np.int32)
        depth_map_reproj.append(ref_depth_reproj_tmp)
    
    # build geo mask and final depth map
    geo_mask = geo_mask_sum >= geo_consitency_thre
    depth_map_est = (sum(depth_map_reproj) + ref_depth) / (geo_mask_sum + 1) # init depth map + (N - 1) reproject depth map
    final_mask = np.logical_and(photo_mask, geo_mask)

    return final_mask, depth_map_est
        
def reproj_and_check(ref_depth, ref_rpc, src_depth, src_rpc, p_ratio, d_ratio):
    ref_rpc_model = RPCModel(ref_rpc)
    src_rpc_model = RPCModel(src_rpc)

    width, height = ref_depth.shape[1], ref_depth.shape[0]
    ref_x, ref_y = np.meshgrid(np.arange(0, width), np.arange(0, height))
    ref_x, ref_y = ref_x.reshape([-1]), ref_y.reshape([-1])

    # project: ref(2d) -> world(3d) -> src(2d), and interpolate to get new depth map
    lat, lon = ref_rpc_model.photo2obj(ref_x.astype(float), ref_y.astype(float), ref_depth.reshape(-1))
    src_x, src_y = src_rpc_model.obj2photo(lat, lon, ref_depth.reshape(-1))
    src_x, src_y = src_x.reshape([height, width]), src_y.reshape([height, width])
    src_depth_sample = cv2.remap(src_depth, src_x.astype(np.float32), src_y.astype(np.float32), interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=-999)

    # back-project: src(2d) -> wolrd(3d) -> ref(2d)
    lat, lon = src_rpc_model.photo2obj(src_x.astype(float), src_y.astype(float), src_depth_sample.reshape(-1))
    reproj_x, reproj_y = ref_rpc_model.obj2photo(lat, lon, src_depth_sample.reshape(-1))

    # check x,y error
    xy_diff = np.sqrt((reproj_x - ref_x) ** 2 + (reproj_y - ref_y) ** 2)
    depth_diff = np.abs(src_depth_sample - ref_depth)

    # build mask
    mask = np.logical_and(xy_diff < p_ratio, depth_diff < d_ratio)
    depth_diff[~mask] = 0 # fliter high-error points

    return mask, depth_diff

# build dsm map
def build_dsm(points, ul_e, ul_n, xunit, yunit, e_size, n_size):
    dsm = proj_to_grid(points, ul_e, ul_n, xunit, yunit, e_size, n_size)
    dsm = cv2.medianBlur(dsm.astype(np.float32), 3) # filter

    return dsm

# points: [N, 3]
# xoff, yoff: dsm upper-left points' geo coordinate
# xresoultion, yresolution: grid's resolution, one pixel to geometric meters
# xszie, ysize: dsm image's resolution
def proj_to_grid(points, xoff, yoff, xresolution, yresolution, xsize, ysize):

    # build pixel grid by normalization
    row = np.floor((yoff - points[:, 1]) / xresolution).astype(dtype=int)
    col = np.floor((points[:, 0] - xoff) / yresolution).astype(dtype=int)

    # build plane's index and number
    points_group_idx = row * xsize + col
    points_val = points[:, 2]

    # remove points that lie out of the dsm boundary
    mask = ((row >= 0) * (col >= 0) * (row < ysize) * (col < xsize)) > 0
    points_group_idx = points_group_idx[mask]
    points_val = points_val[mask]

    # create a place holder for all pixels in the dsm
    group_idx = np.arange(xsize * ysize).astype(dtype=int)
    group_val = np.empty(xsize * ysize)
    group_val.fill(np.nan)

    # concatenate place holders with the real valuies, then aggregate
    # the placeholders contains NaN part and real points' part
    group_idx = np.concatenate((group_idx, points_group_idx))
    group_val = np.concatenate((group_val, points_val))

    # use npg to aggregate, project 3d to 2d
    # nanmax reprents if multi 3d points project to same 2d point, decide the MAX VALUE point as the result
    dsm = npg.aggregate(group_idx, group_val, func='nanmax', fill_value=np.nan)
    dsm = dsm.reshape((ysize, xsize))

    # try to fill very small holes
    dsm_new = dsm.copy()
    nan_places = np.argwhere(np.isnan(dsm_new))
    for i in range(nan_places.shape[0]):
        row = nan_places[i, 0]
        col = nan_places[i, 1]
        neighbors = []
        for j in range(row-1, row+2):
            for k in range(col-1, col+2):
                if j >= 0 and j < dsm_new.shape[0] and k >=0 and k < dsm_new.shape[1]:
                    val = dsm_new[j, k]
                    if not np.isnan(val):
                        neighbors.append(val)
        # fill the nan place with 3*3 median number
        if neighbors:
            dsm[row, col] = np.median(neighbors)

    return dsm
