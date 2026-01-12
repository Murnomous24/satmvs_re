import torch

# build all pixel's same depth range
def get_same_depth_range(cur_depth, ndepth, device, dtype, shape):
    # cur_depth: [B, H, W] if sub-pixel or [B, D] 
    # shape: [B, H, W]
    # return depth_range_samples: [B, D, H, W] pixel's depth range

    # get depth interval
    cur_depth_min = cur_depth[:, 0] # [B]
    cur_depth_max = cur_depth[:, -1] # [B]
    depth_interval = (cur_depth_max - cur_depth_min) / (ndepth - 1) # [B]

    # build each pixel's depth range
    depth_range_samples = cur_depth_min.unsqueeze(1) + torch.arange(
        0,
        ndepth,
        device = device,
        dtype = dtype,
        requires_grad = False
    ).reshape(1, -1) * depth_interval.unsqueeze(1) # [B, D]
    # all pixel share same depth range
    depth_range_samples = depth_range_samples.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, shape[1], shape[2]) # [B, D, H, W]

    return depth_range_samples

# build pixelwise depth range
def get_pixelwise_depth_range(cur_depth, ndepth, depth_interval_pixelwise, device, dtype, shape):
    # cur_depth: [B, H, W] if sub-pixel or [B, D] 
    # shape: [B, H, W]
    # depth_interval_pixelwise: [B, H, W]
    # return depth_range_samples: [B, D, H, W] pixel's depth range

    half_range = (ndepth - 1) / 2 * depth_interval_pixelwise
    cur_depth_min = (cur_depth - half_range) # [B, H, W]
    cur_depth_max = (cur_depth + half_range) # [B, H, W]

    assert cur_depth.shape == torch.Size(shape), f"depth_range: cur_depth: {cur_depth.shape}, shape: {shape} do not match"

    depth_interval = (cur_depth_max - cur_depth_min) / (ndepth - 1) # [B, H, W]
    depth_range_samples = cur_depth_min.unsqueeze(1) + torch.arange(
        0,
        ndepth,
        device = device,
        dtype = dtype,
        requires_grad = False
    ).reshape(1, -1, 1, 1) * depth_interval.unsqueeze(1) # [B, D, H, W]

    return depth_range_samples

# build pixel's depth range for each cost volume setup
def get_depth_range_samples(cur_depth, ndepth, depth_interval_pixel, device, dtype, shape):
    # cur_depth: [B, H, W] if sub-pixel or [B, D] 
    # shape: [B, H, W]
    # return depth_range_samples: [B, D, H, W] pixel's depth range

    # print("depth_range: start forward")

    if cur_depth.dim() == 2:
        # print("depth_range: start same")
        return get_same_depth_range(cur_depth, ndepth, device, dtype, shape)
    else: # sub-pixel acc
        # print("depth_range: start pixelwise")
        return get_pixelwise_depth_range(cur_depth, ndepth, depth_interval_pixel, device, dtype, shape)