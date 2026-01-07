import torch
import torch.nn.functional as F

# pin hole feature map warping
def pinhole_warping(src_fea, src_proj, ref_proj, depth_values):
    # src_fea: [B, C, H, W]
    # src_proj: [B, 4, 4]
    # ref_proj: [B, 4, 4]
    # depth_values: [B, Ndepth] -> all pixel same depth range, [B, Ndepth, H, W] -> pixel-wise depth range
    # out: [B, C, Ndepth, H, W]

    batch = src_fea.shape[0]
    channel = src_fea.shape[1]
    num_depth = depth_values.shape[1]
    height, width = src_fea.shape[2], src_fea.shape[3]

    with torch.no_grad(): # TODO: no grad for what
        proj = torch.matmul(src_proj, torch.linalg.inv(ref_proj)) # TODO: check newest inverse method
        rot = proj[:, :3, :3].double()
        trans = proj[:, :3, 3:4]

        # build pixel grid
        y, x = torch.meshgrid(
            torch.arange(0, height, dtype = torch.float32, device = src_fea.device),
            torch.arange(0, width, dtype = torch.float32, device = src_fea.device),
            indexing = 'ij'
        )
        # build origin grid
        y, x = y.contiguous(), x.contiguous() # make grid align in mem, avoid function 'view()''s error
        y, x = y.view(height * width), x.view(height * width)
        xyz = torch.stack((x, y, torch.ones_like(x))).double() # [3, H * W] TODO: why double
        xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1) # [B, 3, H * W]
        
        # rot, trans and set up depth dimension
        rot_xyz = torch.matmul(rot, xyz)
        rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_values.view(batch, 1, num_depth, -1).double() # [B, 3, Ndepth, H * W]
        proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1) # [B, 3, Ndepth, H * W]
        
        # image coord & normalization
        proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :] # [B, 3, Ndepth, H * W] TODO: slice keep the coord dim
        proj_x_norm = proj_xy[:, 0, :, :] / ((width - 1) / 2) - 1 # [B, Ndetph, H * W]
        proj_y_norm = proj_xy[:, 1, :, :] / ((height - 1) / 2) - 1 # [B, Ndetph, H * W]
        proj_xy = torch.stack((proj_x_norm, proj_y_norm), dim = 3) # [B, Ndepth, H * W, 2]
        grid = proj_xy.float()

        # warping feautre
        warped_src_fea = F.grid_sample(
            src_fea, 
            grid.view(batch, num_depth * height, width, 2),
            mode = 'bilinear',
            padding_mode = 'zeros',
            align_corners = True)
        warped_src_fea = warped_src_fea.view(batch, channel, num_depth, height, width) # [B, C, Ndetph, H, W]

        return warped_src_fea
    
# RPC warping

# build coef
def rpc_plh_coef(P, L, H, coef):
    # P: (B, Ndepth)
    
    with torch.no_grad():
        coef[:, :, 0] = 1.0 # TODO: check necessity
        coef[:, :, 1] = L
        coef[:, :, 2] = P
        coef[:, :, 3] = H
        
        coef[:, :, 4] = L * P
        coef[:, :, 5] = L * H
        coef[:, :, 6] = P * H
        coef[:, :, 7] = L * L
        coef[:, :, 8] = P * P
        coef[:, :, 9] = H * H
        
        coef[:, :, 10] = P * coef[:, :, 5]
        coef[:, :, 11] = L * coef[:, :, 7]
        coef[:, :, 12] = L * coef[:, :, 8]
        coef[:, :, 13] = L * coef[:, :, 9]
        coef[:, :, 14] = L * coef[:, :, 4]
        coef[:, :, 15] = P * coef[:, :, 8]
        coef[:, :, 16] = P * coef[:, :, 9]
        coef[:, :, 17] = L * coef[:, :, 5]
        coef[:, :, 18] = P * coef[:, :, 6]
        coef[:, :, 19] = H * coef[:, :, 9]

# 3d point -> image coord
def rpc_obj2photo(inlat, inlon, inhei, rpc, coef):
    # inlat: [B, Ndepth * H * W]
    # inlon: [B, Ndepth * H * W]
    # inhei: [B, Ndepth * H * W]
    # rpc: [B, 170]

    with torch.no_grad():
        lat = inlat.clone()
        lon = inlon.clone()
        hei = inhei.clone()

        # normalization (lat - lat_off) / lat_scale
        lat -= rpc[:, 2].view(-1, 1) # [B * Ndepth * H * W, 1] 
        lat /= rpc[:, 7].view(-1, 1) # [B * Ndepth * H * W, 1]

        # normalization (lon - lon_off) / lon_scale
        lon -= rpc[:, 3].view(-1, 1) # [B * Ndepth * H * W, 1] 
        lon /= rpc[:, 8].view(-1, 1) # [B * Ndepth * H * W, 1]

        # normalization (hei - hei_off) / hei_scale
        hei -= rpc[:, 4].view(-1, 1) # [B * Ndepth * H * W, 1] 
        hei /= rpc[:, 9].view(-1, 1) # [B * Ndepth * H * W, 1]
        
        # projection
        rpc_plh_coef(lat, lon, hei, coef) # build polynomial coefficients
        samp = torch.sum(coef * rpc[:, 50:70].view(-1, 1, 20), dim = -1) / torch.sum(coef * rpc[:, 70:90].view(-1, 1, 20), dim = -1)
        line = torch.sum(coef * rpc[:, 10:30].view(-1, 1, 20), dim = -1) / torch.sum(coef * rpc[:, 30:50].view(-1, 1, 20), dim = -1)

        # from noramlization to pixel
        samp *= rpc[:, 6].view(-1, 1)
        samp += rpc[:, 1].view(-1, 1)

        line *= rpc[:, 5].view(-1, 1)
        line += rpc[:, 0].view(-1, 1)

    return samp, line # [B, Ndepth * H * W]

def rpc_photo2obj(insamp, inline, inhei, rpc, coef):
    # insamp: [B, Ndepth * H * W]
    # inline: [B, Ndepth * H * W]
    # inhei: [B, Ndepth * H * W]
    # rpc: [B, 170]    

    with torch.no_grad():
        samp = insamp.clone()
        line = inline.clone()
        hei = inhei.clone()

        # normalization (samp - samp_off) / samp_scale
        samp -= rpc[:, 1].view(-1, 1) # [B * Ndepth * H * W, 1] 
        samp /= rpc[:, 6].view(-1, 1) # [B * Ndepth * H * W, 1]

        # normalization (line - line_off) / line_scale
        line -= rpc[:, 0].view(-1, 1) # [B * Ndepth * H * W, 1] 
        line /= rpc[:, 5].view(-1, 1) # [B * Ndepth * H * W, 1]

        # normalization (hei - hei_off) / hei_scale
        hei -= rpc[:, 4].view(-1, 1) # [B * Ndepth * H * W, 1] 
        hei /= rpc[:, 9].view(-1, 1) # [B * Ndepth * H * W, 1]
        
        # projection
        rpc_plh_coef(samp, line, hei, coef) # build polynomial coefficients
        lat = torch.sum(coef * rpc[:, 90:110].view(-1, 1, 20), dim = -1) / torch.sum(coef * rpc[:, 110:130].view(-1, 1, 20), dim = -1)
        lon = torch.sum(coef * rpc[:, 130:150].view(-1, 1, 20), dim = -1) / torch.sum(coef * rpc[:, 150:170].view(-1, 1, 20), dim = -1)

        # from noramlization to pixel
        lat *= rpc[:, 7].view(-1, 1)
        lat += rpc[:, 2].view(-1, 1)

        lon *= rpc[:, 8].view(-1, 1)
        lon += rpc[:, 3].view(-1, 1)

    return lat, lon # [B, Ndepth * H * W]

def rpc_warping(src_fea, src_rpc, ref_rpc, depth_values, coef):
    # src_fea: [B, C, H, W]
    # src_rpc: [B, 170]
    # ref_rpc: [B, 170]
    # depth_values: [B, Ndepth] -> all pixel same depth range, [B, Ndepth, H, W] -> pixel-wise depth range
    # out: [B, C, Ndepth, H, W]

    batch = src_fea.shape[0]
    channel = src_fea.shape[1]
    num_depth = depth_values.shape[1]
    height, width = src_fea.shape[2], src_fea.shape[3]

    with torch.no_grad(): # TODO: no grad for what
        # proj = torch.matmul(src_proj, torch.linalg.inv(ref_proj)) # TODO: check newest inverse method
        # rot = proj[:, :3, :3]
        # trans = proj[:, :3, 3:4]

        # build pixel grid
        y, x = torch.meshgrid(
            torch.arange(0, height, dtype = torch.float32, device = src_fea.device),
            torch.arange(0, width, dtype = torch.float32, device = src_fea.device),
            indexing = 'ij'
        )
        # build samp, line, hei
        y, x = y.contiguous(), x.contiguous() # make grid align in mem, avoid function 'view()''s error
        y, x = y.view(height * width), x.view(height * width)
        y = y.view(1, 1, height, width).repeat(batch, num_depth, 1, 1) # [B, Ndepth, H, W]
        x = x.view(1, 1, height, width).repeat(batch, num_depth, 1, 1) # [B, Ndepth, H, W]
        inline = y.view(batch, -1).double() # [B, Ndepth * H * W]
        insamp = x.view(batch, -1).double() # [B, Ndepth * H * W]

        if len(depth_values.shape) == 2: # all pixel same depth range
            inhei = depth_values.view(batch, num_depth, 1, 1).double().repeat(1, 1, height, width)
        else:
            inhei = depth_values.double()
        inhei = inhei.view(batch, -1) # [B, Ndepth * H * W]

        # rpc proj
        lat, lon = rpc_photo2obj(insamp, inline, inhei, ref_rpc, coef) # [B, Ndepth * H * W]
        outsamp, outline = rpc_obj2photo(lat, lon, inhei, src_rpc, coef) # [B, Ndepth * H * W]
        outsamp = outsamp.float()
        outline = outline.float()
        
        proj_x_norm = outsamp / ((width - 1) / 2) - 1 # [B, Ndepth * H * W]
        proj_y_norm = outline / ((height - 1) / 2) - 1 # [B, Ndepth * H * W]
        proj_x_norm = proj_x_norm.view(batch, num_depth, height * width) # [B, Ndepth, H * W]
        proj_y_norm = proj_y_norm.view(batch, num_depth, height * width) # [B, Ndepth, H * W]

        proj_xy = torch.stack((proj_x_norm, proj_y_norm), dim = 3) # [B, Ndepth, H * W, 2]
        grid = proj_xy.float() # TODO check unit

        # warping feautre
        warped_src_fea = F.grid_sample(
            src_fea, 
            grid.view(batch, num_depth * height, width, 2),
            mode = 'bilinear',
            padding_mode = 'zeros',
            align_corners = True)
        warped_src_fea = warped_src_fea.view(batch, channel, num_depth, height, width) # [B, C, Ndetph, H, W]

        return warped_src_fea

if __name__ == "__main__":
    import numpy as np

    B, C, H, W = 1, 8, 32, 64
    Ndepth = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    src_fea = torch.randn(B, C, H, W).to(device)
    depth_values = torch.linspace(10.0, 50.0, Ndepth).view(1, Ndepth).repeat(B, 1).to(device) # [B, Ndepth]
    print(f"src_fea: {src_fea.shape}")
    print(f"ndepth: {Ndepth}")

    print("pinhole_warping...")
    ref_proj = torch.eye(4).unsqueeze(0).repeat(B, 1, 1).to(device) # [B, 4, 4]
    src_proj = torch.eye(4).unsqueeze(0).repeat(B, 1, 1).to(device) # [B, 4, 4]
    src_proj[:, 0, 3] = 0.1
    try:
        warped_pinhole = pinhole_warping(src_fea, src_proj, ref_proj, depth_values)
        print(f"pinhole warping success, shape: {warped_pinhole.shape}")
    except Exception as e:
        print(f"pinhole warping failed: {e}")

    print("rpc_warping...")
    rpc_dummy = torch.zeros(B, 170).to(device) # [B, 170]
    rpc_dummy[:, 5:10] = 1.0  # LINE_SCALE, SAMP_SCALE, LAT_SCALE, LONG_SCALE, HEIGHT_SCALE
    rpc_dummy[:, 30] = 1.0; rpc_dummy[:, 70] = 1.0
    rpc_dummy[:, 110] = 1.0; rpc_dummy[:, 150] = 1.0
    rpc_dummy[:, 11] = 1.0; rpc_dummy[:, 52] = 1.0
    coef_buffer = torch.zeros(B, Ndepth * H * W, 20).to(device).double() # [B, Ndepth * H * W, 20]
    try:
        warped_rpc = rpc_warping(src_fea, rpc_dummy, rpc_dummy, depth_values, coef_buffer)
        print(f"RPC Warping success. shape: {warped_rpc.shape}")
    except Exception as e:
        print(f"RPC Warping failed: {e}")