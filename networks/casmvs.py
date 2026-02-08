import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.module import *
from modules.warping import *
from modules.depth_range import *

Align_Corners_Range = False # TODO: what

class DepthNet(nn.Module):
    def __init__(self):
        super(DepthNet, self).__init__()

    def forward(
            self,
            features,
            proj_matrices,
            depth_values,
            num_depth,
            cost_regular_func, # TODO
            geo_model, # TODO
    ):
        # print("depthnet forward 0: start forward")

        proj_matrices = torch.unbind(proj_matrices, 1) # [B, N, 4, 4] tensor to length Nview's python list
        assert len(features) == len(proj_matrices), f"casmvs: features and projection matrices do not match, get {len(features)} and {len(proj_matrices)}"
        assert depth_values.shape[1] == num_depth, f"casmvs: depth_values's depth number do not match num_depth, depth_values.shape[1]: {depth_values.shape[1]}, num_depth: {num_depth}"

        # set up feature map and projection matrices
        ref_fea, src_feas = features[0], features[1:] # ref_fea : [B, C, H, W]
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:] # ref_proj: [B, 4, 4]

        ref_volume = ref_fea.unsqueeze(2).repeat(1, 1, num_depth, 1, 1) # [B, C, Ndepth, H, W]
        volume_sum = ref_volume
        volume_sq_sum = ref_volume ** 2
        # print("depthnet forward 1: map and volume setup ok")
        del ref_volume # TODO: why del

        if geo_model == "rpc":
            batch, fea, height, width = ref_fea.shape
            coef = torch.ones((batch, height * width * num_depth, 20), dtype = torch.double).cuda() # [B, H * W * Ndepth, 20]
        elif geo_model == "pinhole":
            coef = None
        else:
            raise Exception(f"casmvs: invaild 'geo_model', get {geo_model}")
        # print("depthnet forward 2: rpc coef ok(if need)")

        # build cost volume
        # print(f"depthnet: src_fea: {src_feas[0].shape}, src_proj: {src_projs[0].shape}, ref_proj: {ref_proj.shape}, depth_values: {depth_values.shape}")
        for src_fea, src_proj in zip(src_feas, src_projs): # TODO: how we loop
            if geo_model == "rpc":
                warped_volume = rpc_warping(src_fea, src_proj, ref_proj, depth_values, coef)
            elif geo_model == "pinhole":
                warped_volume = pinhole_warping(src_fea, src_proj, ref_proj, depth_values)
            
            if self.training:
                volume_sum = volume_sum + warped_volume
                volume_sq_sum = volume_sq_sum + warped_volume ** 2
            else: # TODO: why do this
                volume_sum += warped_volume
                volume_sq_sum += warped_volume.pow_(2)
            del warped_volume
        num_views = len(features)
        volume_var = volume_sq_sum.div_(num_views).sub_(volume_sum.div_(num_views).pow_(2)) # [B, C, Ndepth, H, W]
        # print("depthnet forward 3: cost volume build ok")

        # cost volume regularization & depth
        cost_regular = cost_regular_func(volume_var) # TODO: dim?
        prob_volume = cost_regular.squeeze(1) # TODO: dim? 
        prob_volume = F.softmax(prob_volume, dim = 1) # [B, Ndepth, H, W]
        depth = depth_regression(prob_volume, depth_values)
        # print("depthnet forward 4: cost volume regular and depth estimation ok")

        # confidence
        with torch.no_grad():
            prob_volume_sum4 = 4 * F.avg_pool3d(
                F.pad(prob_volume.unsqueeze(1), pad = (0, 0, 0, 0, 1, 2)),
                (4, 1, 1),
                stride = 1,
                padding = 0
            ).squeeze(1) # [B, Ndepth, H, W]
            depth_index = depth_regression(prob_volume, depth_values = torch.arange(num_depth, device = prob_volume.device, dtype = torch.float)).long() # [B, H, W] get the most probably depth index pixel by pixel
            depth_index = depth_index.clamp(min=0, max=num_depth-1) # fliter illegal index
            photometric_confidence = torch.gather(prob_volume_sum4, 1, depth_index.unsqueeze(1)).squeeze(1) # [B, H, W] get the probability from the most probably depth index
        
        # refine TODO
        # print("depthnet forward 5: confidence ok")
        return {
            "depth": depth,
            "photometric_confidence": photometric_confidence
        }
        
class CascadeMVSNet(nn.Module):
    def __init__(
            self,
            geo_model,
            refine = False,
            min_interval = 2.5,
            ndepths = [48, 32, 8],
            depth_intervals_ratio = [4, 2, 1],
            share_cr = False,
            grad_method = "detach",
            arch_mode = "fpn",
            cr_base_chs = [8, 8, 8]
    ):
        super(CascadeMVSNet, self).__init__()
        
        self.geo_model = geo_model
        self.refine = refine
        self.share_cr = share_cr
        self.ndepths = ndepths
        self.depth_intervals_ratio = depth_intervals_ratio
        self.grad_method = grad_method
        self.arch_mode = arch_mode
        self.cr_base_chs = cr_base_chs
        self.num_stage = len(ndepths)
        self.min_interval = min_interval
        self.stage_infos = {
            "stage1": { "scale": 4.0 },
            "stage2": { "scale": 2.0 },
            "stage3": { "scale": 1.0 },    
        }
        
        assert len(ndepths) == len(depth_intervals_ratio), f"casmvs: ndepths: {ndepths.shape}, depth_intervals_ratio: {depth_intervals_ratio.shape} do not match"

        self.feature = FeatureNet(
            base_channels = 8, 
            stride = 4,
            num_stage = self.num_stage,
            arch_mode = self.arch_mode
        )
        if self.share_cr:
            self.cost_regularization = CostRegNet(
                in_channels = self.feature.out_channels,
                base_channels = 8
            )
        else:
            self.cost_regularization = nn.ModuleList(
                [CostRegNet(
                    in_channels = self.feature.out_channels[index],
                    base_channels = self.cr_base_chs[index]
                ) for index in range (self.num_stage)]
            )
        if self.refine:
            self.refine_network = RefineNet()
        self.DepthNet = DepthNet()

    def forward(
        self,
        images,
        proj_matrices,
        depth_values
    ):
        # print("cascade forward 0: start forward")
        # print(f"cascade forward: images: {images.shape}")
        # feature extraction
        features = []
        for image_index in range(images.size(1)): # [B, Nview, C, H, W]
            image = images[:, image_index] # [B, C, H, W]
            features.append(self.feature(image)) # [B, Nview, C, H, W]
        
        # print("cascade forward 1: feature extraction ok")

        outputs = {}
        last_depth, cur_depth = None, None

        # multi-stage mvs
        for index in range(self.num_stage):
            # get content mvs pipeline needs
            features_stage = [feature[f"stage{index + 1}"] for feature in features] # [b,]
            proj_matrices_stage = proj_matrices[f"stage{index + 1}"]
            scale_stage = self.stage_infos[f"stage{index + 1}"]["scale"]
        
            # print("cascade forward 2: multi-stage input ok")
            # interpolate last_depth -> cur_depth
            if last_depth is not None:
                if self.grad_method == "detach": # TODO: why
                    cur_depth = last_depth.detach()
                else:
                    cur_depth = last_depth # [B, H, W]
                cur_depth = F.interpolate(
                    cur_depth.unsqueeze(1),
                    [image.shape[2], image.shape[3]],
                    mode = 'bilinear',
                    align_corners = Align_Corners_Range
                ).squeeze(1)
            else:
                cur_depth = depth_values

            # print("cascade forward 3: depth update ok")
            # build pixel depth range
            depth_range_samples = get_depth_range_samples(
                cur_depth,
                self.ndepths[index],
                self.depth_intervals_ratio[index] * self.min_interval,
                image[0].device,
                image[0].dtype,
                shape = [image.shape[0], image.shape[2], image.shape[3]]
            )

            # print("cascade forward 4: depth range sample ok")
            # build cost volume and get estimation depth map
            output_stage = self.DepthNet(
                features_stage,
                proj_matrices_stage,
                F.interpolate(
                    depth_range_samples.unsqueeze(1), # [B, 1, Ndepth, H, W] # TODO: why
                    [self.ndepths[index], image.shape[2] // int(scale_stage), image.shape[3] // int(scale_stage)], # [Ndepth, H, W]
                    mode = 'trilinear', # TODO: why trilinear
                    align_corners = Align_Corners_Range
                ).squeeze(1),
                self.ndepths[index],
                self.cost_regularization if self.share_cr else self.cost_regularization[index],
                self.geo_model
            )

            # print("cascade forward 5: depthnet ok")
            # add to output
            last_depth = output_stage['depth']
            outputs[f"stage{index + 1}"] = output_stage
            outputs.update(output_stage) # TODO: update what
        
        # print("cascade forward 6: multi-stage ok")
        # print(f"depth: {depth.shape}")
        if self.refine:
            refine_depth = self.refine_network(images[:, 0], last_depth.unsqueeze(1))
            outputs["refine_depth"] = refine_depth
        # print("cascade forward 7: refine ok")
        return outputs
            
# # test code below
# def test_cascademvs_pinhole():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     batch_size = 1
#     num_views = 3
#     height, width = 512, 640
#     ndepths = [48, 32, 8]

#     model = CascadeMVSNet(
#         geo_model = "pinhole",
#         refine = True,
#         ndepths = ndepths,
#         arch_mode = "fpn"
#     ).to(device)
#     model.eval()
#     print("1: model initialization")

#     images = torch.randn(
#         batch_size,
#         num_views,
#         3,
#         height,
#         width
#     ).to(device) # [B, Nview, C, H, W]

#     # python list style projection matrices
#     single_proj = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1).to(device) # [1, 4, 4]
#     proj_list = [single_proj for _ in range(num_views)]
#     proj_matrices = {
#         "stage1": proj_list,
#         "stage2": proj_list,
#         "stage3": proj_list
#     }

#     # proj_matrix = torch.eye(4).view(1, 1, 4, 4).repeat(batch_size, num_views, 1, 1).to(device) # [B, Nview, 4, 4]
#     # proj_matrices = {
#     #     "stage1": proj_matrix,
#     #     "stage2": proj_matrix,
#     #     "stage3": proj_matrix
#     # }

#     init_depth = torch.ones(batch_size, height, width).to(device) * 10.0

#     print("2: input ok")
#     try:
#         with torch.no_grad():
#             outputs = model(images, proj_matrices, init_depth)

#             for stage in ["stage1", "stage2", "stage3"]:
#                 if stage in outputs:
#                     depth = outputs[stage]["depth"]
#                     conf = outputs[stage]["photometric_confidence"]
#                     print(f"[{stage}] depth shape: {list(depth.shape)}, photometric shape: {list(conf.shape)}")
            
#             if "refine_depth" in outputs:
#                 print(f"refined depth map: {list(outputs['refine_depth'].shape)}")
#     except Exception as e:
#         print(f"error: {e}")

#     print("3: network ok")


# # test rpc
# import numpy as np
# import os
# def load_rpc_as_array(file_name):
#     if os.path.exists(file_name) is False:
#         raise Exception("load_rpc_as_array: pfm file not find")

#     file = open(file_name, 'r')

#     full_text = file.read().splitlines()
#     data = [line.split(' ')[1] for line in full_text]
#     # print(data)

#     data = np.array(data, dtype = np.float64)
    
#     h_min = data[4] - data[9]
#     h_max = data[4] + data[9]

#     return data, h_min, h_max
# def test_cascademvs_rpc():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     batch_size = 1
#     num_views = 3
#     height, width = 512, 640
#     ndepths = [48, 32, 8]

#     model = CascadeMVSNet(
#         geo_model = "rpc",
#         refine = True,
#         ndepths = ndepths,
#         arch_mode = "fpn"
#     ).to(device)
#     model.eval()
#     print("1: RPC model initialization")

#     images = torch.randn(
#         batch_size,
#         num_views,
#         3,
#         height,
#         width
#     ).to(device)

    
#     try:
#         rpc_path = '/home/murph_dl/Paper_Re/SatMVS_Re/test_file/test_dataset_rpc/rpc/0/base0000block0016.rpc'
#         rpc_data, h_min, h_max = load_rpc_as_array(rpc_path)

#         rpc_para = []
#         rpc_para.append(rpc_data)
#         rpc_para = np.stack(rpc_para)

#         rpc_tensor = torch.from_numpy(rpc_para).float().to(device)
#         print(f"rpc_tensor: {rpc_tensor.shape}")
        
#     except Exception as e:
#         print(f"Warning: Could not load RPC file, using zero dummy. Error: {e}")
#         rpc_tensor = torch.zeros(batch_size, 170).to(device)

#     rpc_list = [rpc_tensor for _ in range(num_views)]
#     proj_matrices = {
#         "stage1": rpc_list,
#         "stage2": rpc_list,
#         "stage3": rpc_list
#     }

#     # single_rpc = torch.zeros(batch_size, 170).to(device)
#     # single_rpc[:, 5:10] = 1.0  
#     # single_rpc[:, 30] = 1.0
#     # single_rpc[:, 70] = 1.0
#     # single_rpc[:, 110] = 1.0
#     # rpc_list = [single_rpc for _ in range(num_views)]
#     # proj_matrices = {
#     #     "stage1": rpc_list,
#     #     "stage2": rpc_list,
#     #     "stage3": rpc_list
#     # }

#     init_depth = torch.ones(batch_size, height, width).to(device) * 10.0

#     print("2: RPC input ok")

#     try:
#         with torch.no_grad():
#             outputs = model(images, proj_matrices, init_depth)

#             for stage in ["stage1", "stage2", "stage3"]:
#                 if stage in outputs:
#                     depth = outputs[stage]["depth"]
#                     conf = outputs[stage]["photometric_confidence"]
#                     print(f"[{stage}] RPC depth shape: {list(depth.shape)}, photometric shape: {list(conf.shape)}")
        
#             if "refine_depth" in outputs:
#                 print(f"Refined RPC depth map shape: {list(outputs['refine_depth'].shape)}")
#     except Exception as e:
#         print(f"RPC testing error: {e}")

#     print("3: RPC network ok")

# if __name__ == "__main__":
#     test_cascademvs_pinhole()
#     test_cascademvs_rpc()