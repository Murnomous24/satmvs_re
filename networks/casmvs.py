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
        proj_matrices = torch.unbind(proj_matrices, 1) # [B, N, 4, 4] tensor to length Nview's python list
        assert len(features) == len(proj_matrices), f"casmvs: features and projection matrices do not match, get {len(features)} and {len(proj_matrices)}"
        assert depth_values.shape[1] == num_depth, f"casmvs: depth_values's depth number do not match num_depth, depth_values.shape[1]: {depth_values.shape[1]}, num_depth: {num_depth}"

        # set up feature map and projection matrices
        ref_fea, src_feas = features[0], features[1:] # ref_fea : [B, C, H, W]
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:] # ref_proj: [B, 4, 4]

        ref_volume = ref_fea.unsqueeze(2).repeat(1, 1, num_depth, 1, 1) # [B, C, Ndepth, H, W]
        volume_sum = ref_volume
        volume_sq_sum = ref_volume ** 2
        del ref_volume # TODO: why del

        if geo_model == "rpc":
            batch, fea, height, width = ref_fea.shape
            coef = torch.ones((batch, height * width * num_depth, 20), dtype = torch.double).cuda() # [B, H * W * Ndepth, 20]
        elif geo_model == "pinhole":
            coef = None
        else:
            raise Exception(f"casmvs: invaild 'geo_model', get {geo_model}")

        # build cost volume
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

        # cost volume regularization & depth
        cost_regular = cost_regular_func(volume_var) # TODO: dim?
        prob_volume = cost_regular.squeeze(1) # TODO: dim? 
        prob_volume = F.softmax(prob_volume, dim = 1) # [B, Ndepth, H, W]
        depth = depth_regression(prob_volume, depth_values)

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
        # feature extraction
        features = []
        for image_index in range(images.size(1)): # [B, Nview, C, H, W]
            image = images[:, image_index] # [B, C, H, W]
            features.append(self.feature(image)) # [B, Nview, C, H, W]

        outputs = {}
        last_depth, cur_depth = None, None

        # multi-stage mvs
        for index in range(self.num_stage):
            # get content mvs pipeline needs
            features_stage = [feature[f"stage{index + 1}"] for feature in features] # [b,]
            proj_matrices_stage = proj_matrices[f"stage{index + 1}"]
            scale_stage = self.stage_infos[f"stage{index + 1}"]["scale"]
        
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

            # build pixel depth range
            depth_range_samples = get_depth_range_samples(
                cur_depth,
                self.ndepths[index],
                self.depth_intervals_ratio[index] * self.min_interval,
                image[0].device,
                image[0].dtype,
                shape = [image.shape[0], image.shape[2], image.shape[3]]
            )

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

            last_depth = output_stage['depth']
            outputs[f"stage{index + 1}"] = output_stage
            outputs.update(output_stage) # TODO: update what
        
        if self.refine:
            refine_depth = self.refine_network(images[:, 0], last_depth.unsqueeze(1))
            outputs["refine_depth"] = refine_depth
        return outputs