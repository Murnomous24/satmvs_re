import torch
import torch.nn as nn
import torch.nn.functional as F

def _get_stage_items(inputs):
    stage_keys = sorted(
        [key for key in inputs.keys() if key.startswith("stage")],
        key = lambda key: int(key.replace("stage", ""))
    )
    return [(inputs[key], key) for key in stage_keys]

def _get_stage_weight(weights, stage_index):
    if weights is None:
        return 1.0
    return weights[stage_index]

def _normalize_depth_map(depth, depth_min, depth_max, eps = 1e-12):
    depth_range = (depth_max - depth_min).clamp_min(eps)[:, None, None]
    depth_norm = (depth - depth_min[:, None, None]) / depth_range
    return depth_norm.clamp(0.0, 1.0)

def _soft_histogram(depth_norm, mask, num_bins = 32, sigma = None, eps = 1e-12):
    if sigma is None:
        sigma = 1.0 / num_bins

    bin_centers = torch.linspace(
        0.5 / num_bins,
        1.0 - 0.5 / num_bins,
        steps = num_bins,
        device = depth_norm.device,
        dtype = depth_norm.dtype
    ).view(1, num_bins, 1, 1)

    diff = depth_norm.unsqueeze(1) - bin_centers
    weights = torch.exp(-0.5 * (diff / sigma) ** 2)
    weights = weights / weights.sum(dim = 1, keepdim = True).clamp_min(eps)
    weights = weights * mask.unsqueeze(1).to(depth_norm.dtype)
    hist = weights.sum(dim = [2, 3])
    hist = hist / hist.sum(dim = 1, keepdim = True).clamp_min(eps)
    return hist

def depth_distribution_similarity_loss(
        depth_pred,
        depth_gt,
        mask,
        depth_min,
        depth_max,
        num_bins = 32,
        sigma = None,
        eps = 1e-12
):
    mask = mask > 0.5
    valid_samples = mask.reshape(mask.shape[0], -1).sum(dim = 1) > 0
    if not valid_samples.any():
        return depth_pred.new_tensor(0.0)

    depth_pred_norm = _normalize_depth_map(depth_pred, depth_min, depth_max, eps = eps)
    depth_gt_norm = _normalize_depth_map(depth_gt, depth_min, depth_max, eps = eps)

    pred_hist = _soft_histogram(depth_pred_norm, mask, num_bins = num_bins, sigma = sigma, eps = eps)
    gt_hist = _soft_histogram(depth_gt_norm, mask, num_bins = num_bins, sigma = sigma, eps = eps)

    kl_divergence = torch.sum(
        gt_hist * (torch.log(gt_hist.clamp_min(eps)) - torch.log(pred_hist.clamp_min(eps))),
        dim = 1
    )
    return kl_divergence[valid_samples].mean()

def casmvs_loss(
        inputs,
        depth_gt_ms,
        mask_ms,
        **kwargs
):
    depth_loss_weights = kwargs.get("dlossw", None)

    total_loss = torch.tensor(
        0.0, 
        dtype = torch.float32, 
        device = mask_ms["stage1"].device,
        requires_grad = False)
    depth_loss = torch.tensor(
        0.0,
        dtype = torch.float32,
        device = mask_ms["stage1"].device,
        requires_grad = False
    )

    for stage_index, (stage_inputs, stage_key) in enumerate(_get_stage_items(inputs)):
        depth_est = stage_inputs["depth"]
        depth_gt = depth_gt_ms[stage_key]
        mask = mask_ms[stage_key]
        mask = mask > 0.5 

        stage_depth_loss = F.smooth_l1_loss(depth_est[mask], depth_gt[mask], reduction = "mean")
        stage_weight = _get_stage_weight(depth_loss_weights, stage_index)

        depth_loss += stage_weight * stage_depth_loss
        total_loss += stage_weight * stage_depth_loss
    
    return total_loss, depth_loss

def pixel_wise_loss(prob_volume, depth_gt, mask, depth_hypo):
    mask_true = mask
    valid_pixel_num = torch.sum(mask_true, dim = [1, 2]) + 1e-12

    gt_index_image = torch.argmin(
        torch.abs(depth_hypo - depth_gt.unsqueeze(1)),
        dim = 1
    )
    gt_index_image = (mask_true * gt_index_image.to(torch.float32)).round().long().unsqueeze(1)

    gt_index_volume = torch.zeros_like(prob_volume).scatter_(1, gt_index_image, 1.0)
    cross_entropy_image = -torch.sum(
        gt_index_volume * torch.log(prob_volume.clamp_min(1e-12)),
        dim = 1
    )
    masked_cross_entropy = torch.sum(mask_true * cross_entropy_image, dim = [1, 2])
    return torch.mean(masked_cross_entropy / valid_pixel_num)

def entropy_loss(
        inputs,
        depth_gt_ms,
        mask_ms,
        **kwargs
):
    entropy_loss_weights = kwargs.get("elossw", None)

    total_loss = torch.tensor(
        0.0,
        dtype = torch.float32,
        device = mask_ms["stage1"].device,
        requires_grad = False
    )
    entropy_component = torch.tensor(
        0.0,
        dtype = torch.float32,
        device = mask_ms["stage1"].device,
        requires_grad = False
    )

    for stage_index, (stage_inputs, stage_key) in enumerate(_get_stage_items(inputs)):
        prob_volume = stage_inputs["prob_volume"]
        depth_hypo = stage_inputs["depth_hypo"]
        depth_gt = depth_gt_ms[stage_key]
        mask = mask_ms[stage_key] > 0.5

        stage_entropy_loss = pixel_wise_loss(prob_volume, depth_gt, mask, depth_hypo)
        stage_weight = _get_stage_weight(entropy_loss_weights, stage_index)
        entropy_component += stage_weight * stage_entropy_loss
        total_loss += stage_weight * stage_entropy_loss

    return total_loss, entropy_component

def casmvs_entropy_loss(
        inputs,
        depth_gt_ms,
        mask_ms,
        **kwargs
):
    depth_loss_weights = kwargs.get("dlossw", None)
    entropy_loss_weights = kwargs.get("elossw", depth_loss_weights)
    entropy_weight = kwargs.get("entropy_weight", 1.0)

    total_loss = torch.tensor(
        0.0,
        dtype = torch.float32,
        device = mask_ms["stage1"].device,
        requires_grad = False
    )
    depth_loss = torch.tensor(
        0.0,
        dtype = torch.float32,
        device = mask_ms["stage1"].device,
        requires_grad = False
    )
    entropy_loss = torch.tensor(
        0.0,
        dtype = torch.float32,
        device = mask_ms["stage1"].device,
        requires_grad = False
    )

    for stage_index, (stage_inputs, stage_key) in enumerate(_get_stage_items(inputs)):
        depth_est = stage_inputs["depth"]
        prob_volume = stage_inputs["prob_volume"]
        depth_hypo = stage_inputs["depth_hypo"]
        depth_gt = depth_gt_ms[stage_key]
        mask = mask_ms[stage_key] > 0.5

        stage_depth_loss = F.smooth_l1_loss(depth_est[mask], depth_gt[mask], reduction = "mean")
        stage_entropy_loss = pixel_wise_loss(prob_volume, depth_gt, mask, depth_hypo)
        depth_stage_weight = _get_stage_weight(depth_loss_weights, stage_index)
        entropy_stage_weight = _get_stage_weight(entropy_loss_weights, stage_index)

        depth_loss += depth_stage_weight * stage_depth_loss
        entropy_loss += entropy_stage_weight * stage_entropy_loss
        total_loss += depth_stage_weight * stage_depth_loss
        total_loss += entropy_weight * entropy_stage_weight * stage_entropy_loss

    return total_loss, depth_loss, entropy_loss

def casmvs_dds_loss(
        inputs,
        depth_gt_ms,
        mask_ms,
        **kwargs
):
    depth_loss_weights = kwargs.get("dlossw", None)
    dds_loss_weights = kwargs.get("ddslw", depth_loss_weights)
    dds_weight = kwargs.get("dds_weight", 0.05)
    depth_values = kwargs.get("depth_values")
    dds_num_bins = kwargs.get("dds_num_bins", 32)
    dds_sigma = kwargs.get("dds_sigma", None)

    if depth_values is None:
        raise ValueError("casmvs_dds_loss requires 'depth_values' in kwargs")

    depth_min = depth_values[:, 0]
    depth_max = depth_values[:, -1]

    total_loss = torch.tensor(
        0.0,
        dtype = torch.float32,
        device = mask_ms["stage1"].device,
        requires_grad = False
    )
    depth_loss = torch.tensor(
        0.0,
        dtype = torch.float32,
        device = mask_ms["stage1"].device,
        requires_grad = False
    )
    dds_loss = torch.tensor(
        0.0,
        dtype = torch.float32,
        device = mask_ms["stage1"].device,
        requires_grad = False
    )

    for stage_index, (stage_inputs, stage_key) in enumerate(_get_stage_items(inputs)):
        depth_est = stage_inputs["depth"]
        depth_gt = depth_gt_ms[stage_key]
        mask = mask_ms[stage_key] > 0.5

        stage_depth_loss = F.smooth_l1_loss(depth_est[mask], depth_gt[mask], reduction = "mean")
        stage_dds_loss = depth_distribution_similarity_loss(
            depth_est,
            depth_gt,
            mask,
            depth_min,
            depth_max,
            num_bins = dds_num_bins,
            sigma = dds_sigma
        )

        depth_stage_weight = _get_stage_weight(depth_loss_weights, stage_index)
        dds_stage_weight = _get_stage_weight(dds_loss_weights, stage_index)

        depth_loss += depth_stage_weight * stage_depth_loss
        dds_loss += dds_stage_weight * stage_dds_loss
        total_loss += depth_stage_weight * stage_depth_loss
        total_loss += dds_weight * dds_stage_weight * stage_dds_loss

    return total_loss, depth_loss, dds_loss

def entropy_dds_loss(
        inputs,
        depth_gt_ms,
        mask_ms,
        **kwargs
):
    entropy_loss_weights = kwargs.get("elossw", None)
    dds_loss_weights = kwargs.get("ddslw", None)
    entropy_weight = kwargs.get("entropy_weight", 1.0)
    dds_weight = kwargs.get("dds_weight", 0.05)
    depth_values = kwargs.get("depth_values")
    dds_num_bins = kwargs.get("dds_num_bins", 32)
    dds_sigma = kwargs.get("dds_sigma", None)

    if depth_values is None:
        raise ValueError("entropy_dds_loss requires 'depth_values' in kwargs")

    depth_min = depth_values[:, 0]
    depth_max = depth_values[:, -1]

    total_loss = torch.tensor(
        0.0,
        dtype = torch.float32,
        device = mask_ms["stage1"].device,
        requires_grad = False
    )
    entropy_component = torch.tensor(
        0.0,
        dtype = torch.float32,
        device = mask_ms["stage1"].device,
        requires_grad = False
    )
    dds_component = torch.tensor(
        0.0,
        dtype = torch.float32,
        device = mask_ms["stage1"].device,
        requires_grad = False
    )

    for stage_index, (stage_inputs, stage_key) in enumerate(_get_stage_items(inputs)):
        depth_est = stage_inputs["depth"]
        prob_volume = stage_inputs["prob_volume"]
        depth_hypo = stage_inputs["depth_hypo"]
        depth_gt = depth_gt_ms[stage_key]
        mask = mask_ms[stage_key] > 0.5

        stage_entropy_loss = pixel_wise_loss(prob_volume, depth_gt, mask, depth_hypo)
        stage_dds_loss = depth_distribution_similarity_loss(
            depth_est,
            depth_gt,
            mask,
            depth_min,
            depth_max,
            num_bins = dds_num_bins,
            sigma = dds_sigma
        )

        entropy_stage_weight = _get_stage_weight(entropy_loss_weights, stage_index)
        dds_stage_weight = _get_stage_weight(dds_loss_weights, stage_index)

        entropy_component += entropy_stage_weight * stage_entropy_loss
        dds_component += dds_stage_weight * stage_dds_loss
        total_loss += entropy_weight * entropy_stage_weight * stage_entropy_loss
        total_loss += dds_weight * dds_stage_weight * stage_dds_loss

    return total_loss, entropy_component, dds_component

def build_loss_scalar_outputs(loss_outputs, loss_type, entropy_weight = 1.0, dds_weight = 0.05):
    if loss_type == "casmvs":
        loss, casmvs_component = loss_outputs
        return {
            "loss": loss,
            "casmvs_loss": casmvs_component,
            "depth_loss": casmvs_component,
        }
    if loss_type == "entropy":
        loss, entropy_component = loss_outputs
        return {
            "loss": loss,
            "entropy_loss": entropy_component,
        }
    if loss_type == "casmvs_entropy":
        loss, depth_component, entropy_component = loss_outputs
        return {
            "loss": loss,
            "depth_loss": depth_component,
            "entropy_loss": entropy_component,
            "weighted_entropy_loss": entropy_weight * entropy_component,
        }
    if loss_type == "casmvs_dds":
        loss, depth_component, dds_component = loss_outputs
        return {
            "loss": loss,
            "depth_loss": depth_component,
            "dds_loss": dds_component,
            "weighted_dds_loss": dds_weight * dds_component,
        }
    if loss_type == "entropy_dds":
        loss, entropy_component, dds_component = loss_outputs
        return {
            "loss": loss,
            "entropy_loss": entropy_component,
            "dds_loss": dds_component,
            "weighted_entropy_loss": entropy_weight * entropy_component,
            "weighted_dds_loss": dds_weight * dds_component,
        }
    raise ValueError(f"unsupported loss_type: {loss_type}")
