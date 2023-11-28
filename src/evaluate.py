import torch
import math


def evaluate_error(gt_depth, pred_depth):
    assert type(gt_depth) == torch.Tensor, "gt_depth should be torch.Tensor"
    assert gt_depth.shape(1) == 1, "gt_depth should be (N, 1, H, W)"

    assert type(pred_depth) == torch.Tensor, "pred_depth should be torch.Tensor"
    assert pred_depth.shape(1) == 1, "pred_depth should be (N, 1, H, W)"

    assert gt_depth.shape == pred_depth.shape, "gt_depth and pred_depth should have the same shape"

    # for numerical stability
    depth_mask = gt_depth > 0.0001
    error = {
        "MSE": 0,
        "RMSE": 0,
        "ABS_REL": 0,
        "LG10": 0,
        "MAE": 0,
        "DELTA1.02": 0,
        "DELTA1.05": 0,
        "DELTA1.10": 0,
        "DELTA1.25": 0,
        "DELTA1.25^2": 0,
        "DELTA1.25^3": 0,
    }
    _pred_depth = pred_depth[depth_mask]
    _gt_depth = gt_depth[depth_mask]
    n_valid_element = _gt_depth.size(0)

    assert type(n_valid_element) == int, "n_valid_element should be int"

    if n_valid_element > 0:
        diff_mat = torch.abs(_gt_depth - _pred_depth)
        rel_mat = torch.div(diff_mat, _gt_depth)

        error["MSE"] = torch.sum(torch.pow(diff_mat, 2)) / n_valid_element
        error["RMSE"] = math.sqrt(error["MSE"])
        error["MAE"] = torch.sum(diff_mat) / n_valid_element
        error["ABS_REL"] = torch.sum(rel_mat) / n_valid_element

        y_over_z = torch.div(_gt_depth, _pred_depth)
        z_over_y = torch.div(_pred_depth, _gt_depth)
        max_ratio = torch.max(y_over_z, z_over_y)

        error["DELTA1.02"] = torch.sum(max_ratio < 1.02).numpy() / float(n_valid_element)
        error["DELTA1.05"] = torch.sum(max_ratio < 1.05).numpy() / float(n_valid_element)
        error["DELTA1.10"] = torch.sum(max_ratio < 1.10).numpy() / float(n_valid_element)
        error["DELTA1.25"] = torch.sum(max_ratio < 1.25).numpy() / float(n_valid_element)
        error["DELTA1.25^2"] = torch.sum(max_ratio < 1.25**2).numpy() / float(n_valid_element)
        error["DELTA1.25^3"] = torch.sum(max_ratio < 1.25**3).numpy() / float(n_valid_element)
    return error
