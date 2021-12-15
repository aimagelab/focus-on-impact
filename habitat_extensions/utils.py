#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Dict, Optional, Tuple

import habitat
import numpy as np
import torch
from habitat.core.simulator import (
    Simulator,
)
from habitat.core.utils import try_cv2_import
from habitat.utils.visualizations.maps import COORDINATE_MIN, COORDINATE_MAX
from scipy import stats

from habitat_extensions import maps

cv2 = try_cv2_import()

if habitat.__version__ > "0.1.5":
    FLAG_OLD = False
else:
    FLAG_OLD = True


def draw_collision(view: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    r"""Draw translucent red strips on the border of input view to indicate
    a collision has taken place.
    Args:
        view: input view of size HxWx3 in RGB order.
        alpha: Opacity of red collision strip. 1 is completely non-transparent.
    Returns:
        A view with collision effect drawn.
    """
    strip_width = view.shape[0] // 20
    mask = np.ones(view.shape)
    mask[strip_width:-strip_width, strip_width:-strip_width] = 0
    mask = mask == 1
    view[mask] = (alpha * np.array([255, 0, 0]) + (1.0 - alpha) * view)[mask]
    return view


def truncated_normal_noise_distr(mu, var, width):
    """
    Returns a truncated normal distribution.
    mu - mean of gaussian
    var - variance of gaussian
    width - how much of the normal to sample on either sides of 0
    """
    lower = -width
    upper = width
    sigma = math.sqrt(var)

    X = stats.truncnorm(lower, upper, loc=mu, scale=sigma)

    return X


def observations_to_image(
        observation: Dict, info: Dict, observation_size: Optional[int] = None
) -> np.ndarray:
    r"""Generate image of single frame from observation and info
    returned from a single environment step().

    Args:
        observation: observation returned from an environment step().
        info: info returned from an environment step().

    Returns:
        generated image of a single frame.
    """
    egocentric_view = []
    if "rgb" in observation:
        rgb = observation["rgb"]
        if not isinstance(rgb, np.ndarray):
            rgb = rgb.cpu().numpy()
        if observation_size is None:
            observation_size = observation["rgb"].shape[0]
        else:
            scale = observation_size / rgb.shape[0]
            rgb = cv2.resize(rgb, None, fx=scale, fy=scale)
        egocentric_view.append(rgb)

    # draw depth map if observation has depth info
    if "depth" in observation:
        depth_map = observation["depth"].squeeze() * 255.0
        if not isinstance(depth_map, np.ndarray):
            depth_map = depth_map.cpu().numpy()
        depth_map = depth_map.astype(np.uint8)
        depth_map = np.stack([depth_map for _ in range(3)], axis=2)
        if observation_size is None:
            observation_size = depth_map.shape[0]
        else:
            scale = observation_size / depth_map.shape[0]
            depth_map = cv2.resize(depth_map, None, fx=scale, fy=scale)
        egocentric_view.append(depth_map)

    # add image goal if observation has image_goal info
    if "imagegoal" in observation:
        observation_size = observation["imagegoal"].shape[0]
        rgb = observation["imagegoal"]
        if not isinstance(rgb, np.ndarray):
            rgb = rgb.cpu().numpy()

        egocentric_view.append(rgb)

    assert len(egocentric_view) > 0, "Expected at least one visual sensor enabled."
    egocentric_view = np.concatenate(egocentric_view, axis=1)

    # draw collision
    if "collisions" in info and info["collisions"]["is_collision"]:
        egocentric_view = draw_collision(egocentric_view)

    frame = egocentric_view

    if "top_down_map_exp" in info:
        info["top_down_map"] = info["top_down_map_exp"]

    if "top_down_map" in info:
        top_down_map = topdown_to_image(info["top_down_map"])
        scale = observation_size / top_down_map.shape[0]
        top_down_map = cv2.resize(top_down_map, None, fx=scale, fy=scale)
        frame = np.concatenate((egocentric_view, top_down_map), axis=1)

    return frame


def topdown_to_image(topdown_info: np.ndarray) -> np.ndarray:
    r"""Convert topdown map to an RGB image.
    """
    top_down_map = topdown_info["map"]
    fog_of_war_mask = topdown_info["fog_of_war_mask"]
    top_down_map = maps.colorize_topdown_map(top_down_map, fog_of_war_mask)
    map_agent_pos = topdown_info["agent_map_coord"]

    # Add zero padding
    min_map_size = 200
    if top_down_map.shape[0] != top_down_map.shape[1]:
        H = top_down_map.shape[0]
        W = top_down_map.shape[1]
        if H > W:
            pad_value = (H - W) // 2
            padding = ((0, 0), (pad_value, pad_value), (0, 0))
            map_agent_pos = (map_agent_pos[0], map_agent_pos[1] + pad_value)
        else:
            pad_value = (W - H) // 2
            padding = ((pad_value, pad_value), (0, 0), (0, 0))
            map_agent_pos = (map_agent_pos[0] + pad_value, map_agent_pos[1])
        top_down_map = np.pad(
            top_down_map, padding, mode="constant", constant_values=255
        )

    if top_down_map.shape[0] < min_map_size:
        H, W = top_down_map.shape[:2]
        top_down_map = cv2.resize(top_down_map, (min_map_size, min_map_size))
        map_agent_pos = (
            int(map_agent_pos[0] * min_map_size // H),
            int(map_agent_pos[1] * min_map_size // W),
        )
    top_down_map = maps.draw_agent(
        image=top_down_map,
        agent_center_coord=map_agent_pos,
        agent_rotation=topdown_info["agent_angle"],
        agent_radius_px=top_down_map.shape[0] // 40,
    )

    return top_down_map


def compute_confusion_matrix(true, pred):
    """Computes a confusion matrix using numpy for two np.arrays
  true and pred.
    """

    true = true + 1
    pred = pred + 1
    K = 3
    result = torch.zeros((K, K))

    for i in range(K):
        for j in range(K):
            result[i][j] = torch.sum(((true == i) * (pred == j)).float())

    return result


def compute_iou_acc(true, pred, eps=1e-8):
    '''
    class -1 : removed object
    class 1 : added object
    class 0 : unchanged

    returns batch wise IoU and accuracy '''

    assert pred.shape == true.shape
    batch, h, w = pred.shape

    # compute class masks
    added_gt = (true == 1)
    removed_gt = (true == -1)

    added_pred = (pred == 1)
    removed_pred = (pred == -1)

    # batch wise iou metrics
    added_intersection, added_union, removed_intersection, removed_union = [], [], [], []

    for b in range(batch):
        added_intersection.append(torch.logical_and(added_gt[b, ...], added_pred[b, ...]))
        added_union.append(torch.logical_or(added_gt[b, ...], added_pred)[b, ...])

        removed_intersection.append(torch.logical_and(removed_gt[b, ...], removed_pred[b, ...]))
        removed_union.append(torch.logical_or(removed_gt[b, ...], removed_pred[b, ...]))

    added_intersection = torch.stack(added_intersection).float()
    added_union = torch.stack(added_union).float()
    removed_intersection = torch.stack(removed_intersection).float()
    removed_union = torch.stack(removed_union).float()

    batch_sum_ai = torch.sum(added_intersection, dim=(1, 2))
    batch_sum_au = torch.sum(added_union, dim=(1, 2))
    batch_sum_ri = torch.sum(removed_intersection, dim=(1, 2))
    batch_sum_ru = torch.sum(removed_union, dim=(1, 2))

    added_iou = batch_sum_ai / (batch_sum_au + eps)
    removed_iou = batch_sum_ri / (batch_sum_ru + eps)
    total_iou = (batch_sum_ai + batch_sum_ri) / (batch_sum_au + batch_sum_ru + eps)

    # batch wise accuracy
    total = torch.sum(added_gt.float(), dim=(1, 2)) + torch.sum(removed_gt.float(), dim=(1, 2)) + eps
    correct = batch_sum_ai + batch_sum_ri
    accuracy = correct / total

    return added_iou, removed_iou, total_iou, accuracy


def modified_get_topdown_map(
        sim: Simulator,
        map_resolution: Tuple[int, int] = (1250, 1250),
        num_samples: int = 20000,
        draw_border: bool = True,
) -> np.ndarray:
    r"""Return a top-down occupancy map for a sim. Note, this only returns valid
    values for whatever floor the agent is currently on.
    Args:
        sim: The simulator.
        map_resolution: The resolution of map which will be computed and
            returned.
        num_samples: The number of random navigable points which will be
            initially
            sampled. For large environments it may need to be increased.
        draw_border: Whether to outline the border of the occupied spaces.
    Returns:
        Image containing 0 if occupied, 1 if unoccupied, and 2 if border (if
        the flag is set).
    """
    top_down_map = np.zeros(map_resolution, dtype=np.uint8)
    border_padding = 3

    start_height = sim.get_agent_state().position[1]

    cur_coordinate_min = -1 * COORDINATE_MIN
    cur_coordinate_max = -1 * COORDINATE_MAX

    # Use sampling to find the extrema points that might be navigable.
    range_x = (map_resolution[0], 0)
    range_y = (map_resolution[1], 0)

    point_list = []
    grid_point_list = []

    for _ in range(num_samples):
        point = sim.sample_navigable_point()
        # Check if on same level as original
        if np.abs(start_height - point[1]) > 0.5:
            continue

        point_list.append(point)

        cur_coordinate_min = min([cur_coordinate_min, point[0], point[2]])
        cur_coordinate_max = max([cur_coordinate_max, point[0], point[2]])

    for i in range(len(point_list)):
        point = point_list[i]
        if FLAG_OLD:
            g_x, g_y = maps.to_grid(
                point[0],
                point[2],
                cur_coordinate_min,
                cur_coordinate_max,
                map_resolution,
            )
        else:
            g_x, g_y = maps.to_grid(point[0], point[2], map_resolution, sim=sim, )
        range_x = (min(range_x[0], g_x), max(range_x[1], g_x))
        range_y = (min(range_y[0], g_y), max(range_y[1], g_y))

    # Pad the range just in case not enough points were sampled to get the true
    # extrema.
    padding = int(np.ceil(map_resolution[0] / 125))
    range_x = (
        max(range_x[0] - padding, 0),
        min(range_x[-1] + padding + 1, top_down_map.shape[0]),
    )
    range_y = (
        max(range_y[0] - padding, 0),
        min(range_y[-1] + padding + 1, top_down_map.shape[1]),
    )

    # Search over impact_grid for valid points.
    for ii in range(range_x[0], range_x[1]):
        for jj in range(range_y[0], range_y[1]):
            if FLAG_OLD:
                realworld_x, realworld_y = maps.from_grid(
                    ii, jj, cur_coordinate_min, cur_coordinate_max, map_resolution
                )
            else:
                realworld_x, realworld_y = maps.from_grid(
                    ii, jj, map_resolution, sim=sim,
                )
            valid_point = sim.is_navigable([realworld_x, start_height, realworld_y])
            top_down_map[ii, jj] = (
                maps.MAP_VALID_POINT if valid_point else maps.MAP_INVALID_POINT
            )

    # Draw border if necessary
    if draw_border:
        # Recompute range in case padding added any more values.
        range_x = np.where(np.any(top_down_map, axis=1))[0]
        range_y = np.where(np.any(top_down_map, axis=0))[0]
        range_x = (
            max(range_x[0] - border_padding, 0),
            min(range_x[-1] + border_padding + 1, top_down_map.shape[0]),
        )
        range_y = (
            max(range_y[0] - border_padding, 0),
            min(range_y[-1] + border_padding + 1, top_down_map.shape[1]),
        )

        maps._outline_border(
            top_down_map[range_x[0]: range_x[1], range_y[0]: range_y[1]]
        )

    return top_down_map, cur_coordinate_min, cur_coordinate_max


def modified_to_grid(
        realworld_x: float,
        realworld_y: float,
        coordinate_min: float,
        coordinate_max: float,
        grid_resolution: Tuple[int, int],
) -> Tuple[int, int]:
    r"""Return gridworld index of realworld coordinates assuming top-left corner
    is the origin. The real world coordinates of lower left corner are
    (coordinate_min, coordinate_min) and of top right corner are
    (coordinate_max, coordinate_max)
    """
    grid_size = (
        (coordinate_max - coordinate_min) / grid_resolution[0],
        (coordinate_max - coordinate_min) / grid_resolution[1],
    )
    grid_x = min(
        int((coordinate_max - realworld_x) / grid_size[0]), grid_resolution[0] - 1
    )
    grid_y = min(
        int((realworld_y - coordinate_min) / grid_size[1]), grid_resolution[1] - 1
    )
    return grid_x, grid_y
