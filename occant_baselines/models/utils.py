#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import copy

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from einops import repeat


def crop_map(h, x, crop_size, mode="bilinear"):
    r"""
    Crops a tensor h centered around location x with size crop_size

    Inputs:
        h - (bs, F, H, W)
        x - (bs, 2) --- (x, y) locations
        crop_size - scalar integer

    Conventions for x:
        The origin is at the top-left, X is rightward, and Y is downward.
    """

    bs, _, H, W = h.size()
    Hby2 = (H - 1) / 2 if H % 2 == 1 else H // 2
    Wby2 = (W - 1) / 2 if W % 2 == 1 else W // 2
    start = -(crop_size - 1) / 2 if crop_size % 2 == 1 else -(crop_size // 2)
    end = start + crop_size - 1
    x_grid = repeat(
        torch.arange(start, end + 1, step=1), "w -> h w", h=crop_size
    ).float()
    y_grid = repeat(
        torch.arange(start, end + 1, step=1), "h -> h w", w=crop_size
    ).float()
    center_grid = torch.stack([x_grid, y_grid], dim=2).to(
        h.device
    )  # (crop_size, crop_size, 2)

    x_pos = x[:, 0] - Wby2  # (bs, )
    y_pos = x[:, 1] - Hby2  # (bs, )

    crop_grid = repeat(center_grid, "h w 2 -> b h w 2", b=bs)

    # Convert the impact_grid to (-1, 1) range
    crop_grid[:, :, :, 0] = (
                                    crop_grid[:, :, :, 0] + x_pos.unsqueeze(1).unsqueeze(2)
                            ) / Wby2
    crop_grid[:, :, :, 1] = (
                                    crop_grid[:, :, :, 1] + y_pos.unsqueeze(1).unsqueeze(2)
                            ) / Hby2

    h_cropped = F.grid_sample(h, crop_grid, mode=mode)

    return h_cropped


class NormalizedDepth2TopDownViewHabitatTorch:
    def __init__(
            self,
            min_depth,
            max_depth,
            vis_size_h,
            vis_size_w,
            hfov_rad,
            ksize=3,
            rows_around_center=50,
            flag_center_crop=True,
    ):
        self._epsilon = 0.01

        self._min_depth = min_depth
        self._max_depth = max_depth
        self._vis_size_h = vis_size_h
        self._vis_size_w = vis_size_w
        self._hfov_rad = hfov_rad
        self._ksize = ksize
        self._rows_around_center = rows_around_center
        self._flag_center_crop = flag_center_crop

        self._get_intrinsic_mat()

    def gen_top_down_view(self, normalized_depth):

        # normalized_depth: [vis_size_h, vis_size_w, 1]
        depth_no_zero_border, depth_nonzero_infos = self._remove_depth_zero_border(
            normalized_depth
        )
        if torch.numel(depth_no_zero_border) == 0:
            return torch.zeros((self._vis_size_h, self._vis_size_w, 1)).to(
                normalized_depth.device
            )

        # [H, W]
        # Use OpenCV to avoid discrepancy with previous trained model
        new_depth = cv2.GaussianBlur(
            depth_no_zero_border.cpu().numpy(),
            (self._ksize, self._ksize),
            sigmaX=0,
            sigmaY=0,
            borderType=cv2.BORDER_ISOLATED,
        )
        new_depth = torch.FloatTensor(new_depth).to(normalized_depth.device)

        coords_3d = self._compute_coords_3d(new_depth, depth_nonzero_infos[2])
        new_coords_2d = coords_3d[:2, :].clone()

        top_down_cnt, _ = self._cnt_points_in_pixel(new_coords_2d)

        cnt_list = top_down_cnt[top_down_cnt > 0]
        cnt_bound = torch.max(top_down_cnt)

        if torch.max(top_down_cnt) == 0:
            # NOTE: we must have epsilon here since we center crop depth observations,
            # which will result in zero count if the depth is all black around center.
            top_down_view = torch.zeros((self._vis_size_h, self._vis_size_w)).to(
                normalized_depth.device
            )
        else:
            top_down_view = top_down_cnt / cnt_bound
            top_down_view[top_down_view > 1.0] = 1.0

        return top_down_view.unsqueeze(-1)

    def _get_true_depth(self, depth):
        true_depth = depth * (self._max_depth - self._min_depth) + self._min_depth
        return true_depth

    def _get_intrinsic_mat(self):
        u0 = self._vis_size_w / 2
        v0 = self._vis_size_h / 2

        f = (self._vis_size_w / 2) / (np.tan(self._hfov_rad / 2))

        self._K = torch.FloatTensor([[f, 0, u0], [0, f, v0], [0, 0, 1.0]])

    def _get_x_range(self, depth, device):
        # essestially, it is just np.tan(hfov_rad / 2) * depth
        homo_coords_2d = (self._vis_size_w - 0.5, 0, 1)  # right-most coordinate
        coords_3d = (
                torch.matmul(
                    torch.inverse(self._K), torch.FloatTensor(homo_coords_2d).unsqueeze(-1)
                )
                * depth
        )
        coords_3d = coords_3d.to(device)
        return -coords_3d[0], coords_3d[0]

    def _remove_depth_zero_border(self, depth):
        for i in torch.arange(depth.shape[0]):
            if torch.sum(depth[i, :]) > 0:
                break
        min_row = i

        for i in torch.arange(depth.shape[0] - 1, -1, -1):
            if torch.sum(depth[i, :]) > 0:
                break
        max_row = i

        for j in torch.arange(depth.shape[1]):
            if torch.sum(depth[:, j]) > 0:
                break
        min_col = j

        for j in torch.arange(depth.shape[1] - 1, -1, -1):
            if torch.sum(depth[:, j]) > 0:
                break
        max_col = j

        return (
            depth[min_row: (max_row + 1), min_col: (max_col + 1), 0],
            (min_row, max_row, min_col, max_col),
        )

    def _compute_coords_3d(self, depth, min_nonzero_col):
        if self._flag_center_crop:
            # we select pixels around center horizontal line
            min_row = max(
                0, int(np.ceil(depth.shape[0] / 2)) - self._rows_around_center
            )
            max_row = min(
                depth.shape[0],
                int(np.ceil(depth.shape[0] / 2)) + self._rows_around_center,
            )
        else:
            min_row = 0
            max_row = min(self._rows_around_center * 2, depth.shape[0])

        valid_rows = max_row - min_row

        assert valid_rows <= depth.shape[0]

        # (u, v), u for horizontal, v for vertical
        v_coords, u_coords = torch.meshgrid(
            torch.arange(valid_rows).to(depth.device),
            torch.arange(depth.shape[1]).to(depth.device),
        )
        v_coords = v_coords.reshape(-1).float()  # .astype(np.float16)
        u_coords = (
                u_coords.reshape(-1).float() + min_nonzero_col
        )  # .astype(np.float16) + min_nonzero_col

        # add 0.5 to generate 3D points from the center of pixels
        v_coords += 0.5
        u_coords += 0.5

        assert torch.all(v_coords < self._vis_size_h)
        assert torch.all(u_coords < self._vis_size_w)

        # [3, width * height]
        homo_coords_2d = torch.stack(
            [u_coords, v_coords, torch.ones(u_coords.shape).to(depth.device)], dim=0
        )

        coords_3d = torch.matmul(
            torch.inverse(self._K).to(homo_coords_2d.device), homo_coords_2d
        )
        assert torch.all(coords_3d[-1, :] == 1)

        true_depth = self._get_true_depth(depth[min_row:max_row, :]).reshape(-1)

        coords_3d *= true_depth

        # change coordinate configuration from Habitat to normal one: positive-1st: right, positive-2nd: forward, postive-3rd: up
        coords_3d = coords_3d[[0, 2, 1], :]

        # the following is time-consuming
        """
        for i in range(coords_3d.shape[1]):
            tmp_min, tmp_max = get_x_range(coords_3d[1, i], hfov_rad, vis_size_h, vis_size_w)
            assert (
                coords_3d[0, i] >= tmp_min) and (coords_3d[0, i] <= tmp_max
            ), f"{i}, {v_coords[i]}, {u_coords[i]}, {coords_3d[2, i]}, {coords_3d[:, i]}, {tmp_min}, {tmp_max}, {np.tan(hfov_rad / 2) * coords_3d[2, i]}"
        """

        return coords_3d

    def _compute_pixel_coords(self, coords_2d):

        min_x, max_x = self._get_x_range(self._max_depth, device=coords_2d.device)
        x_range = max_x - min_x

        # normalize to [0, 1]
        ndc = coords_2d
        ndc[0, :] = (ndc[0, :] - min_x) / (x_range * (1 + self._epsilon))
        ndc[1, :] = (ndc[1, :] - self._min_depth) / (
                (self._max_depth - self._min_depth) * (1 + self._epsilon)
        )

        # rescale to impact_pixel
        # - in cartesian, origin locates at bottom-left, first element is for horizontal
        # - in image, origin locates at top-left, first element is for row
        pixel_coords = ndc[[1, 0], :]
        pixel_coords[0, :] = self._vis_size_h - torch.ceil(
            self._vis_size_h * pixel_coords[0, :]
        )
        pixel_coords[1, :] = torch.floor(self._vis_size_w * pixel_coords[1, :])
        # assert np.all(pixel_coords >= 0)
        pixel_coords = pixel_coords.long()  # .astype(np.int)

        return pixel_coords

    def _cnt_points_in_pixel(self, coords_2d):

        pixel_coords = self._compute_pixel_coords(coords_2d)

        # unique_pixel_coords: [2, #]
        unique_pixel_coords, unique_cnt = torch.unique(
            pixel_coords, dim=1, sorted=False, return_counts=True
        )
        top_down_cnt = torch.zeros((self._vis_size_h, self._vis_size_w)).to(
            coords_2d.device
        )

        flag1 = unique_pixel_coords[0, :] >= 0
        flag2 = unique_pixel_coords[0, :] < self._vis_size_h
        flag3 = unique_pixel_coords[1, :] >= 0
        flag4 = unique_pixel_coords[1, :] < self._vis_size_w
        # [#points, ]
        valid_flags = torch.all(torch.stack((flag1, flag2, flag3, flag4), dim=0), dim=0)

        cnt_oob_points = unique_pixel_coords.shape[1] - torch.sum(valid_flags)

        top_down_cnt[
            unique_pixel_coords[0, valid_flags], unique_pixel_coords[1, valid_flags]
        ] = unique_cnt.float()[valid_flags]

        return top_down_cnt, cnt_oob_points


class NormalizedDepth2TopDownViewHabitat:
    def __init__(
            self,
            min_depth,
            max_depth,
            vis_size_h,
            vis_size_w,
            hfov_rad,
            ksize=3,
            rows_around_center=50,
            flag_center_crop=True,
    ):
        self._epsilon = 0.01

        self._min_depth = min_depth
        self._max_depth = max_depth
        self._vis_size_h = vis_size_h
        self._vis_size_w = vis_size_w
        self._hfov_rad = hfov_rad
        self._ksize = ksize
        self._rows_around_center = rows_around_center
        self._flag_center_crop = flag_center_crop

        self._get_intrinsic_mat()

    def gen_top_down_view(self, normalized_depth):
        # normalized_depth: [vis_size_h, vis_size_w, 1]
        depth_no_zero_border, depth_nonzero_infos = self._remove_depth_zero_border(
            normalized_depth
        )
        if depth_no_zero_border.size == 0:
            return np.zeros((self._vis_size_h, self._vis_size_w, 1))

        new_depth = cv2.GaussianBlur(
            depth_no_zero_border.astype(np.float32),
            (self._ksize, self._ksize),
            sigmaX=0,
            sigmaY=0,
            borderType=cv2.BORDER_ISOLATED,
        )

        # NOTE: DEBUG
        self._raw_depth = normalized_depth
        self._new_depth = new_depth

        coords_3d = self._compute_coords_3d(new_depth, depth_nonzero_infos[2])
        # new_coords_2d = pickle.loads(pickle.dumps(coords_3d[:2, :]))
        new_coords_2d = copy.deepcopy(coords_3d[:2, :])

        top_down_cnt, _ = self._cnt_points_in_pixel(new_coords_2d)

        cnt_list = top_down_cnt[top_down_cnt > 0].tolist()
        cnt_bound = np.max(cnt_list)

        if np.max(top_down_cnt) == 0:
            top_down_view = np.zeros((self._vis_size_h, self._vis_size_w))
        else:
            top_down_view = top_down_cnt / cnt_bound
            top_down_view[top_down_view > 1.0] = 1.0

        return top_down_view[..., np.newaxis]

    def _get_true_depth(self, depth):
        true_depth = depth * (self._max_depth - self._min_depth) + self._min_depth
        return true_depth

    def _get_intrinsic_mat(self):
        u0 = self._vis_size_w / 2
        v0 = self._vis_size_h / 2

        f = (self._vis_size_w / 2) / (np.tan(self._hfov_rad / 2))

        self._K = np.array([[f, 0, u0], [0, f, v0], [0, 0, 1.0]])

    def _get_x_range(self, depth):
        # essestially, it is just np.tan(hfov_rad / 2) * depth
        homo_coords_2d = (self._vis_size_w - 0.5, 0, 1)  # right-most coordinate
        coords_3d = np.matmul(np.linalg.inv(self._K), homo_coords_2d) * depth
        return -coords_3d[0], coords_3d[0]

    def _remove_depth_zero_border(self, depth):
        for i in np.arange(depth.shape[0]):
            if np.sum(depth[i, :]) > 0:
                break
        min_row = i

        for i in np.arange(depth.shape[0] - 1, -1, -1):
            if np.sum(depth[i, :]) > 0:
                break
        max_row = i

        for j in np.arange(depth.shape[1]):
            if np.sum(depth[:, j]) > 0:
                break
        min_col = j

        for j in np.arange(depth.shape[1] - 1, -1, -1):
            if np.sum(depth[:, j]) > 0:
                break
        max_col = j

        return (
            depth[min_row: (max_row + 1), min_col: (max_col + 1), 0],
            (min_row, max_row, min_col, max_col),
        )

    def _compute_coords_3d(self, depth, min_nonzero_col):
        if self._flag_center_crop:
            # we select pixels around center horizontal line
            min_row = max(
                0, int(np.ceil(depth.shape[0] / 2)) - self._rows_around_center
            )
            max_row = min(
                depth.shape[0],
                int(np.ceil(depth.shape[0] / 2)) + self._rows_around_center,
            )
        else:
            min_row = 0
            max_row = min(self._rows_around_center * 2, depth.shape[0])

        valid_rows = max_row - min_row

        assert valid_rows <= depth.shape[0]

        # (u, v), u for horizontal, v for vertical
        v_coords, u_coords = np.meshgrid(
            np.arange(valid_rows), np.arange(depth.shape[1]), indexing="ij"
        )
        v_coords = v_coords.reshape(-1).astype(np.float16)
        u_coords = u_coords.reshape(-1).astype(np.float16) + min_nonzero_col

        # add 0.5 to generate 3D points from the center of pixels
        v_coords += 0.5
        u_coords += 0.5

        assert np.all(v_coords < self._vis_size_h)
        assert np.all(u_coords < self._vis_size_w)

        # [3, width * height]
        homo_coords_2d = np.array([u_coords, v_coords, np.ones(u_coords.shape)])

        coords_3d = np.matmul(np.linalg.inv(self._K), homo_coords_2d)
        assert np.all(coords_3d[-1, :] == 1)

        true_depth = self._get_true_depth(depth[min_row:max_row, :]).reshape(-1)
        coords_3d *= true_depth

        # change coordinate configuration from Habitat to normal one: positive-1st: right, positive-2nd: forward, postive-3rd: up
        coords_3d = coords_3d[[0, 2, 1], :]

        # the following is time-consuming
        """
        for i in range(coords_3d.shape[1]):
            tmp_min, tmp_max = get_x_range(coords_3d[1, i], hfov_rad, vis_size_h, vis_size_w)
            assert (
                coords_3d[0, i] >= tmp_min) and (coords_3d[0, i] <= tmp_max
            ), f"{i}, {v_coords[i]}, {u_coords[i]}, {coords_3d[2, i]}, {coords_3d[:, i]}, {tmp_min}, {tmp_max}, {np.tan(hfov_rad / 2) * coords_3d[2, i]}"
        """

        return coords_3d

    def _compute_pixel_coords(self, coords_2d):

        min_x, max_x = self._get_x_range(self._max_depth)
        x_range = max_x - min_x

        # normalize to [0, 1]
        ndc = coords_2d
        ndc[0, :] = (ndc[0, :] - min_x) / (x_range * (1 + self._epsilon))
        ndc[1, :] = (ndc[1, :] - self._min_depth) / (
                (self._max_depth - self._min_depth) * (1 + self._epsilon)
        )

        # assert np.all((ndc >= 0) & (ndc < 1)), f"{np.max(ndc[0, :])}, {np.max(coords_3d[0, :])}, {np.max(ndc[1, :])}, {np.max(coords_3d[1, :])}"

        # rescale to impact_pixel
        # - in cartesian, origin locates at bottom-left, first element is for horizontal
        # - in image, origin locates at top-left, first element is for row
        pixel_coords = ndc[[1, 0], :]
        pixel_coords[0, :] = self._vis_size_h - np.ceil(
            self._vis_size_h * pixel_coords[0, :]
        )
        pixel_coords[1, :] = np.floor(self._vis_size_w * pixel_coords[1, :])
        # assert np.all(pixel_coords >= 0)
        pixel_coords = pixel_coords.astype(np.int)

        return pixel_coords

    def _cnt_points_in_pixel(self, coords_2d):
        pixel_coords = self._compute_pixel_coords(coords_2d)

        # unique_pixel_coords: [2, #]
        unique_pixel_coords, unique_cnt = np.unique(
            pixel_coords, axis=1, return_counts=True
        )

        top_down_cnt2 = np.zeros((self._vis_size_h, self._vis_size_w))

        flag1 = unique_pixel_coords[0, :] >= 0
        flag2 = unique_pixel_coords[0, :] < self._vis_size_h
        flag3 = unique_pixel_coords[1, :] >= 0
        flag4 = unique_pixel_coords[1, :] < self._vis_size_w
        # [#points, ]
        valid_flags = np.all(np.array((flag1, flag2, flag3, flag4)), axis=0)

        cnt_oob_points2 = unique_pixel_coords.shape[1] - np.sum(valid_flags)

        top_down_cnt2[
            unique_pixel_coords[0, valid_flags], unique_pixel_coords[1, valid_flags]
        ] = unique_cnt[valid_flags]

        return top_down_cnt2, cnt_oob_points2
