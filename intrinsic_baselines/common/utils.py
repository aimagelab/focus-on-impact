import copy
import numbers
import os

import cv2
import numpy as np
import quaternion
import torch
import torch.nn as nn
import torch_scatter
from PIL import ImageDraw, ImageFont
from habitat import logger
from habitat.utils.geometry_utils import quaternion_rotate_vector
from habitat_baselines.common.utils import batch_obs, overwrite_gym_box_shape, image_resize_shortest_edge
from torchvision.transforms import transforms

mapping = {'0': 'A',
           '1': 'L',
           '2': 'R',
           '3': 'Start',
           }


def group_observations(grouped_observations, step_observations):
    for k in step_observations.keys():
        if k in grouped_observations.keys():
            grouped_observations[k] = torch.cat(
                (grouped_observations[k],
                 step_observations[k].unsqueeze(0)),
                0)
        else:
            grouped_observations[k] = step_observations[k].unsqueeze(
                0)
    return grouped_observations


def initialize_action_counters(num_envs):
    ahead = [0] * num_envs
    turn = [0] * num_envs
    left = [0] * num_envs
    right = [0] * num_envs
    stop = [0] * num_envs
    return ahead, turn, left, right, stop


def actions_count(actions, ahead, turn, left, right, stop):
    for i, a in enumerate(actions):
        print(mapping[str(a[0].item())])
        if a[0].item() == 0:
            ahead[i] += 1
            turn[i] = 0
            left[i] = 0
            right[i] = 0
            stop[i] = 0
        if a[0].item() != 0 and a[0].item() != 3:
            turn[i] += 1
            ahead[i] = 0
            stop[i] = 0
        if a[0].item() == 1:
            left[i] += 1
            right[i] = 0
            ahead[i] = 0
            stop[i] = 0
        if a[0].item() == 2:
            right[i] += 1
            left[i] = 0
            ahead[i] = 0
            stop[i] = 0
        if a[0].item() == 3:
            stop[i] += 1
            turn[i] = 0
            left[i] = 0
            right[i] = 0
            ahead[i] = 0
        return ahead, turn, left, right, stop


def save_img_and_caption(img, caption, reward, opt=None, name=None):
    save_preprocess = transforms.ToPILImage()
    saved_image = save_preprocess(img.squeeze(0).permute(2, 0, 1))
    draw = ImageDraw.Draw(saved_image)
    font = ImageFont.truetype(
        os.path.join(os.path.dirname(__file__), 'captioning/times-ro.ttf'), 12)
    text = caption + " ({:.3f})".format(reward)
    if opt is not None:
        text = text + " ({:.3f})".format(opt)
    draw.text((20, img.size(0) - 20), text, (0, 0, 0), font=font)
    count = len([name for name in os.listdir(
        os.path.join(os.path.dirname(__file__),
                     'captioning/saved_img_and_captions/'))])
    if os.path.isfile(os.path.join(os.path.dirname(__file__),
                                   'captioning/saved_img_and_captions/best.png')):
        count = count - 1
    if os.path.isfile(os.path.join(os.path.dirname(__file__),
                                   'captioning/saved_img_and_captions/worst.png')):
        count = count - 1
    if name is not None:
        fname = 'captioning/saved_img_and_captions/' + name
    else:
        fname = 'captioning/saved_img_and_captions/img_{}.png'.format(
            count)
    saved_image.save(os.path.join(os.path.dirname(__file__),
                                  fname))
    # if count == 500:
    #     sys.exit()


def preprocessing_pixel_rgb(config, observations, cpu=True):
    resized_dim = config.RL.PPO.DENSITY_MODEL.pixel_resized_dim
    num_buckets = config.RL.PPO.DENSITY_MODEL.num_buckets
    images = torch.stack([torch.tensor(obs['rgb']) for obs in observations])
    crop_size = min(images.shape[1], images.shape[2])

    preprocess = transforms.Compose(
        [transforms.ToPILImage(),
         transforms.CenterCrop(crop_size),
         transforms.Resize(size=(resized_dim, resized_dim)),
         transforms.Grayscale(1),
         ])
    if cpu:
        processed_images = [preprocess(img.permute(2, 0, 1).cpu()) for
                            img
                            in images]
    else:
        processed_images = [preprocess(img.permute(2, 0, 1)) for img in
                            images]

    buckets = np.linspace(0, 256.0, num_buckets + 1)
    processed_images = [
        (torch.Tensor((np.digitize(np.array(img).reshape(1, img.size[0], img.size[1]), buckets) - 1) / num_buckets))
        for img in
        processed_images]

    return torch.stack(processed_images)


def preprocessing_curious_rgb(config, images, cpu=False):
    resized_dim = config.RL.ANS.obs_resized_dim
    crop_size = min(images.shape[1], images.shape[2])

    preprocess = transforms.Compose(
        [transforms.ToPILImage(),
         transforms.CenterCrop(crop_size),
         transforms.Resize(size=(resized_dim, resized_dim)),
         transforms.Grayscale(1), ])
    if cpu:
        processed_images = [preprocess(img.permute(2, 0, 1).cpu()) for
                            img
                            in images]
    else:
        processed_images = [preprocess(img.permute(2, 0, 1)) for img in
                            images]

    return torch.stack(
        [(torch.Tensor(np.array(img).reshape(img.size[0], img.size[1], 1)))
         for img in
         processed_images])


def preprocessing_curious_depth(config, images, cpu=False):
    resized_dim = config.RL.ANS.obs_resized_dim
    crop_size = min(images.shape[1], images.shape[2])

    preprocess = transforms.Compose(
        [transforms.ToPILImage(),
         transforms.CenterCrop(crop_size),
         transforms.Resize(size=(resized_dim, resized_dim)),
         transforms.ToTensor()
         ])

    if cpu:
        processed_images = [preprocess(img.permute(2, 0, 1).cpu()) for
                            img
                            in images]
    else:
        processed_images = [preprocess(img.permute(2, 0, 1)) for img in
                            images]

    return torch.stack([img.permute(1, 2, 0)
                        for img in
                        processed_images])


def preprocess_curiosity_observations(config, observation):
    if 'rgb' in observation.keys():
        observation['rgb'] = preprocessing_curious_rgb(config, observation['rgb'])
    if 'depth' in observation.keys():
        observation['depth'] = preprocessing_curious_depth(config, observation['depth'])
    return observation


def append_observations(config, curiosity_obs, observation, device):
    batch = batch_obs(observation)
    batch_pre = preprocess_curiosity_observations(config, batch)
    new_curiosity_obs = {
        'rgb': curiosity_obs['rgb'].clone().detach(),
        'depth': curiosity_obs['depth'].clone().detach(),
    }
    if 'rgb' in batch_pre.keys():
        new_curiosity_obs['rgb'] = torch.cat((new_curiosity_obs['rgb'][..., 1:], batch_pre['rgb']), dim=-1).detach()
    if 'depth' in batch_pre.keys():
        new_curiosity_obs['depth'] = torch.cat((new_curiosity_obs['depth'][..., 1:], batch_pre['depth']),
                                               dim=-1).detach()
    return new_curiosity_obs


def convert_to_pointcloud(sim_cfg, sensor_cfg, depth):
    """
    Inputs:
        depth = (H, W, 1) numpy array

    Returns:
        xyz_camera = (N, 3) numpy array for (X, Y, Z) in egocentric world coordinates
    """
    min_depth = float(sim_cfg.DEPTH_SENSOR.MIN_DEPTH)
    max_depth = float(sim_cfg.DEPTH_SENSOR.MAX_DEPTH)

    W = sim_cfg.DEPTH_SENSOR.WIDTH
    H = sim_cfg.DEPTH_SENSOR.HEIGHT
    proj_xs, proj_ys = np.meshgrid(
        np.linspace(-1, 1, W), np.linspace(1, -1, H)
    )

    hfov = float(sim_cfg.DEPTH_SENSOR.HFOV) * np.pi / 180
    vfov = 2 * np.arctan((H / W) * np.tan(hfov / 2.0))
    intrinsic_matrix = np.array(
        [
            [1 / np.tan(hfov / 2.0), 0.0, 0.0, 0.0],
            [0.0, 1 / np.tan(vfov / 2.0), 0.0, 0.0],
            [0.0, 0.0, 1, 0],
            [0.0, 0.0, 0, 1],
        ]
    )
    inverse_intrinsic_matrix = np.linalg.inv(intrinsic_matrix)

    map_size = sensor_cfg.MAP_SIZE
    map_scale = sensor_cfg.MAP_SCALE

    if sim_cfg.DEPTH_SENSOR.NORMALIZE_DEPTH:
        depth = depth * (max_depth - min_depth) + min_depth

    if sensor_cfg.MAX_SENSOR_RANGE > 0:
        max_forward_range = sensor_cfg.MAX_SENSOR_RANGE
    else:
        max_forward_range = map_size * map_scale

    depth_float = depth.astype(np.float32)[..., 0]

    # =========== Convert to camera coordinates ============
    W = depth.shape[1]
    xs = np.copy(proj_xs).reshape(-1)
    ys = np.copy(proj_ys).reshape(-1)
    depth_float = depth_float.reshape(-1)
    # Filter out invalid depths
    valid_depths = (depth_float != min_depth) & (
            depth_float <= max_forward_range
    )
    xs = xs[valid_depths]
    ys = ys[valid_depths]
    depth_float = depth_float[valid_depths]
    # Unproject
    # negate depth as the camera looks along -Z
    xys = np.vstack(
        (
            xs * depth_float,
            ys * depth_float,
            -depth_float,
            np.ones(depth_float.shape),
        )
    )
    inv_K = inverse_intrinsic_matrix
    xyz_camera = np.matmul(inv_K, xys).T  # XYZ in the camera coordinate system
    xyz_camera = xyz_camera[:, :3] / xyz_camera[:, 3][:, np.newaxis]

    return xyz_camera


def get_depth_projection(sim_cfg, sensor_cfg, point_cloud):
    """
    Project pixels visible in depth-map to ground-plane
    """

    camera_height = sim_cfg.DEPTH_SENSOR.POSITION[1]
    map_size = sensor_cfg.MAP_SIZE
    map_scale = sensor_cfg.MAP_SCALE
    height_thresh = sensor_cfg.HEIGHT_THRESH

    XYZ_ego = point_cloud

    # Adding agent's height to the pointcloud
    XYZ_ego[:, 1] += camera_height

    # Convert to impact_grid coordinate system
    V = map_size
    Vby2 = V // 2

    points = XYZ_ego

    grid_x = (points[:, 0] / map_scale) + Vby2
    grid_y = (points[:, 2] / map_scale) + V

    # Filter out invalid points
    valid_idx = (
            (grid_x >= 0) & (grid_x <= V - 1) & (grid_y >= 0) & (grid_y <= V - 1)
    )
    points = points[valid_idx, :]
    grid_x = grid_x[valid_idx].astype(int)
    grid_y = grid_y[valid_idx].astype(int)

    # Create empty maps for the two channels
    obstacle_mat = np.zeros((map_size, map_size), np.uint8)
    explore_mat = np.zeros((map_size, map_size), np.uint8)

    # Compute obstacle locations
    high_filter_idx = points[:, 1] < height_thresh[1]
    low_filter_idx = points[:, 1] > height_thresh[0]
    obstacle_idx = np.logical_and(low_filter_idx, high_filter_idx)

    safe_assign(obstacle_mat, grid_y[obstacle_idx], grid_x[obstacle_idx], 1)
    kernel = np.ones((3, 3), np.uint8)
    obstacle_mat = cv2.dilate(obstacle_mat, kernel, iterations=1)

    # Compute explored locations
    explored_idx = high_filter_idx
    safe_assign(explore_mat, grid_y[explored_idx], grid_x[explored_idx], 1)
    kernel = np.ones((3, 3), np.uint8)
    obstacle_mat = cv2.dilate(obstacle_mat, kernel, iterations=1)

    # Smoothen the maps
    kernel = np.ones((3, 3), np.uint8)

    obstacle_mat = cv2.morphologyEx(obstacle_mat, cv2.MORPH_CLOSE, kernel)
    explore_mat = cv2.morphologyEx(explore_mat, cv2.MORPH_CLOSE, kernel)

    # Ensure all expanded regions in obstacle_mat are accounted for in explored_mat
    explore_mat = np.logical_or(explore_mat, obstacle_mat)

    return np.stack([obstacle_mat, explore_mat], axis=2)


def get_depth_projection_unfiltered(sim_cfg, sensor_cfg, point_cloud):
    """
    Project pixels visible in depth-map to ground-plane
    """

    camera_height = sim_cfg.DEPTH_SENSOR.POSITION[1]
    map_size = sensor_cfg.MAP_SIZE
    map_scale = sensor_cfg.MAP_SCALE
    height_thresh = sensor_cfg.HEIGHT_THRESH

    XYZ_ego = point_cloud

    # Adding agent's height to the pointcloud
    XYZ_ego[:, 1] += camera_height

    # Convert to impact_grid coordinate system
    V = map_size
    Vby2 = V // 2

    points = XYZ_ego

    grid_x = (points[:, 0] / map_scale) + Vby2
    grid_y = (points[:, 2] / map_scale) + V

    # Filter out invalid points
    valid_idx = (
            (grid_x >= 0) & (grid_x <= V - 1) & (grid_y >= 0) & (grid_y <= V - 1)
    )
    points = points[valid_idx, :]
    grid_x = grid_x[valid_idx].astype(int)
    grid_y = grid_y[valid_idx].astype(int)

    # Create empty maps for the two channels
    heights_mat = np.zeros((map_size, map_size), np.uint8)
    explore_mat = np.zeros((map_size, map_size), np.uint8)

    # Compute obstacle locations
    high_filter_idx = points[:, 1] < height_thresh[1]
    low_filter_idx = points[:, 1] > height_thresh[0]
    obstacle_idx = np.logical_and(low_filter_idx, high_filter_idx)

    heights_mat = safe_assign_unfiltered(heights_mat, grid_y, grid_x, points[:, 1])

    # Compute explored locations
    explored_idx = high_filter_idx
    safe_assign(explore_mat, grid_y[explored_idx], grid_x[explored_idx], 1)

    # Ensure all expanded regions in obstacle_mat are accounted for in explored_mat
    explore_mat = np.logical_or(explore_mat, heights_mat.astype(np.bool))

    return np.stack([heights_mat, explore_mat], axis=2)


def safe_assign(im_map, x_idx, y_idx, value):
    try:
        im_map[x_idx, y_idx] = value
    except IndexError:
        valid_idx1 = np.logical_and(x_idx >= 0, x_idx < im_map.shape[0])
        valid_idx2 = np.logical_and(y_idx >= 0, y_idx < im_map.shape[1])
        valid_idx = np.logical_and(valid_idx1, valid_idx2)
        im_map[x_idx[valid_idx], y_idx[valid_idx]] = value


def safe_assign_unfiltered(im_map, x_idx, y_idx, value, type="max"):
    unrolled_map = torch.zeros(im_map.size).double()
    index = torch.tensor(x_idx * im_map.shape[0] + y_idx)
    if type == "max":
        torch_scatter.scatter_max(torch.tensor(value), index, out=unrolled_map)
    elif type == "mean":
        torch_scatter.scatter_mean(torch.tensor(value), index, out=unrolled_map)
    unrolled_map = unrolled_map.reshape((im_map.shape[0], im_map.shape[1])).cpu().numpy()
    return unrolled_map


def stitch_images(batch, prev_batch, actions, feature_detector):
    # Image Stitching
    if actions[0] == 1:
        left = batch['rgb'][0].type(torch.uint8).cpu().numpy()
        left_depth = batch['depth'][0].cpu().numpy()
        right = prev_batch['rgb'][0].type(torch.uint8).cpu().numpy()
        right_depth = prev_batch['depth'][0].cpu().numpy()
        rightg = cv2.cvtColor(right, cv2.COLOR_RGB2GRAY)
    else:
        right = batch['rgb'][0].type(torch.uint8).cpu().numpy()
        right_depth = batch['depth'][0].cpu().numpy()
        left = prev_batch['rgb'][0].type(torch.uint8).cpu().numpy()
        left_depth = prev_batch['depth'][0].cpu().numpy()
        rightg = cv2.cvtColor(np.pad(right, [(0, 0), (right.shape[1] // 2, 0), (0, 0)]), cv2.COLOR_RGB2GRAY)
    leftg = cv2.cvtColor(left, cv2.COLOR_RGB2GRAY)

    kp_left, des_left = feature_detector.detectAndCompute(leftg, None)
    kp_right, des_right = feature_detector.detectAndCompute(rightg, None)

    img1 = cv2.drawKeypoints(left, kp_left, None, color=(0, 255, 0), flags=0)
    # plt.imshow(img1)
    # plt.show()

    img2 = cv2.drawKeypoints(right, kp_right, None, color=(0, 0, 255), flags=0)
    # plt.imshow(img2)
    # plt.show()

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des_left, des_right, k=2)

    # keep best matches
    good = []
    for m in matches:
        if m[0].distance < 0.5 * m[1].distance:
            good.append(m)
    matches = np.asarray(good)

    if len(matches[:, 0]) >= 4:
        if actions[0] == 1:
            src = np.float32([kp_right[m.trainIdx].pt for m in matches[:, 0]]).reshape(-1, 1, 2)
            dst = np.float32([kp_left[m.queryIdx].pt for m in matches[:, 0]]).reshape(-1, 1, 2)
        else:
            src = np.float32([kp_left[m.queryIdx].pt for m in matches[:, 0]]).reshape(-1, 1, 2)
            dst = np.float32([kp_right[m.trainIdx].pt for m in matches[:, 0]]).reshape(-1, 1, 2)
        H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    else:
        # raise AssertionError("Can't find enough keypoints")
        return 0

    if H is None:
        return 0

    if actions[0] == 0:
        dst = cv2.warpPerspective(right, H, (left.shape[1], right.shape[0]))
        dst[0:left.shape[0], 0:left.shape[1]] = left
        dst = np.pad(dst, [(0, 0), (left.shape[1] // 2, right.shape[1] // 2), (0, 0)])
    elif actions[0] == 1:
        dst = cv2.warpPerspective(right, H, (right.shape[1] + left.shape[1] // 2, left.shape[0]))
        dst[0:left.shape[0], 0:left.shape[1]] = left
        dst = np.pad(dst, [(0, 0), (left.shape[1] // 2, 0), (0, 0)])
    else:
        dst = cv2.warpPerspective(left, H, (left.shape[1] + right.shape[1] // 2, left.shape[0]))
        dst[0:left.shape[0], left.shape[1] // 2:] = right
        dst = np.pad(dst, [(0, 0), (0, right.shape[1] // 2), (0, 0)])
    cv2.imwrite('output.jpg', dst)
    # plt.imshow(dst)
    # plt.show()

    return dst


def rotation_matrix_to_euler_angles(R):
    xyz = np.zeros((R.shape[0], 3))
    for i, rot in enumerate(R):

        assert (is_rotation_matrix(rot))

        sy = np.sqrt(rot[0, 0] * rot[0, 0] + rot[1, 0] * rot[1, 0])

        singular = sy < 1e-6

        if not singular:
            x = np.arctan2(rot[2, 1], rot[2, 2])
            y = np.arctan2(-rot[2, 0], sy)
            z = np.arctan2(rot[1, 0], rot[0, 0])
        else:
            x = np.arctan2(-rot[1, 2], rot[1, 1])
            y = np.arctan2(-rot[2, 0], sy)
            z = 0
        xyz[i] = np.array([x, y, z])

    return np.stack(xyz, axis=0)


# Checks if a matrix is a valid rotation matrix.
def is_rotation_matrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


class Resizer(nn.Module):
    def __init__(self, size, channels_last: bool = False):
        r"""An nn module the resizes and center crops your input.
        Args:
            size: A sequence (w, h) or int of the size you wish to resize/center_crop.
                    If int, assumes square crop
            channels_list: indicates if channels is the last dimension
        """
        super().__init__()
        if isinstance(size, numbers.Number):
            size = (int(size), int(size))
        assert len(size) == 2, "forced input size must be len of 2 (w, h)"
        self._size = size
        self.channels_last = channels_last

    def transform_observation_space(
            self, observation_space, trans_keys=["rgb", "depth", "semantic"]
    ):
        size = self._size
        observation_space = copy.deepcopy(observation_space)
        if size:
            for key in observation_space.spaces:
                if key in trans_keys and observation_space.spaces[key].shape != size:
                    logger.info("Overwriting CNN input size of %s: %s" % (key, size))
                    observation_space.spaces[key] = overwrite_gym_box_shape(
                        observation_space.spaces[key], size
                    )
        self.observation_space = observation_space
        return observation_space

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self._size is None:
            return input

        return image_resize_shortest_edge(
            input, min(self._size), channels_last=self.channels_last
        )


def compute_goal_pos(prev_goal_pos, local_delta_state):
    r"""Compute goal position w.r.t local coordinate system at time t + 1 from
    - goal position w.r.t. local coordinate system at time t
    - state changes in local coordinate system at time t.

    Assume prev_goal_pos as prev_v_g.
    Meanwhile, set local_delta_pos and local_delta_rot as v and q respectively.

    cur_v_g = q^{-1} * (prev_v_g - v) * q

    :param prev_goal_pos: np.array
    :param local_delta_state: [dx, dz, dyaw]
    """
    dx, dz, dyaw = local_delta_state

    local_pos = np.array([dx, 0.0, dz])
    local_delta_quaternion = quat_from_angle_axis(
        theta=dyaw, axis=np.array([0, 1.0, 0])
    )
    cur_goal_pos = quaternion_rotate_vector(
        local_delta_quaternion.inverse(), prev_goal_pos - local_pos,
    )

    rho, phi = cartesian_to_polar(-cur_goal_pos[2], cur_goal_pos[0])

    out_dict = {
        "cartesian": cur_goal_pos,
        "polar": np.array([rho, -phi], dtype=np.float32),
    }
    return out_dict


def quat_from_angle_axis(theta: float, axis: np.ndarray) -> np.quaternion:
    r"""Creates a quaternion from angle axis format
    :param theta: The angle to rotate about the axis by
    :param axis: The axis to rotate about
    :return: The quaternion
    """
    axis = axis.astype(np.float)
    axis /= np.linalg.norm(axis)
    return quaternion.from_rotation_vector(theta * axis)


def cartesian_to_polar(x, y):
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return rho, phi
