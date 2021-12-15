#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
from typing import Type, Union

import habitat
import numpy as np
import torch
from habitat import Config, Env, RLEnv, VectorEnv, make_dataset
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import quaternion_rotate_vector, quaternion_from_coeff
from habitat_baselines.common.env_utils import make_env_fn

from occant_utils.common import spatial_transform_map


def transform_map(map_scale, p, x, invert=False):
    """
    Given the locally computed map, register it to the global map based
    on the current position.

    Inputs:
        m - (bs, F, M, M) new map
        p - (bs, F, M, M) old map
        x - (bs, 3) in global coordinates
    """
    # Register the local map
    p_trans = spatial_transform(map_scale, p, x, invert=invert)
    return p_trans


def spatial_transform(map_scale, p, dx, invert=False):
    """
    Applies the transformation dx to image p.
    Inputs:
        p - (bs, 2, H, W) map
        dx - (bs, 3) egocentric transformation --- (dx, dy, dtheta)

    Conventions:
        The origin is at the center of the map.
        X is upward with agent's forward direction
        Y is rightward with agent's rightward direction

    Note: These denote transforms in an agent's position. Not the image directly.
    For example, if an agent is moving upward, then the map will be moving downward.
    To disable this behavior, set invert=False.
    """
    s = map_scale
    # Convert dx to map image coordinate system with X as rightward and Y as downward
    dx_map = torch.stack(
        [(dx[:, 1] / s), -(dx[:, 0] / s), dx[:, 2]], dim=1
    )  # anti-clockwise rotation

    # insert context manager hopefully fixes bug
    # with sequence():
    p_trans = spatial_transform_map(p, dx_map, invert=invert)

    return p_trans


def get_position_vector_from_obs(observations):
    origin = np.array(observations[0]['start_coords'][:3], dtype=np.float32)
    rotation_world_start = quaternion_from_coeff(observations[0]['start_coords'][3:])

    agent_position = quaternion_rotate_vector(
        rotation_world_start, origin
    )

    rotation_world_start = quaternion_from_coeff(observations[0]['start_coords'][3:])

    agent_heading = quat_to_xy_heading(
        rotation_world_start
    )
    # This is rotation from -Z to -X. We want -Z to X for this particular sensor.
    agent_heading = -agent_heading

    return np.array(
        [-agent_position[2], agent_position[0], agent_heading], dtype=np.float32,
    )


def get_position_vector(env):
    origin = np.array(env.current_episode.start_position, dtype=np.float32)
    rotation_world_start = quaternion_from_coeff(env.current_episode.start_rotation)

    agent_position = quaternion_rotate_vector(
        rotation_world_start, origin
    )

    rotation_world_start = quaternion_from_coeff(env.current_episode.start_rotation)

    agent_heading = quat_to_xy_heading(
        rotation_world_start
    )
    # This is rotation from -Z to -X. We want -Z to X for this particular sensor.
    agent_heading = -agent_heading

    return np.array(
        [-agent_position[2], agent_position[0], agent_heading], dtype=np.float32,
    )


def quat_to_xy_heading(quat):
    direction_vector = np.array([0, 0, -1])
    heading_vector = quaternion_rotate_vector(quat, direction_vector)
    phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
    return np.array(phi)


def construct_envs(
        config: Config,
        env_class: Type[Union[Env, RLEnv]],
        workers_ignore_signals: bool = False,
        devices=None,
) -> VectorEnv:
    r"""Create VectorEnv object with specified config and env class type.
    To allow better performance, dataset are split into small ones for
    each individual env, grouped by scenes.

    :param config: configs that contain num_processes as well as information
    :param necessary to create individual environments.
    :param env_class: class type of the envs to be created.
    :param workers_ignore_signals: Passed to :ref:`habitat.VectorEnv`'s constructor
    :param devices: list of devices over which environments are distributed

    :return: VectorEnv object created according to specification.
    """

    num_processes = config.NUM_PROCESSES
    configs = []
    env_classes = [env_class for _ in range(num_processes)]
    dataset = make_dataset(config.TASK_CONFIG.DATASET.TYPE)
    scenes = config.TASK_CONFIG.DATASET.CONTENT_SCENES
    if "*" in config.TASK_CONFIG.DATASET.CONTENT_SCENES:
        scenes = dataset.get_scenes_to_load(config.TASK_CONFIG.DATASET)

    if num_processes > 1:
        if len(scenes) == 0:
            raise RuntimeError(
                "No scenes to load, multiple process logic relies on being able to split scenes uniquely between processes"
            )

        if 'estensi' in scenes[0]:
            scenes = [scenes[0]] * num_processes

        if len(scenes) < num_processes:
            raise RuntimeError(
                "reduce the number of processes as there "
                "aren't enough number of scenes"
            )

        random.shuffle(scenes)

    if devices is None:
        devices = [config.SIMULATOR_GPU_ID]

    world_size = config.WORLD_SIZE
    rank = config.RANK
    scenes_per_node = [[] for _ in range(world_size)]
    for idx, scene in enumerate(scenes):
        scenes_per_node[idx % len(scenes_per_node)].append(scene)
    scenes = scenes_per_node[rank]

    scene_splits = [[] for _ in range(num_processes)]
    for idx, scene in enumerate(scenes):
        scene_splits[idx % len(scene_splits)].append(scene)

    assert sum(map(len, scene_splits)) == len(scenes)

    for i in range(num_processes):
        proc_config = config.clone()
        proc_config.defrost()

        task_config = proc_config.TASK_CONFIG
        task_config.SEED = task_config.SEED + i
        if len(scenes) > 0:
            task_config.DATASET.CONTENT_SCENES = scene_splits[i]

        task_config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = devices[i % len(devices)]

        task_config.SIMULATOR.AGENT_0.SENSORS = config.SENSORS

        proc_config.freeze()
        configs.append(proc_config)

    envs = habitat.VectorEnv(
        make_env_fn=make_env_fn,
        env_fn_args=tuple(tuple(zip(configs, env_classes))),
        workers_ignore_signals=workers_ignore_signals,
    )
    return envs


def make_env_fn(
        config: Config, env_class: Type[Union[Env, RLEnv]]
) -> Union[Env, RLEnv]:
    r"""Creates an env of type env_class with specified config and rank.
    This is to be passed in as an argument when creating VectorEnv.

    Args:
        config: root exp config that has core env config node as well as
            env-specific config node.
        env_class: class type of the env to be created.

    Returns:
        env object created according to specification.
    """
    dataset = make_dataset(
        config.TASK_CONFIG.DATASET.TYPE, config=config.TASK_CONFIG.DATASET
    )
    env = env_class(config=config, dataset=dataset)
    env.seed(config.TASK_CONFIG.SEED)
    return env
