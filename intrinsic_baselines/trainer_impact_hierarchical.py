#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import glob
import gzip
import itertools
import json
import logging
import math
import os
import re
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional

import cv2
import gym.spaces as spaces
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from einops import rearrange, asnumpy
from gym.spaces import Box
from habitat import Config, logger
from habitat.core.spaces import EmptySpace, ActionSpace
from habitat_baselines.common.base_trainer import BaseRLTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.common.utils import (
    batch_obs,
    generate_video,
)
from habitat_baselines.rl.ppo import PPO
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from intrinsic_baselines.common.impact_models_hierarchical import ImpactModels
from intrinsic_baselines.common.rollout_storage_hierarchical import RolloutStorageHierarchical
from intrinsic_baselines.common.utils import append_observations, preprocessing_pixel_rgb
from habitat_extensions.utils import observations_to_image, topdown_to_image
from occant_baselines.common.env_utils import construct_envs
from occant_baselines.common.rollout_storage import (
    RolloutStorageExtended,
    MapLargeRolloutStorageMP,
)
from occant_baselines.models.mapnet import DepthProjectionNet
from occant_baselines.models.occant import OccupancyAnticipator
from occant_baselines.rl.ans import ActiveNeuralSLAMExplorer
from occant_baselines.rl.policy_utils import OccupancyAnticipationWrapper
from occant_baselines.supervised.imitation import Imitation
from occant_baselines.supervised.map_update import MapUpdate
from occant_utils.common import (
    add_pose,
    convert_world2map,
    convert_gt2channel_to_gtrgb,
)
from occant_utils.metrics import (
    measure_pose_estimation_performance,
    measure_area_seen_performance,
    measure_anticipation_reward,
    measure_map_quality,
    TemporalMetric,
    measure_diff_reward,
)
from occant_utils.visualization import generate_topdown_allocentric_map


class FalseLogger(logging.Logger):
    def __init__(
            self,
            name,
            level,
    ):
        super().__init__(name, level)

    def add_filehandler(self, log_filename):
        pass

    def info(self, msg, *args, **kwargs):
        pass


class FalseWriter:
    def __init__(self, *args: Any, **kwargs: Any):
        pass

    def add_scalar(self, *args: Any, **kwargs: Any):
        pass


@baseline_registry.register_trainer(name="ppo_impact_hierarchical")
class PPOTrainerImpactHierarchical(BaseRLTrainer):
    r"""Trainer class for Occupancy Anticipated based exploration algorithm.
    """
    supported_tasks = ["Exp-v0"]
    frozen_mapper_types = ["ans_depth", "occant_ground_truth"]

    def __init__(self, config=None):
        if config is not None:
            self._synchronize_configs(config)
        super().__init__(config)

        # Set pytorch random seed for initialization
        torch.manual_seed(config.PYT_RANDOM_SEED)

        self.mapper = None
        self.local_actor_critic = None
        self.global_actor_critic = None
        self.ans_net = None
        self.planner = None
        self.mapper_agent = None
        self.local_agent = None
        self.global_agent = None
        self.envs = None

        self.impact_obs = None
        self.impact_hidden_states = None
        self.episode_custom_rewards = None
        self.visitation_map = None
        self.density_loss_fn = None

        if self._is_master_process():
            self.logger = logger
        else:
            self.logger = FalseLogger(name="habitat", level=logging.INFO)

        if config is not None:
            self.logger.info(f"config: {config}")

    def _synchronize_configs(self, config):
        r"""Matches configs for different parts of the model as well as the simulator.
        """
        config.defrost()
        config.RL.ANS.PLANNER.nplanners = config.NUM_PROCESSES
        config.RL.ANS.MAPPER.thresh_explored = config.RL.ANS.thresh_explored
        config.RL.ANS.pyt_random_seed = config.PYT_RANDOM_SEED
        config.RL.ANS.OCCUPANCY_ANTICIPATOR.pyt_random_seed = config.PYT_RANDOM_SEED
        # Compute the EGO_PROJECTION options based on the
        # depth sensor information and agent parameters.
        map_size = config.RL.ANS.MAPPER.map_size
        map_scale = config.RL.ANS.MAPPER.map_scale
        min_depth = config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH
        max_depth = config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH
        hfov = config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.HFOV
        width = config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.WIDTH
        height = config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.HEIGHT
        hfov_rad = np.radians(float(hfov))
        vfov_rad = 2 * np.arctan((height / width) * np.tan(hfov_rad / 2.0))
        vfov = np.degrees(vfov_rad).item()
        camera_height = config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.POSITION[1]
        height_thresholds = [0.2, 1.5]
        # Set the EGO_PROJECTION options
        ego_proj_config = config.RL.ANS.OCCUPANCY_ANTICIPATOR.EGO_PROJECTION
        ego_proj_config.local_map_shape = (2, map_size, map_size)
        ego_proj_config.map_scale = map_scale
        ego_proj_config.min_depth = min_depth
        ego_proj_config.max_depth = max_depth
        ego_proj_config.hfov = hfov
        ego_proj_config.vfov = vfov
        ego_proj_config.camera_height = camera_height
        ego_proj_config.height_thresholds = height_thresholds
        config.RL.ANS.OCCUPANCY_ANTICIPATOR.EGO_PROJECTION = ego_proj_config
        # Set the GT anticipation options
        wall_fov = config.RL.ANS.OCCUPANCY_ANTICIPATOR.GP_ANTICIPATION.wall_fov
        config.TASK_CONFIG.TASK.GT_EGO_MAP_ANTICIPATED.WALL_FOV = wall_fov
        config.TASK_CONFIG.TASK.GT_EGO_MAP_ANTICIPATED.MAP_SIZE = map_size
        config.TASK_CONFIG.TASK.GT_EGO_MAP_ANTICIPATED.MAP_SCALE = map_scale
        config.TASK_CONFIG.TASK.GT_EGO_MAP_ANTICIPATED.MAX_SENSOR_RANGE = -1
        # Set the correct image scaling values
        config.RL.ANS.MAPPER.image_scale_hw = config.RL.ANS.image_scale_hw
        config.RL.ANS.LOCAL_POLICY.image_scale_hw = config.RL.ANS.image_scale_hw
        # Set the agent dynamics for the local policy
        config.RL.ANS.LOCAL_POLICY.AGENT_DYNAMICS.forward_step = (
            config.TASK_CONFIG.SIMULATOR.FORWARD_STEP_SIZE
        )
        config.RL.ANS.LOCAL_POLICY.AGENT_DYNAMICS.turn_angle = (
            config.TASK_CONFIG.SIMULATOR.TURN_ANGLE
        )
        # Enable global_maps measure if imitation learning is used for local policy
        if config.RL.ANS.LOCAL_POLICY.learning_algorithm == "il":
            if "GT_GLOBAL_MAP" not in config.TASK_CONFIG.TASK.MEASUREMENTS:
                config.TASK_CONFIG.TASK.MEASUREMENTS.append("GT_GLOBAL_MAP")
            config.TASK_CONFIG.TASK.GT_GLOBAL_MAP.MAP_SIZE = (
                config.RL.ANS.overall_map_size
            )
            config.TASK_CONFIG.TASK.GT_GLOBAL_MAP.MAP_SCALE = (
                config.RL.ANS.MAPPER.map_scale
            )
        config.freeze()

    def _setup_actor_critic_agent(self, ppo_cfg: Config, ans_cfg: Config) -> None:
        r"""Sets up actor critic and agent for PPO.

        Args:
            ppo_cfg: config node with relevant params
            ans_cfg: config node for ActiveNeuralSLAM model

        Returns:
            None
        """
        try:
            os.mkdir(self.config.TENSORBOARD_DIR)
        except:
            pass
        logger.add_filehandler(os.path.join(self.config.TENSORBOARD_DIR, "run.log"))

        occ_cfg = ans_cfg.OCCUPANCY_ANTICIPATOR
        mapper_cfg = ans_cfg.MAPPER
        # Create occupancy anticipation model
        occupancy_model = OccupancyAnticipator(occ_cfg)
        occupancy_model = OccupancyAnticipationWrapper(
            occupancy_model, mapper_cfg.map_size, (128, 128)
        )
        # Create ANS model
        self.ans_net = ActiveNeuralSLAMExplorer(ans_cfg, occupancy_model)
        if hasattr(ppo_cfg, 'DENSITY_MODEL'):
            density_cfg = ppo_cfg.DENSITY_MODEL
        else:
            density_cfg = None
        self.reward_models = ImpactModels(
            clip_param=ppo_cfg.clip_param,
            update_epochs=ans_cfg.CURIOSITY.update_epochs,
            num_mini_batch=ppo_cfg.num_mini_batch,
            value_loss_coef=ppo_cfg.value_loss_coef,
            entropy_coef=ppo_cfg.entropy_coef,
            lr=ppo_cfg.lr,
            eps=ppo_cfg.eps,
            max_grad_norm=ppo_cfg.max_grad_norm,
            use_normalized_advantage=ans_cfg.CURIOSITY.use_normalized_advantage,
            observation_spaces=self.envs.observation_spaces[0],
            visitation_count_type=ppo_cfg.visitation_count,
            density_model_cfg=density_cfg,
            impact_hidden_size=ans_cfg.CURIOSITY.curiosity_hidden_size,

            device=self.device,
            use_ddp=ans_cfg.use_ddp,
        )

        self.mapper = self.ans_net.mapper
        self.local_actor_critic = self.ans_net.local_policy
        self.global_actor_critic = self.ans_net.global_policy
        self.obs_encoder = self.reward_models.obs_encoder
        if self.reward_models.density_model is not None:
            self.density_model = self.reward_models.density_model
            self.density_model_optimizer = self.reward_models.density_model_optimizer
        # Create depth projection model to estimate visible occupancy
        self.depth_projection_net = DepthProjectionNet(ans_cfg.OCCUPANCY_ANTICIPATOR.EGO_PROJECTION)

        # Set to device
        self.mapper.to(self.device)
        self.local_actor_critic.to(self.device)
        self.global_actor_critic.to(self.device)
        self.obs_encoder.to(self.device)
        self.depth_projection_net.to(self.device)
        self.reward_models.to(self.device)

        self.ans_net.to_ddp()
        self.reward_models.to_ddp(device_ids=ans_cfg.gpu_ids, output_device=ans_cfg.gpu_ids[0])
        # ============================== Create agents ================================
        # Mapper agent
        self.mapper_agent = MapUpdate(
            self.mapper,
            lr=mapper_cfg.lr,
            eps=mapper_cfg.eps,
            label_id=mapper_cfg.label_id,
            max_grad_norm=mapper_cfg.max_grad_norm,
            pose_loss_coef=mapper_cfg.pose_loss_coef,
            occupancy_anticipator_type=ans_cfg.OCCUPANCY_ANTICIPATOR.type,
            freeze_projection_unit=mapper_cfg.freeze_projection_unit,
            num_update_batches=mapper_cfg.num_update_batches,
            batch_size=mapper_cfg.map_batch_size,
            mapper_rollouts=self.mapper_rollouts,
        )
        # Local policy
        if ans_cfg.LOCAL_POLICY.use_heuristic_policy:
            self.local_agent = None
        elif ans_cfg.LOCAL_POLICY.learning_algorithm == "rl":
            self.local_agent = PPO(
                actor_critic=self.local_actor_critic,
                clip_param=ppo_cfg.clip_param,
                ppo_epoch=ppo_cfg.ppo_epoch,
                num_mini_batch=ppo_cfg.num_mini_batch,
                value_loss_coef=ppo_cfg.value_loss_coef,
                entropy_coef=ppo_cfg.local_entropy_coef,
                lr=ppo_cfg.local_policy_lr,
                eps=ppo_cfg.eps,
                max_grad_norm=ppo_cfg.max_grad_norm,
            )
        else:
            self.local_agent = Imitation(
                actor_critic=self.local_actor_critic,
                lr=ppo_cfg.local_policy_lr,
                eps=ppo_cfg.eps,
                max_grad_norm=ppo_cfg.max_grad_norm,
            )
        # Global policy
        self.global_agent = PPO(
            actor_critic=self.global_actor_critic,
            clip_param=ppo_cfg.clip_param,
            ppo_epoch=ppo_cfg.ppo_epoch,
            num_mini_batch=ppo_cfg.num_mini_batch,
            value_loss_coef=ppo_cfg.value_loss_coef,
            entropy_coef=ppo_cfg.entropy_coef,
            lr=ppo_cfg.lr,
            eps=ppo_cfg.eps,
            max_grad_norm=ppo_cfg.max_grad_norm,
        )
        if ans_cfg.model_path != "":
            self.resume_checkpoint(ans_cfg.model_path)

        self.episode_custom_rewards = torch.zeros(self.envs.num_envs, 1)

    def save_checkpoint(
            self, file_name: str, extra_state: Optional[Dict] = None
    ) -> None:
        r"""Save checkpoint with specified name.

        Args:
            file_name: file name for checkpoint

        Returns:
            None
        """
        checkpoint = {
            "mapper_state_dict": self.mapper_agent.state_dict(),
            "local_state_dict": self.local_agent.state_dict(),
            "global_state_dict": self.global_agent.state_dict(),
            "obs_encoder": self.obs_encoder.state_dict(),
            "config": self.config,
        }
        if hasattr(self, 'density_model'):
            checkpoint["density_model"] = self.density_model.state_dict()
        if extra_state is not None:
            checkpoint["extra_state"] = extra_state

        torch.save(checkpoint, os.path.join(self.config.CHECKPOINT_FOLDER, file_name))

    def load_checkpoint(self, checkpoint_path: str, *args, **kwargs) -> Dict:
        r"""Load checkpoint of specified path as a dict.

        Args:
            checkpoint_path: path of target checkpoint
            *args: additional positional args
            **kwargs: additional keyword args

        Returns:
            dict containing checkpoint info
        """
        return torch.load(checkpoint_path, *args, **kwargs)

    def resume_checkpoint(self, path=None):
        r"""If an existing checkpoint already exists, resume training.
        """
        checkpoints = glob.glob(f"{self.config.CHECKPOINT_FOLDER}/*.pth")
        if path is None:
            if len(checkpoints) == 0:
                num_updates_start = 0
                count_steps = 0
                count_checkpoints = 0
            else:
                # Load lastest checkpoint
                last_ckpt = sorted(checkpoints, key=lambda x: int(x.split(".")[1]))[-1]
                checkpoint_path = last_ckpt

                # Restore checkpoints to models
                ckpt_dict = self.load_checkpoint(checkpoint_path)

                self.mapper_agent.load_state_dict(ckpt_dict["mapper_state_dict"])
                self.local_agent.load_state_dict(ckpt_dict["local_state_dict"])
                self.global_agent.load_state_dict(ckpt_dict["global_state_dict"])
                self.obs_encoder.load_state_dict(ckpt_dict["obs_encoder"])
                if 'density_model' in ckpt_dict.keys():
                    self.density_model.load_state_dict(ckpt_dict["density_model"])

                self.mapper = self.mapper_agent.mapper
                self.local_actor_critic = self.local_agent.actor_critic
                self.global_actor_critic = self.global_agent.actor_critic

                # Set the logging counts
                ckpt_id = int(last_ckpt.split("/")[-1].split(".")[1])
                num_updates_start = ckpt_dict["extra_state"]["update"] + 1
                count_steps = ckpt_dict["extra_state"]["step"]
                count_checkpoints = ckpt_id + 1
                print(f"Resuming checkpoint {last_ckpt} at {count_steps} frames")
        else:
            print(f"Loading pretrained model!")
            # Restore checkpoints to models
            ckpt_dict = self.load_checkpoint(path)
            self.mapper_agent.load_state_dict(ckpt_dict["mapper_state_dict"])
            self.local_agent.load_state_dict(ckpt_dict["local_state_dict"])
            self.global_agent.load_state_dict(ckpt_dict["global_state_dict"])
            self.mapper = self.mapper_agent.mapper
            self.local_actor_critic = self.local_agent.actor_critic
            self.global_actor_critic = self.global_agent.actor_critic
            num_updates_start = 0
            count_steps = 0
            count_checkpoints = 0

        return num_updates_start, count_steps, count_checkpoints

    def _create_mapper_rollout_inputs(
            self, prev_batch, batch,
    ):
        ans_cfg = self.config.RL.ANS
        mapper_rollout_inputs = {
            "rgb_at_t_1": prev_batch["rgb"],
            "depth_at_t_1": prev_batch["depth"],
            "ego_map_gt_at_t_1": prev_batch["ego_map_gt"],
            "pose_at_t_1": prev_batch["pose"],
            "pose_gt_at_t_1": prev_batch["pose_gt"],
            "rgb_at_t": batch["rgb"],
            "depth_at_t": batch["depth"],
            "ego_map_gt_at_t": batch["ego_map_gt"],
            "pose_at_t": batch["pose"],
            "pose_gt_at_t": batch["pose_gt"],
            "ego_map_gt_anticipated_at_t": batch["ego_map_gt_anticipated"],
            "action_at_t_1": batch["prev_actions"],
        }
        if ans_cfg.OCCUPANCY_ANTICIPATOR.type == "baseline_gt_anticipation":
            mapper_rollout_inputs["ego_map_gt_anticipated_at_t_1"] = prev_batch[
                "ego_map_gt_anticipated"
            ]

        return mapper_rollout_inputs

    def _convert_actions_to_delta(self, actions):
        """actions -> torch Tensor
        """
        sim_cfg = self.config.TASK_CONFIG.SIMULATOR
        delta_xyt = torch.zeros(self.envs.num_envs, 3, device=self.device)
        # Forward step
        act_mask = actions.squeeze(1) == 0
        delta_xyt[act_mask, 0] = sim_cfg.FORWARD_STEP_SIZE
        # Turn left
        act_mask = actions.squeeze(1) == 1
        delta_xyt[act_mask, 2] = math.radians(-sim_cfg.TURN_ANGLE)
        # Turn right
        act_mask = actions.squeeze(1) == 2
        delta_xyt[act_mask, 2] = math.radians(sim_cfg.TURN_ANGLE)
        return delta_xyt

    def _compute_mapbased_global_metric(self, ground_truth_states, mapper_outputs, state_estimates=None):
        """Estimates global reward metric for the current states.
        """
        if self.config.RL.ANS.reward_type == "area_seen":
            global_reward_metric = measure_area_seen_performance(
                ground_truth_states["visible_occupancy"], reduction="none"
            )["area_seen"]
        else:
            global_reward_metric = measure_anticipation_reward(
                mapper_outputs["mt"], ground_truth_states["environment_layout"],
                reduction="none",
                apply_mask="True",
            )
        if self.config.RL.ANS.diff_reward:
            diff_reward = measure_diff_reward(
                gt_map_states=ground_truth_states["gt_map_states"], state_estimates=state_estimates, reduction="none"
            )
            global_reward_metric = global_reward_metric + diff_reward * self.config.RL.ANS.diff_reward_coeff

        return global_reward_metric.unsqueeze(1)

    def _compute_pixel_prediction_gain(self, recoding_log_probs, log_probs, target):
        indexes = target.unsqueeze(1).type(torch.int64)
        log_prob = log_probs.gather(1, indexes)
        recoding_log_prob = recoding_log_probs.gather(1, indexes)
        prediction_gain = (recoding_log_prob - log_prob).sum(-1).sum(-1).sum(-1).sum(-1)
        return prediction_gain

    def _compute_pseudo_counts(self, step_number, c, prediction_gains):
        prediction_gains = torch.max(prediction_gains, torch.zeros_like(prediction_gains))
        pseudo_counts = (torch.exp((c * (step_number ** (-0.5))) * prediction_gains.to(self.device)) - 1) ** (-1)
        return pseudo_counts

    def _compute_density_impact_rewards(self, features, new_features, pseudo_counts):
        denominator = torch.sqrt(torch.max(pseudo_counts, torch.ones_like(pseudo_counts)))
        impact_rewards = ((torch.mean(torch.sqrt(F.mse_loss(features, new_features, reduction='none')),
                                      dim=1)) / denominator).view(-1, 1)
        return impact_rewards

    def _collect_rollout_step(
            self,
            batch,
            prev_batch,
            episode_step_count,
            state_estimates,
            ground_truth_states,
            masks,
            new_masks,
            mapper_rollouts,
            local_rollouts,
            global_rollouts,
            current_local_episode_reward,
            current_global_episode_reward,
    ):
        pth_time = 0.0
        env_time = 0.0

        device = self.device
        ppo_cfg = self.config.RL.PPO
        ans_cfg = self.config.RL.ANS

        NUM_LOCAL_STEPS = ppo_cfg.num_local_steps

        self.ans_net.eval()

        for t in range(NUM_LOCAL_STEPS):

            # ---------------------------- sample actions -----------------------------
            t_sample_action = time.time()

            with torch.no_grad():
                (
                    mapper_inputs,
                    local_policy_inputs,
                    global_policy_inputs,
                    mapper_outputs,
                    local_policy_outputs,
                    global_policy_outputs,
                    state_estimates,
                    intrinsic_rewards,
                ) = self.ans_net.act(
                    batch,
                    prev_batch,
                    state_estimates,
                    episode_step_count,
                    masks.to(self.device),
                    deterministic=ans_cfg.LOCAL_POLICY.deterministic_flag,
                )

            pth_time += time.time() - t_sample_action

            # -------------------- update global rollout stats ------------------------
            t_update_stats = time.time()

            if t == 0:
                # Sanity check
                assert global_policy_inputs is not None

                if hasattr(self.config.RL.PPO,
                           'global_reward_type') and 'impact' == self.config.RL.PPO.global_reward_type:
                    global_reward_metric = self.episode_custom_rewards
                else:
                    global_reward_metric = self._compute_mapbased_global_metric(
                        ground_truth_states, mapper_outputs, state_estimates,
                    )

                # Update reward for previous global_policy action
                if global_rollouts.step == 0:
                    global_rewards = torch.zeros(self.envs.num_envs, 1)
                else:
                    global_rewards = (
                            global_reward_metric
                            - ground_truth_states["prev_global_reward_metric"]
                    ).cpu()
                ground_truth_states["prev_global_reward_metric"].copy_(
                    global_reward_metric
                )
                global_rollouts.rewards[global_rollouts.step - 1].copy_(
                    global_rewards * ppo_cfg.global_reward_scale
                )

                global_rollouts.insert(
                    global_policy_inputs,
                    None,
                    global_policy_outputs["actions"],
                    global_policy_outputs["action_log_probs"],
                    global_policy_outputs["values"],
                    torch.zeros_like(global_rewards),
                    masks.to(device),
                )
                current_global_episode_reward += global_rewards

            pth_time += time.time() - t_update_stats

            # --------------------- update mapper rollout stats -----------------------
            t_update_stats = time.time()

            mapper_rollout_inputs = self._create_mapper_rollout_inputs(
                prev_batch, batch
            )
            mapper_rollouts.insert(mapper_rollout_inputs)

            pth_time += time.time() - t_update_stats

            # ------------------ update local_policy rollout stats --------------------
            t_update_stats = time.time()

            # Assign local rewards to previous local action
            local_rewards = (
                    intrinsic_rewards["local_rewards"]
                    + batch["collision_sensor"] * ans_cfg.local_collision_reward
            ).cpu()
            current_local_episode_reward += local_rewards
            # The intrinsic rewards correspond to the previous action, not
            # the one executed currently.
            if local_rollouts.step > 0:
                local_rollouts.rewards[local_rollouts.step - 1].copy_(
                    local_rewards * ppo_cfg.local_reward_scale
                )
            # Update local_rollouts
            if ans_cfg.LOCAL_POLICY.learning_algorithm == "rl":
                local_policy_actions = local_policy_outputs["actions"]
            else:
                local_policy_actions = local_policy_outputs["gt_actions"]

            local_rollouts.insert(
                local_policy_inputs,
                state_estimates["recurrent_hidden_states"],
                local_policy_actions,
                local_policy_outputs["action_log_probs"],
                local_policy_outputs["values"],
                torch.zeros_like(local_rewards),
                local_policy_outputs["local_masks"].to(device),
            )

            pth_time += time.time() - t_update_stats

            # ---------------------- execute environment action -----------------------
            t_step_env = time.time()

            actions = local_policy_outputs["actions"]
            outputs = self.envs.step([a[0].item() for a in actions])
            observations, _, dones, infos = [list(x) for x in zip(*outputs)]

            env_time += time.time() - t_step_env

            # ---------------------- compute impact reward -----------------------

            new_masks.copy_(
                torch.tensor(
                    [[0.0] if done else [1.0] for done in dones], dtype=torch.float
                )
            )

            if hasattr(self.config.RL.PPO,
                       'global_reward_type'):
                new_imp_obs = append_observations(self.config, self.impact_obs, observations,
                                                  self.device)
                # Update reward for previous global_policy action
                self.obs_encoder.eval()

                with torch.no_grad():
                    features, self.impact_hidden_states = self.obs_encoder(
                        self._dict_to_device(self.impact_obs, device),
                        self.impact_hidden_states,
                        None, masks.to(device),
                        use_rnn=ans_cfg.CURIOSITY.use_curiosity_rnn)
                    new_features, _ = self.obs_encoder(self._dict_to_device(new_imp_obs, device),
                                                       self.impact_hidden_states,
                                                       None, new_masks.to(device),
                                                       use_rnn=ans_cfg.CURIOSITY.use_curiosity_rnn)

                self.obs_encoder.train()

                if self.impact_hidden_states is None:
                    self.impact_hidden_states = torch.zeros(ppo_cfg.num_local_steps + 1,
                                                            self.obs_encoder.num_recurrent_layers,
                                                            self.envs.num_envs,
                                                            ans_cfg.CURIOSITY.curiosity_hidden_size, )

                if 'imp' in self.config.RL.PPO.global_reward_type:
                    b, f, h, w = mapper_outputs['mt'].shape
                    key = 'pose'
                    new_poses = torch.tensor([obs[key] for obs in observations]).to(batch['pose'].device)

                    # sanity check
                    assert h == w and (h == 2001 or h == 961)

                    xy_abs = convert_world2map(new_poses, (h, w), 0.05)
                    if self.config.RL.PPO.visitation_count == 'impact_grid':
                        self.visitation_map[
                            list(range(xy_abs.shape[0])), list(xy_abs[:, 0] // ans_cfg.visitation_count_divider),
                            list(xy_abs[:, 1] // ans_cfg.visitation_count_divider)] = self.visitation_map[list(
                            range(xy_abs.shape[0])), list(
                            xy_abs[:, 0] // ans_cfg.visitation_count_divider), list(
                            xy_abs[:, 1] // ans_cfg.visitation_count_divider)] + 1
                        impact_rewards = self._compute_grid_impact_reward(features, new_features, self.visitation_map[
                            list(range(xy_abs.shape[0])), list(xy_abs[:, 0] // ans_cfg.visitation_count_divider),
                            list(xy_abs[:, 1] // ans_cfg.visitation_count_divider)].to(self.device))
                    elif self.config.RL.PPO.visitation_count == 'pseudo_grid':
                        self.visitation_map[
                            list(range(xy_abs.shape[0])), list(xy_abs[:, 0] // ans_cfg.visitation_count_divider),
                            list(xy_abs[:, 1] // ans_cfg.visitation_count_divider)] = self.visitation_map[list(
                            range(xy_abs.shape[0])), list(
                            xy_abs[:, 0] // ans_cfg.visitation_count_divider), list(
                            xy_abs[:, 1] // ans_cfg.visitation_count_divider)] + 1
                        impact_rewards = self._compute_grid_pseudo_reward(self.visitation_map[
                            list(range(xy_abs.shape[0])), list(
                                xy_abs[:,
                                0] // ans_cfg.visitation_count_divider),
                            list(xy_abs[:,
                                 1] // ans_cfg.visitation_count_divider)].to(
                            self.device))
                    elif self.config.RL.PPO.visitation_count == 'impact_pixel':
                        density_obs = preprocessing_pixel_rgb(self.config, observations).to(self.device)
                        self.density_model.train()
                        unnormalized_probs = self.density_model(density_obs)
                        log_probs = torch.log_softmax(unnormalized_probs, dim=1)

                        self.density_model_optimizer.zero_grad()
                        target_image = torch.round(density_obs * self.config.RL.PPO.DENSITY_MODEL.num_buckets)
                        density_loss = torch.nn.functional.cross_entropy(unnormalized_probs, target_image.long())
                        density_loss.backward()
                        self.density_model_optimizer.step()
                        self.density_model.eval()
                        with torch.no_grad():
                            recoding_unnormalized_probs = self.density_model(density_obs)
                            recoding_log_probs = torch.log_softmax(recoding_unnormalized_probs, dim=1)

                            prediction_gains = self._compute_pixel_prediction_gain(recoding_log_probs, log_probs,
                                                                                   target_image)
                            pseudo_counts = self._compute_pseudo_counts(episode_step_count.squeeze(-1) + 1,
                                                                        self.config.RL.PPO.DENSITY_MODEL.c_decay,
                                                                        prediction_gains)
                            impact_rewards = self._compute_density_impact_rewards(features, new_features, pseudo_counts)

                    elif self.config.RL.PPO.visitation_count == 'pseudo':
                        density_obs = preprocessing_pixel_rgb(self.config, observations).to(self.device)
                        self.density_model.train()
                        unnormalized_probs = self.density_model(density_obs)
                        log_probs = torch.log_softmax(unnormalized_probs, dim=1)
                        self.density_model_optimizer.zero_grad()
                        target_image = torch.round(density_obs * self.config.RL.PPO.DENSITY_MODEL.num_buckets)
                        density_loss = torch.nn.functional.cross_entropy(unnormalized_probs, target_image.long())
                        density_loss.backward()
                        self.density_model_optimizer.step()
                        self.density_model.eval()
                        with torch.no_grad():
                            recoding_unnormalized_probs = self.density_model(density_obs)
                            recoding_log_probs = torch.log_softmax(recoding_unnormalized_probs, dim=1)

                            prediction_gains = self._compute_pixel_prediction_gain(recoding_log_probs, log_probs,
                                                                                   target_image)
                            pseudo_counts = self._compute_pseudo_counts(episode_step_count.squeeze(-1) + 1,
                                                                        self.config.RL.PPO.DENSITY_MODEL.c_decay,
                                                                        prediction_gains)
                            denominator = torch.sqrt(torch.max(pseudo_counts, torch.ones_like(pseudo_counts)))
                            impact_rewards = (torch.ones_like(pseudo_counts) / denominator).view(-1, 1)
                    else:
                        raise ValueError

                    self.episode_custom_rewards = self.episode_custom_rewards.to(
                        device) + impact_rewards.clone().detach()

                self.impact_obs = new_imp_obs
                del new_imp_obs

            # -------------------- update ground-truth states -------------------------
            t_update_stats = time.time()

            masks.copy_(new_masks)

            # Sanity check
            assert episode_step_count[0].item() <= self.config.T_EXP - 1
            assert not dones[0], "DONE must not be called during training"

            del prev_batch
            prev_batch = batch
            if 'scene_name' in observations[0].keys():
                for i, obs in enumerate(observations):
                    del observations[i]['scene_name']
            batch = self._prepare_batch(
                observations, prev_batch=prev_batch, device=device, actions=actions
            )

            # Update visible occupancy
            ground_truth_states["visible_occupancy"] = self.mapper.ext_register_map(
                ground_truth_states["visible_occupancy"],
                rearrange(batch["ego_map_gt"], "b h w c -> b c h w"),
                batch["pose_gt"],
            )
            ground_truth_states["pose"].copy_(batch["pose_gt"])
            # Update ground_truth world layout that is provided only at episode start
            # to avoid data transfer bottlenecks
            if episode_step_count[0].item() == 0 and "gt_global_map" in infos[0].keys():
                environment_layout = np.stack(
                    [info["gt_global_map"] for info in infos], axis=0
                )
                environment_layout = rearrange(environment_layout, "b h w c -> b c h w")
                environment_layout = torch.Tensor(environment_layout)
                if (
                        ans_cfg.reward_type == "map_accuracy"
                        or ans_cfg.LOCAL_POLICY.learning_algorithm == "il"
                ):
                    ground_truth_states["environment_layout"].copy_(environment_layout)

            # The ground_truth world layout is used to generate ground_truth action
            # labels for local policy during imitation.
            if ans_cfg.LOCAL_POLICY.learning_algorithm == "il":
                batch["gt_global_map"] = ground_truth_states["environment_layout"]

            pth_time += time.time() - t_update_stats

            episode_step_count += 1

        return (
            pth_time,
            env_time,
            self.envs.num_envs * NUM_LOCAL_STEPS,
            prev_batch,
            batch,
            state_estimates,
            ground_truth_states,
        )

    def _supplementary_rollout_update(
            self,
            batch,
            prev_batch,
            episode_step_count,
            state_estimates,
            ground_truth_states,
            masks,
            local_rollouts,
            global_rollouts,
            update_option="local",
    ):
        """
        Since the inputs for local, global policies are obtained only after
        a forward pass, it will not be possible to update the rollouts immediately
        after self.envs.step() . This causes a delay of 1 step in the rollout
        updates for local, global policies. To account for this, perform this
        supplementary rollout update just before updating the corresponding policy.
        """
        pth_time = 0.0
        ppo_cfg = self.config.RL.PPO
        ans_cfg = self.config.RL.ANS

        t_sample_action = time.time()

        # Copy states before sampling actions
        ans_states_copy = self.ans_net.get_states()

        self.ans_net.eval()

        with torch.no_grad():
            (
                mapper_inputs,
                local_policy_inputs,
                global_policy_inputs,
                mapper_outputs,
                local_policy_outputs,
                _,
                _,
                intrinsic_rewards,
            ) = self.ans_net.act(
                batch,
                prev_batch,
                state_estimates,
                episode_step_count,
                masks,
                deterministic=ans_cfg.LOCAL_POLICY.deterministic_flag,
            )

        self.ans_net.train()

        # Restore states
        self.ans_net.update_states(ans_states_copy)

        pth_time += time.time() - t_sample_action

        t_update_stats = time.time()

        # Update local_rollouts
        if update_option == "local":
            for k, v in local_policy_inputs.items():
                local_rollouts.observations[k][local_rollouts.step].copy_(v)
            local_rewards = intrinsic_rewards["local_rewards"].cpu()
            local_rollouts.rewards[local_rollouts.step - 1].copy_(
                local_rewards * ppo_cfg.local_reward_scale
            )
            local_masks = local_policy_outputs["local_masks"]
            local_rollouts.masks[local_rollouts.step].copy_(local_masks)

        # Update global_rollouts if available
        if update_option == "global":
            # Sanity check
            assert episode_step_count[0].item() % ans_cfg.goal_interval == 0
            assert global_policy_inputs is not None
            for k, v in global_policy_inputs.items():
                global_rollouts.observations[k][global_rollouts.step].copy_(v)
            if hasattr(self.config.RL.PPO,
                       'global_reward_type') and self.config.RL.PPO.global_reward_type == 'impact':
                global_reward_metric = self.episode_custom_rewards
            else:
                global_reward_metric = self._compute_mapbased_global_metric(
                    ground_truth_states, mapper_outputs, state_estimates
                )
            global_rewards = (
                    global_reward_metric - ground_truth_states["prev_global_reward_metric"]
            ).cpu()
            global_rollouts.rewards[global_rollouts.step - 1].copy_(
                global_rewards * self.config.RL.PPO.global_reward_scale
            )
            global_rollouts.masks[global_rollouts.step].copy_(masks)

        pth_time += time.time() - t_update_stats

        return pth_time

    def _update_mapper_agent(self, mapper_rollouts):
        t_update_model = time.time()

        losses = self.mapper_agent.update(mapper_rollouts)

        return time.time() - t_update_model, losses

    def _update_local_agent(self, local_rollouts):
        t_update_model = time.time()

        ppo_cfg = self.config.RL.PPO

        with torch.no_grad():
            last_observation = {
                k: v[-1] for k, v in local_rollouts.observations.items()
            }
            next_local_value = self.local_actor_critic.get_value(
                last_observation,
                local_rollouts.recurrent_hidden_states[-1],
                local_rollouts.prev_actions[-1],
                local_rollouts.masks[-1],
            ).detach()

        local_rollouts.compute_returns(
            next_local_value, ppo_cfg.use_gae, ppo_cfg.gamma, ppo_cfg.tau
        )

        (
            local_value_loss,
            local_action_loss,
            local_dist_entropy,
        ) = self.local_agent.update(local_rollouts)

        update_metrics = {
            "value_loss": local_value_loss,
            "action_loss": local_action_loss,
            "dist_entropy": local_dist_entropy,
        }

        local_rollouts.after_update()

        return time.time() - t_update_model, update_metrics

    def _update_global_agent(self, global_rollouts):
        t_update_model = time.time()

        ppo_cfg = self.config.RL.PPO

        with torch.no_grad():
            last_observation = {
                k: v[-1].to(self.device)
                for k, v in global_rollouts.observations.items()
            }
            next_global_value = self.global_actor_critic.get_value(
                last_observation,
                None,
                global_rollouts.prev_actions[-1].to(self.device),
                global_rollouts.masks[-1].to(self.device),
            ).detach()

        global_rollouts.compute_returns(
            next_global_value, ppo_cfg.use_gae, ppo_cfg.gamma, ppo_cfg.tau
        )

        (
            global_value_loss,
            global_action_loss,
            global_dist_entropy,
        ) = self.global_agent.update(global_rollouts)

        update_metrics = {
            "value_loss": global_value_loss,
            "action_loss": global_action_loss,
            "dist_entropy": global_dist_entropy,
        }

        global_rollouts.after_update()

        return time.time() - t_update_model, update_metrics

    def _assign_devices(self):
        # Assign devices for the simulator
        if len(self.config.SIMULATOR_GPU_IDS) > 0:
            devices = self.config.SIMULATOR_GPU_IDS
        elif "CUDA_VISIBLE_DEVICES" in os.environ:
            visible_devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
            devices = [int(dev) for dev in visible_devices]
            # Devices need to be indexed between 0 to N-1
            devices = [dev for dev in range(len(devices))]
            if len(devices) > 1:
                devices = devices[1:]
        else:
            devices = None
        return devices

    def _create_mapper_rollouts(self, ppo_cfg, ans_cfg):
        V = ans_cfg.MAPPER.map_size
        imH, imW = ans_cfg.image_scale_hw
        mapper_observation_space = {
            "rgb_at_t_1": spaces.Box(
                low=0.0, high=255.0, shape=(imH, imW, 3), dtype=np.float32
            ),
            "depth_at_t_1": spaces.Box(
                low=0.0, high=255.0, shape=(imH, imW, 1), dtype=np.float32
            ),
            "ego_map_gt_at_t_1": spaces.Box(
                low=0.0, high=1.0, shape=(V, V, 2), dtype=np.float32
            ),
            "pose_at_t_1": spaces.Box(
                low=-100000.0, high=100000.0, shape=(3,), dtype=np.float32
            ),
            "pose_gt_at_t_1": spaces.Box(
                low=-100000.0, high=100000.0, shape=(3,), dtype=np.float32
            ),
            "pose_hat_at_t_1": spaces.Box(
                low=-100000.0, high=100000.0, shape=(3,), dtype=np.float32
            ),
            "rgb_at_t": spaces.Box(
                low=0.0, high=255.0, shape=(imH, imW, 3), dtype=np.float32
            ),
            "depth_at_t": spaces.Box(
                low=0.0, high=255.0, shape=(imH, imW, 1), dtype=np.float32
            ),
            "ego_map_gt_at_t": spaces.Box(
                low=0.0, high=1.0, shape=(V, V, 2), dtype=np.float32
            ),
            "pose_at_t": spaces.Box(
                low=-100000.0, high=100000.0, shape=(3,), dtype=np.float32
            ),
            "pose_gt_at_t": spaces.Box(
                low=-100000.0, high=100000.0, shape=(3,), dtype=np.float32
            ),
            "ego_map_gt_anticipated_at_t": self.envs.observation_spaces[0].spaces[
                "ego_map_gt_anticipated"
            ],
            "action_at_t_1": spaces.Box(low=0, high=4, shape=(1,), dtype=np.int32),
        }
        if ans_cfg.OCCUPANCY_ANTICIPATOR.type == "baseline_gt_anticipation":
            mapper_observation_space[
                "ego_map_gt_anticipated_at_t_1"
            ] = self.envs.observation_spaces[0].spaces["ego_map_gt_anticipated"]
        mapper_observation_space = spaces.Dict(mapper_observation_space)
        # Multiprocessing manager
        mapper_manager = mp.Manager()
        mapper_device = self.device
        if ans_cfg.MAPPER.use_data_parallel and len(ans_cfg.MAPPER.gpu_ids) > 0:
            mapper_device = ans_cfg.MAPPER.gpu_ids[0]
        mapper_rollouts = MapLargeRolloutStorageMP(
            ans_cfg.MAPPER.replay_size,
            mapper_observation_space,
            mapper_device,
            mapper_manager,
        )

        return mapper_rollouts

    def _create_global_rollouts(self, ppo_cfg, ans_cfg):
        M = ans_cfg.overall_map_size
        G = ans_cfg.GLOBAL_POLICY.map_size
        global_observation_space = spaces.Dict(
            {
                "pose_in_map_at_t": spaces.Box(
                    low=-100000.0, high=100000.0, shape=(2,), dtype=np.float32
                ),
                "map_at_t": spaces.Box(
                    low=0.0, high=1.0, shape=(4, M, M), dtype=np.float32
                ),
            }
        )
        global_action_space = ActionSpace(
            {
                f"({x[0]}, {x[1]})": EmptySpace()
                for x in itertools.product(range(G), range(G))
            }
        )
        global_rollouts = RolloutStorageExtended(
            ppo_cfg.num_global_steps,
            self.envs.num_envs,
            global_observation_space,
            global_action_space,
            1,
            enable_recurrence=False,
            delay_observations_entry=True,
            delay_masks_entry=True,
            enable_memory_efficient_mode=True,
        )
        return global_rollouts

    def _create_local_rollouts(self, ppo_cfg, ans_cfg):
        imH, imW = ans_cfg.image_scale_hw
        local_action_space = ActionSpace(
            {
                "move_forward": EmptySpace(),
                "turn_left": EmptySpace(),
                "turn_right": EmptySpace(),
            }
        )
        local_observation_space = {
            "rgb_at_t": spaces.Box(
                low=0.0, high=255.0, shape=(imH, imW, 3), dtype=np.float32
            ),
            "goal_at_t": spaces.Box(
                low=-100000.0, high=100000.0, shape=(2,), dtype=np.float32
            ),
            "t": spaces.Box(low=0.0, high=100000.0, shape=(1,), dtype=np.float32),
        }

        local_observation_space = spaces.Dict(local_observation_space)
        local_rollouts = RolloutStorageExtended(
            ppo_cfg.num_local_steps,
            self.envs.num_envs,
            local_observation_space,
            local_action_space,
            ans_cfg.LOCAL_POLICY.hidden_size,
            enable_recurrence=True,
            delay_observations_entry=True,
            delay_masks_entry=True,
        )
        return local_rollouts

    def _create_impact_obs(self, ppo_cfg, ans_cfg):
        rollout_obs_spaces = copy.deepcopy(self.envs.observation_spaces[0])
        rollout_obs_spaces.spaces = {}
        resized_dim_x = ans_cfg.obs_resized_dim
        resized_dim_y = ans_cfg.obs_resized_dim
        channels = ans_cfg.CURIOSITY.num_concat_obs

        if 'RGB_SENSOR' in self.config['SENSORS']:
            rollout_obs_spaces.spaces['rgb'] = Box(low=0, high=255,
                                                   shape=(
                                                       resized_dim_y,
                                                       resized_dim_x,
                                                       channels,
                                                   ),
                                                   dtype='uint8')
        if 'DEPTH_SENSOR' in self.config['SENSORS']:
            rollout_obs_spaces.spaces['depth'] = Box(low=0.0, high=1.0,
                                                     shape=(
                                                         resized_dim_y,
                                                         resized_dim_x,
                                                         channels,
                                                     ),
                                                     dtype='float32')
        imp_rollouts = RolloutStorageHierarchical(
            ppo_cfg.num_local_steps,
            self.envs.num_envs,
            rollout_obs_spaces,
            self.envs.action_spaces[0],
            ans_cfg.CURIOSITY.curiosity_hidden_size,
            num_recurrent_layers=self.obs_encoder.num_recurrent_layers
        )

        imp_obs = dict()
        if 'rgb' in imp_rollouts.observations.keys():
            imp_obs['rgb'] = torch.zeros_like(imp_rollouts.observations['rgb'][0])
        if 'depth' in imp_rollouts.observations.keys():
            imp_obs['depth'] = torch.zeros_like(imp_rollouts.observations['depth'][0])

        return imp_obs

    def _prepare_batch(self, observations, prev_batch=None, device=None, actions=None):
        imH, imW = self.config.RL.ANS.image_scale_hw
        device = self.device if device is None else device
        batch = batch_obs(observations, device=device)
        if batch["rgb"].size(1) != imH or batch["rgb"].size(2) != imW:
            rgb = rearrange(batch["rgb"], "b h w c -> b c h w")
            rgb = F.interpolate(rgb, (imH, imW), mode="bilinear")
            batch["rgb"] = rearrange(rgb, "b c h w -> b h w c")
        if batch["depth"].size(1) != imH or batch["depth"].size(2) != imW:
            depth = rearrange(batch["depth"], "b h w c -> b c h w")
            depth = F.interpolate(depth, (imH, imW), mode="nearest")
            batch["depth"] = rearrange(depth, "b c h w -> b h w c")
        # Compute ego_map_gt from depth
        ego_map_gt_b = self.depth_projection_net(
            rearrange(batch["depth"], "b h w c -> b c h w")
        )
        batch["ego_map_gt"] = rearrange(ego_map_gt_b, "b c h w -> b h w c")
        if actions is None:
            # Initialization condition
            # If pose estimates are not available, set the initial estimate to zeros.
            if "pose" not in batch:
                # Set initial pose estimate to zero
                batch["pose"] = torch.zeros(self.envs.num_envs, 3).to(self.device)
            batch["prev_actions"] = torch.zeros(self.envs.num_envs, 1).to(self.device)
        else:
            # Rollouts condition
            # If pose estimates are not available, compute them from action taken.
            if "pose" not in batch:
                assert prev_batch is not None
                actions_delta = self._convert_actions_to_delta(actions)
                batch["pose"] = add_pose(prev_batch["pose"], actions_delta)
            batch["prev_actions"] = actions

        return batch

    def spatial_transform_map(self, p, x, invert=True, mode="bilinear"):
        """
        Inputs:
            p     - (bs, f, H, W) Tensor
            x     - (bs, 3) Tensor (x, y, theta) transforms to perform
        Outputs:
            p_trans - (bs, f, H, W) Tensor
        Conventions:
            Shift in X is rightward, and shift in Y is downward. Rotation is clockwise.

        Note: These denote transforms in an agent's position. Not the image directly.
        For example, if an agent is moving upward, then the map will be moving downward.
        To disable this behavior, set invert=False.
        """
        device = p.device
        H, W = p.shape[2:]

        trans_x = x[:, 0]
        trans_y = x[:, 1]
        # Convert translations to -1.0 to 1.0 range
        Hby2 = (H - 1) / 2 if H % 2 == 1 else H / 2
        Wby2 = (W - 1) / 2 if W % 2 == 1 else W / 2

        trans_x = trans_x / Wby2
        trans_y = trans_y / Hby2
        rot_t = x[:, 2]

        sin_t = torch.sin(rot_t)
        cos_t = torch.cos(rot_t)

        # This R convention means Y axis is downwards.
        A = torch.zeros(p.size(0), 3, 3).to(device)
        A[:, 0, 0] = cos_t
        A[:, 0, 1] = -sin_t
        A[:, 1, 0] = sin_t
        A[:, 1, 1] = cos_t
        A[:, 0, 2] = trans_x
        A[:, 1, 2] = trans_y
        A[:, 2, 2] = 1

        # Since this is a source to target mapping, and F.affine_grid expects
        # target to source mapping, we have to invert this for normal behavior.
        Ainv = torch.inverse(A)

        # If target to source mapping is required, invert is enabled and we invert
        # it again.
        if invert:
            Ainv = torch.inverse(Ainv)

        Ainv = Ainv[:, :2]
        grid = F.affine_grid(Ainv, p.size(), align_corners=True)
        p_trans = F.grid_sample(p, grid, mode=mode, align_corners=True)

        return p_trans

    def _compute_grid_impact_reward(self, features, new_features, vis_count):
        return ((torch.mean(torch.sqrt(F.mse_loss(features, new_features, reduction='none')),
                            dim=1)) / vis_count).view(-1, 1)

    def _compute_grid_pseudo_reward(self, vis_count):
        return ((torch.ones_like(vis_count)) / vis_count).view(-1, 1)

    def _is_master_process(self):
        return self.config.RANK == 0

    def setup_for_distributed(self, is_master):
        """
        This function disables printing when not in master process
        """
        import builtins as __builtin__
        builtin_print = __builtin__.print

        def print(*args, **kwargs):
            force = kwargs.pop('force', False)
            if is_master or force:
                builtin_print(*args, **kwargs)

        __builtin__.print = print

    def _dict_to_device(self, dictionary, device):
        new_dictionary = dict()
        for k in dictionary.keys():
            new_dictionary[k] = dictionary[k].to(device)
        return new_dictionary

    def train(self) -> None:
        r"""Main method for training PPO.

        Returns:
            None
        """
        if self.config.DISTRIBUTED is True:
            self.setup_for_distributed(self._is_master_process())

        self.envs = construct_envs(
            self.config,
            get_env_class(self.config.ENV_NAME),
            devices=self._assign_devices(),
        )

        ppo_cfg = self.config.RL.PPO
        ans_cfg = self.config.RL.ANS

        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        if not os.path.isdir(self.config.CHECKPOINT_FOLDER):
            os.makedirs(self.config.CHECKPOINT_FOLDER)
        self.mapper_rollouts = self._create_mapper_rollouts(ppo_cfg, ans_cfg)
        self._setup_actor_critic_agent(ppo_cfg, ans_cfg)
        self.logger.info(
            "mapper_agent number of parameters: {}".format(
                sum(param.numel() for param in self.mapper_agent.parameters())
            )
        )
        self.logger.info(
            "local_agent number of parameters: {}".format(
                sum(param.numel() for param in self.local_agent.parameters())
            )
        )
        self.logger.info(
            "global_agent number of parameters: {}".format(
                sum(param.numel() for param in self.global_agent.parameters())
            )
        )

        mapper_rollouts = self.mapper_rollouts
        global_rollouts = self._create_global_rollouts(ppo_cfg, ans_cfg)
        local_rollouts = self._create_local_rollouts(ppo_cfg, ans_cfg)
        self.impact_obs = self._create_impact_obs(ppo_cfg, ans_cfg)
        self.visitation_map = torch.zeros((self.envs.num_envs,
                                           ans_cfg.overall_map_size // ans_cfg.visitation_count_divider + 1,
                                           ans_cfg.overall_map_size // ans_cfg.visitation_count_divider + 1))
        # set visitation count of starting position to 1
        self.visitation_map[:, (self.visitation_map.shape[1] - 1) // 2, (self.visitation_map.shape[2] - 1) // 2] = 1

        global_rollouts.to(self.device)
        local_rollouts.to(self.device)

        # ===================== Create statistics buffers =====================
        statistics_dict = {}
        # Mapper statistics
        statistics_dict["mapper"] = defaultdict(
            lambda: deque(maxlen=ppo_cfg.loss_stats_window_size)
        )
        # Local policy statistics
        statistics_dict["local_policy"] = defaultdict(
            lambda: deque(maxlen=ppo_cfg.loss_stats_window_size)
        )
        # Global policy statistics
        statistics_dict["global_policy"] = defaultdict(
            lambda: deque(maxlen=ppo_cfg.loss_stats_window_size)
        )
        # Overall count statistics
        t_start = time.time()
        env_time = 0
        pth_time = 0

        # ==================== Measuring memory consumption ===================
        total_memory_size = 0
        print("=================== Mapper rollouts ======================")
        for k, v in mapper_rollouts.observations.items():
            mem = v.element_size() * v.nelement() * 1e-9
            print(f"key: {k:<40s}, memory: {mem:>10.4f} GB")
            total_memory_size += mem
        print(f"Total memory: {total_memory_size:>10.4f} GB")

        total_memory_size = 0
        print("================== Local policy rollouts =====================")
        for k, v in local_rollouts.observations.items():
            mem = v.element_size() * v.nelement() * 1e-9
            print(f"key: {k:<40s}, memory: {mem:>10.4f} GB")
            total_memory_size += mem
        print(f"Total memory: {total_memory_size:>10.4f} GB")

        total_memory_size = 0
        print("================== Global policy rollouts ====================")
        for k, v in global_rollouts.observations.items():
            mem = v.element_size() * v.nelement() * 1e-9
            print(f"key: {k:<40s}, memory: {mem:>10.4f} GB")
            total_memory_size += mem
        print(f"Total memory: {total_memory_size:>10.4f} GB")

        # Resume checkpoint if available
        (
            num_updates_start,
            count_steps_start,
            count_checkpoints,
        ) = self.resume_checkpoint()
        count_steps = count_steps_start

        M = ans_cfg.overall_map_size
        # ==================== Create state variables =================
        state_estimates = {
            # Agent's pose estimate
            "pose_estimates": torch.zeros(self.envs.num_envs, 3).to(self.device),
            # Agent's map
            "map_states": torch.zeros(self.envs.num_envs, 2, M, M).to(self.device),
            "recurrent_hidden_states": torch.zeros(
                1, self.envs.num_envs, ans_cfg.LOCAL_POLICY.hidden_size
            ).to(self.device),
            "visited_states": torch.zeros(self.envs.num_envs, 1, M, M).to(self.device),
        }
        ground_truth_states = {
            # To measure area seen
            "visible_occupancy": torch.zeros(
                self.envs.num_envs, 2, M, M, device=self.device
            ),
            "pose": torch.zeros(self.envs.num_envs, 3, device=self.device),
            "prev_global_reward_metric": torch.zeros(
                self.envs.num_envs, 1, device=self.device
            ),
        }
        if (
                ans_cfg.reward_type == "map_accuracy"
                or ans_cfg.LOCAL_POLICY.learning_algorithm == "il"
        ):
            ground_truth_states["environment_layout"] = torch.zeros(
                self.envs.num_envs, 2, M, M
            ).to(self.device)
        masks = torch.zeros(self.envs.num_envs, 1)
        new_masks = torch.zeros(self.envs.num_envs, 1)
        episode_step_count = torch.zeros(self.envs.num_envs, 1, device=self.device)

        # ==================== Reset the environments =================
        observations = self.envs.reset()
        if 'scene_name' in observations[0].keys():
            for i, obs in enumerate(observations):
                del observations[i]['scene_name']

        self.impact_obs = append_observations(self.config, self.impact_obs, observations, self.device)

        batch = self._prepare_batch(observations)
        prev_batch = batch
        # Update visible occupancy
        ground_truth_states["visible_occupancy"] = self.mapper.ext_register_map(
            ground_truth_states["visible_occupancy"],
            rearrange(batch["ego_map_gt"], "b h w c -> b c h w"),
            batch["pose_gt"],
        )
        ground_truth_states["pose"].copy_(batch["pose_gt"])

        current_local_episode_reward = torch.zeros(self.envs.num_envs, 1)
        current_global_episode_reward = torch.zeros(self.envs.num_envs, 1)
        running_episode_stats = dict(
            count=torch.zeros(self.envs.num_envs, 1),
            local_reward=torch.zeros(self.envs.num_envs, 1),
            global_reward=torch.zeros(self.envs.num_envs, 1),
        )
        window_episode_stats = defaultdict(
            lambda: deque(maxlen=ppo_cfg.reward_window_size)
        )

        # Useful variables
        NUM_MAPPER_STEPS = ans_cfg.MAPPER.num_mapper_steps
        NUM_LOCAL_STEPS = ppo_cfg.num_local_steps
        NUM_GLOBAL_STEPS = ppo_cfg.num_global_steps
        GLOBAL_UPDATE_INTERVAL = NUM_GLOBAL_STEPS * ans_cfg.goal_interval
        NUM_GLOBAL_UPDATES_PER_EPISODE = self.config.T_EXP // GLOBAL_UPDATE_INTERVAL
        NUM_GLOBAL_UPDATES = (
                self.config.NUM_EPISODES
                * NUM_GLOBAL_UPDATES_PER_EPISODE
                // self.config.NUM_PROCESSES
        )
        # Sanity checks
        assert (
                NUM_MAPPER_STEPS % NUM_GLOBAL_STEPS == 0
        ), "Mapper steps must be a multiple of global steps"
        assert (
                NUM_LOCAL_STEPS == ans_cfg.goal_interval
        ), "Local steps must be same as subgoal sampling interval"

        if self._is_master_process():
            writer = SummaryWriter(self.config.TENSORBOARD_DIR)
        else:
            writer = FalseWriter()

        for update in tqdm(range(num_updates_start, NUM_GLOBAL_UPDATES)):
            self.update = update
            for _ in tqdm(range(NUM_GLOBAL_STEPS)):

                (
                    delta_pth_time,
                    delta_env_time,
                    delta_steps,
                    prev_batch,
                    batch,
                    state_estimates,
                    ground_truth_states,
                ) = self._collect_rollout_step(
                    batch,
                    prev_batch,
                    episode_step_count,
                    state_estimates,
                    ground_truth_states,
                    masks,
                    new_masks,
                    mapper_rollouts,
                    local_rollouts,
                    global_rollouts,
                    current_local_episode_reward,
                    current_global_episode_reward
                )

                pth_time += delta_pth_time
                env_time += delta_env_time
                count_steps += delta_steps

                # Useful flags
                UPDATE_MAPPER_FLAG = (
                    True
                    if episode_step_count[0].item() % NUM_MAPPER_STEPS == 0
                    else False
                )
                UPDATE_LOCAL_FLAG = True

                # ------------------------ update mapper --------------------------
                if UPDATE_MAPPER_FLAG:
                    (
                        delta_pth_time,
                        update_metrics_mapper,
                    ) = self._update_mapper_agent(mapper_rollouts)

                    for k, v in update_metrics_mapper.items():
                        statistics_dict["mapper"][k].append(v)

                pth_time += delta_pth_time

                # -------------------- update local policy ------------------------
                if UPDATE_LOCAL_FLAG:
                    delta_pth_time = self._supplementary_rollout_update(
                        batch,
                        prev_batch,
                        episode_step_count,
                        state_estimates,
                        ground_truth_states,
                        masks.to(self.device),
                        local_rollouts,
                        global_rollouts,
                        update_option="local",
                    )

                    # Sanity check
                    assert local_rollouts.step == local_rollouts.num_steps

                    pth_time += delta_pth_time
                    (
                        delta_pth_time,
                        update_metrics_local,
                    ) = self._update_local_agent(local_rollouts)

                    for k, v in update_metrics_local.items():
                        statistics_dict["local_policy"][k].append(v)

                # -------------------------- log statistics -----------------------
                self.logger.info('')
                self.logger.info(f"================ UPDATE: {update:<4d} =================")
                for k, v in statistics_dict.items():
                    self.logger.info(
                        "=========== {:22s} ============".format(k + " stats")
                    )
                    for kp, vp in v.items():
                        if len(vp) > 0 and self._is_master_process():
                            if len(vp) > 25:  # arbitrary number to avoid initial tensorboard logging
                                writer.add_scalar(f"{k}/{kp}", np.mean(vp), count_steps * self.config.WORLD_SIZE)
                            self.logger.info(f"{kp:25s}: {np.mean(vp).item():10.5f}")

                for k, v in running_episode_stats.items():
                    window_episode_stats[k].append(v.clone())

                deltas = {
                    k: (
                        (v[-1] - v[0]).sum().item()
                        if len(v) > 1
                        else v[0].sum().item()
                    )
                    for k, v in window_episode_stats.items()
                }
                deltas["count"] = max(deltas["count"], 1.0)

                world_step = count_steps * self.config.WORLD_SIZE
                fps = (world_step - count_steps_start) / (time.time() - t_start)
                if self._is_master_process():
                    writer.add_scalar("local_reward", deltas["local_reward"] / deltas["count"], )
                    writer.add_scalar("global_reward", deltas["global_reward"] / deltas["count"], world_step)
                    writer.add_scalar("fps", fps, world_step)

                if update >= 0:
                    self.logger.info(
                        "fps: {:.3f}, env-time: {:.3f}s, pth-time: {:.3f}s, "
                        "frames: {}".format(fps, env_time, pth_time, count_steps * self.config.WORLD_SIZE)
                    )
                    self.logger.info(
                        "Average window size: {}  {}".format(
                            len(window_episode_stats["count"]),
                            "  ".join(
                                "{}: {:.3f},".format(k, v / deltas["count"])
                                for k, v in deltas.items()
                                if k != "count"
                            ),
                        )
                    )
                    self.logger.info("================================================\n")

                pth_time += delta_pth_time

            # At episode termination, manually set masks to zeros.
            if episode_step_count[0].item() == self.config.T_EXP:
                masks.fill_(0)

            # -------------------- update global policy -----------------------
            self._supplementary_rollout_update(
                batch,
                prev_batch,
                episode_step_count,
                state_estimates,
                ground_truth_states,
                masks.to(self.device),
                local_rollouts,
                global_rollouts,
                update_option="global",
            )

            # Sanity check
            assert global_rollouts.step == NUM_GLOBAL_STEPS

            (delta_pth_time, update_metrics_global,) = self._update_global_agent(
                global_rollouts
            )

            for k, v in update_metrics_global.items():
                statistics_dict["global_policy"][k].append(v)

            pth_time += delta_pth_time

            # checkpoint model
            if update % self.config.CHECKPOINT_INTERVAL == 0:
                if self._is_master_process():
                    if count_checkpoints != 0:
                        self.save_checkpoint(
                            f"ckpt.{count_checkpoints}.pth",
                            dict(step=count_steps, update=update),
                        )
                    count_checkpoints += 1

            # Manually enforce episode termination criterion
            if episode_step_count[0].item() == self.config.T_EXP:

                # Update episode rewards
                running_episode_stats["local_reward"] += (
                                                                 1 - masks
                                                         ) * current_local_episode_reward
                running_episode_stats["global_reward"] += (
                                                                  1 - masks
                                                          ) * current_global_episode_reward
                running_episode_stats["count"] += 1 - masks

                current_local_episode_reward = current_local_episode_reward * masks
                current_global_episode_reward = current_global_episode_reward * masks
                self.episode_custom_rewards = self.episode_custom_rewards * masks.cuda()
                # print(f"Masks: {masks}")
                for k in self.impact_obs.keys():
                    self.impact_obs[k].fill_(0)
                self.visitation_map.fill_(0)
                # set visitation count of starting position to 1
                self.visitation_map[:, (self.visitation_map.shape[1] - 1) // 2,
                (self.visitation_map.shape[2] - 1) // 2] = 1

                # Measure accumulative error in pose estimates
                pose_estimation_metrics = measure_pose_estimation_performance(
                    state_estimates["pose_estimates"], ground_truth_states["pose"]
                )
                for k, v in pose_estimation_metrics.items():
                    statistics_dict["mapper"]["episode_" + k].append(v)

                observations = self.envs.reset()

                # Handle impact observations
                self.impact_obs = append_observations(self.config, self.impact_obs, observations,
                                                      self.device)

                batch = self._prepare_batch(observations)
                prev_batch = batch
                # Reset episode step counter
                episode_step_count.fill_(0)
                # Reset states
                for k in ground_truth_states.keys():
                    ground_truth_states[k].fill_(0)
                for k in state_estimates.keys():
                    state_estimates[k].fill_(0)
                # Update visible occupancy
                ground_truth_states[
                    "visible_occupancy"
                ] = self.mapper.ext_register_map(
                    ground_truth_states["visible_occupancy"],
                    rearrange(batch["ego_map_gt"], "b h w c -> b c h w"),
                    batch["pose_gt"],
                )
                ground_truth_states["pose"].copy_(batch["pose_gt"])

        if self._is_master_process():
            writer.close()
        self.envs.close()

    def _eval_checkpoint(
            self,
            checkpoint_path: str,
            writer: TensorboardWriter,
            checkpoint_index: int = 0
    ) -> None:
        r"""Evaluates a single checkpoint.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object for logging to tensorboard
            checkpoint_index: index of cur checkpoint for logging

        Returns:
            None
        """
        checkpoint_index = int((re.findall('\d+', checkpoint_path))[-1])
        # Map location CPU is almost always better than mapping to a CUDA device.
        ckpt_dict = self.load_checkpoint(checkpoint_path, map_location="cpu")

        if self.config.EVAL.USE_CKPT_CONFIG:
            config = self._setup_eval_config(ckpt_dict["config"])
        else:
            config = self.config.clone()

        ppo_cfg = config.RL.PPO
        ans_cfg = config.RL.ANS

        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT
        config.freeze()

        if "COLLISION_SENSOR" not in config.TASK_CONFIG.TASK.SENSORS:
            config.TASK_CONFIG.TASK.SENSORS.append("COLLISION_SENSOR")
        if len(self.config.VIDEO_OPTION) > 0:
            config.defrost()
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP_EXP")
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("COLLISIONS")
            config.freeze()

        self.logger.info(f"env config: {config}")

        episode_number = None
        if episode_number is not None:
            config = self._create_tmp_dataset(episode_number, config)

        self.envs = construct_envs(config, get_env_class(config.ENV_NAME))
        self.mapper_rollouts = None
        self._setup_actor_critic_agent(ppo_cfg, ans_cfg)

        self.mapper_agent.load_state_dict(ckpt_dict["mapper_state_dict"])
        if self.local_agent is not None:
            self.local_agent.load_state_dict(ckpt_dict["local_state_dict"])
            self.local_actor_critic = self.local_agent.actor_critic
        else:
            self.local_actor_critic = self.ans_net.local_policy
        self.global_agent.load_state_dict(ckpt_dict["global_state_dict"])
        self.mapper = self.mapper_agent.mapper
        self.global_actor_critic = self.global_agent.actor_critic

        rgb_frames = [
            [] for _ in range(self.config.NUM_PROCESSES)
        ]  # type: List[List[np.ndarray]]
        if len(self.config.VIDEO_OPTION) > 0:
            os.makedirs(self.config.VIDEO_DIR, exist_ok=True)

        number_of_eval_episodes = self.config.TEST_EPISODE_COUNT
        if number_of_eval_episodes == -1:
            number_of_eval_episodes = sum(self.envs.number_of_episodes)
        else:
            total_num_eps = sum(self.envs.number_of_episodes)
            if total_num_eps < number_of_eval_episodes:
                logger.warn(
                    f"Config specified {number_of_eval_episodes} eval episodes"
                    f", dataset only has {total_num_eps}."
                )
                logger.warn(f"Evaluating with {total_num_eps} instead.")
                number_of_eval_episodes = total_num_eps

        assert (
                self.envs.num_envs == 1
        ), "Number of environments needs to be 1 for evaluation"

        # Set models to evaluation
        self.mapper.eval()
        self.local_actor_critic.eval()
        self.global_actor_critic.eval()

        M = ans_cfg.overall_map_size
        s = ans_cfg.MAPPER.map_scale

        # Define metric accumulators
        mapping_metrics = defaultdict(lambda: TemporalMetric())
        pose_estimation_metrics = defaultdict(lambda: TemporalMetric())
        s2d2_metrics = defaultdict(lambda: TemporalMetric())

        # Environment statistics
        episode_statistics = []
        episode_visualization_maps = []

        times_per_episode = deque(maxlen=100)

        increm_json_save_path = f"{self.config.VIDEO_DIR}/incremental_statistics_ckpt_{checkpoint_index}.json"
        if os.path.isfile(increm_json_save_path):
            with open(increm_json_save_path) as json_file:
                increm_per_episode_statistics = json.load(json_file)
        else:
            increm_per_episode_statistics = []

        if self.config.QUALITATIVE:
            increm_per_episode_statistics = []

        for ep in tqdm(range(number_of_eval_episodes)):
            if ep == 0:
                observations = self.envs.reset()

            if len(increm_per_episode_statistics) == number_of_eval_episodes:
                print("All episodes have been already evaluated!")
                break

            current_episodes = self.envs.current_episodes()

            if ep < len(increm_per_episode_statistics) and 'Nav' not in config.ENV_NAME:
                print(f"Len episode evaluated: {len(increm_per_episode_statistics)}")
                print(f"Skipping Episode {self.envs.current_episodes()[0].episode_id}...")
                observations = self.envs.reset()
                continue

            if self.config.QUALITATIVE and ep < self.config.QUALITATIVE_EPISODES[0]:
                print(f"Skipping Episode {self.envs.current_episodes()[0].episode_id}...")
                observations = self.envs.reset()
                continue
            if self.config.QUALITATIVE and ep >= self.config.QUALITATIVE_EPISODES[1]:
                print("All qualitative episodes have been already evaluated!")
                break

            batch = self._prepare_batch(observations)
            prev_batch = batch
            state_estimates = {
                "pose_estimates": torch.zeros(self.envs.num_envs, 3).to(self.device),
                "map_states": torch.zeros(self.envs.num_envs, 2, M, M).to(self.device),
                "recurrent_hidden_states": torch.zeros(
                    1, self.envs.num_envs, ans_cfg.LOCAL_POLICY.hidden_size
                ).to(self.device),
                "visited_states": torch.zeros(self.envs.num_envs, 1, M, M).to(
                    self.device
                ),
            }
            ground_truth_states = {
                "visible_occupancy": torch.zeros(self.envs.num_envs, 2, M, M).to(
                    self.device
                ),
                "pose": torch.zeros(self.envs.num_envs, 3).to(self.device),
                "environment_layout": None,
            }

            gt_map_states = torch.zeros_like(state_estimates['map_states'])

            # Reset ANS states
            self.ans_net.reset()

            current_episode_reward = torch.zeros(
                self.envs.num_envs, 1, device=self.device
            )

            prev_actions = torch.zeros(
                self.config.NUM_PROCESSES, 1, device=self.device, dtype=torch.long,
            )

            masks = torch.zeros(self.config.NUM_PROCESSES, 1, device=self.device)

            # Visualization stuff
            gt_agent_poses_over_time = [[] for _ in range(self.config.NUM_PROCESSES)]
            pred_agent_poses_over_time = [[] for _ in range(self.config.NUM_PROCESSES)]
            gt_map_agent = asnumpy(
                convert_world2map(ground_truth_states["pose"], (M, M), s)
            )
            pred_map_agent = asnumpy(
                convert_world2map(state_estimates["pose_estimates"], (M, M), s)
            )
            pred_map_agent = np.concatenate(
                [pred_map_agent, asnumpy(state_estimates["pose_estimates"][:, 2:3]), ],
                axis=1,
            )
            for i in range(self.config.NUM_PROCESSES):
                gt_agent_poses_over_time[i].append(gt_map_agent[i])
                pred_agent_poses_over_time[i].append(pred_map_agent[i])

            ep_start_time = time.time()
            for ep_step in tqdm(range(self.config.T_EXP)):
                ep_time = torch.zeros(
                    self.config.NUM_PROCESSES, 1, device=self.device
                ).fill_(ep_step)

                prev_pose_hat = state_estimates["pose_estimates"]
                with torch.no_grad():
                    (
                        mapper_inputs,
                        local_policy_inputs,
                        global_policy_inputs,
                        mapper_outputs,
                        local_policy_outputs,
                        global_policy_outputs,
                        state_estimates,
                        intrinsic_rewards,
                    ) = self.ans_net.act(
                        batch,
                        prev_batch,
                        state_estimates,
                        ep_time,
                        masks,
                        deterministic=ans_cfg.LOCAL_POLICY.deterministic_flag,
                    )

                    actions = local_policy_outputs["actions"]
                    prev_actions.copy_(actions)

                # Update GT estimates at t = ep_step
                ground_truth_states["pose"] = batch["pose_gt"]

                if self.config.RL.ANS.use_ddp:
                    ground_truth_states[
                        "visible_occupancy"
                    ] = self.ans_net.mapper.module.ext_register_map(
                        ground_truth_states["visible_occupancy"],
                        batch["ego_map_gt"].permute(0, 3, 1, 2),
                        batch["pose_gt"],
                    )
                else:
                    ground_truth_states[
                        "visible_occupancy"
                    ] = self.ans_net.mapper.ext_register_map(
                        ground_truth_states["visible_occupancy"],
                        batch["ego_map_gt"].permute(0, 3, 1, 2),
                        batch["pose_gt"],
                    )

                # Visualization stuff
                gt_map_agent = asnumpy(
                    convert_world2map(ground_truth_states["pose"], (M, M), s)
                )
                gt_map_agent = np.concatenate(
                    [gt_map_agent, asnumpy(ground_truth_states["pose"][:, 2:3])],
                    axis=1,
                )
                pred_map_agent = asnumpy(
                    convert_world2map(state_estimates["pose_estimates"], (M, M), s)
                )
                pred_map_agent = np.concatenate(
                    [
                        pred_map_agent,
                        asnumpy(state_estimates["pose_estimates"][:, 2:3]),
                    ],
                    axis=1,
                )
                for i in range(self.config.NUM_PROCESSES):
                    gt_agent_poses_over_time[i].append(gt_map_agent[i])
                    pred_agent_poses_over_time[i].append(pred_map_agent[i])

                outputs = self.envs.step([a[0].item() for a in actions])

                observations, _, dones, infos = [list(x) for x in zip(*outputs)]

                if ep_step == 0:
                    environment_layout = np.stack(
                        [info["gt_global_map"] for info in infos], axis=0
                    )  # (bs, M, M, 2)
                    environment_layout = rearrange(
                        environment_layout, "b h w c -> b c h w"
                    )  # (bs, 2, M, M)
                    environment_layout = torch.Tensor(environment_layout).to(
                        self.device
                    )
                    ground_truth_states["environment_layout"] = environment_layout
                    # Update environment statistics
                    if 'episode_statistics' in infos[0].keys():
                        for i in range(self.envs.num_envs):
                            episode_statistics.append(infos[i]["episode_statistics"])

                if ep_step == self.config.T_EXP - 1:
                    assert dones[0]

                prev_batch = batch
                batch = self._prepare_batch(observations, prev_batch, actions=actions)

                masks = torch.tensor(
                    [[0.0] if done else [1.0] for done in dones],
                    dtype=torch.float,
                    device=self.device,
                )

                n_envs = self.envs.num_envs

                if ep_step == 0 or (ep_step + 1) % 500 == 0:
                    curr_all_metrics = {}
                    # Compute accumulative pose estimation error
                    pose_hat_final = state_estimates["pose_estimates"]  # (bs, 3)
                    pose_gt_final = ground_truth_states["pose"]  # (bs, 3)
                    curr_pose_estimation_metrics = measure_pose_estimation_performance(
                        pose_hat_final, pose_gt_final, reduction="sum",
                    )
                    for k, v in curr_pose_estimation_metrics.items():
                        pose_estimation_metrics[k].update(
                            v, self.envs.num_envs, ep_step
                        )
                    curr_all_metrics.update(curr_pose_estimation_metrics)

                    # Compute map quality
                    curr_map_quality_metrics = measure_map_quality(
                        state_estimates["map_states"],
                        ground_truth_states["environment_layout"],
                        s,
                        entropy_thresh=1.0,
                        reduction="sum",
                        apply_mask=True,
                    )
                    for k, v in curr_map_quality_metrics.items():
                        mapping_metrics[k].update(v, self.envs.num_envs, ep_step)
                    curr_all_metrics.update(curr_map_quality_metrics)

                    # Compute area seen
                    area_reduction = "sum"
                    curr_area_seen_metrics = measure_area_seen_performance(
                        ground_truth_states["visible_occupancy"], s, reduction=area_reduction
                    )
                    for k, v in curr_area_seen_metrics.items():
                        mapping_metrics[k].update(v, self.envs.num_envs, ep_step)
                    curr_all_metrics.update(curr_area_seen_metrics)

                    # Debug stuff
                    if (ep_step + 1) == self.config.T_EXP:
                        times_per_episode.append(time.time() - ep_start_time)
                        mins_per_episode = np.mean(times_per_episode).item() / 60.0
                        eta_completion = mins_per_episode * (
                                number_of_eval_episodes - ep - 1
                        )
                        self.logger.info(
                            f"====> episode {ep}/{number_of_eval_episodes} done"
                        )
                        self.logger.info(
                            f"Time per episode: {mins_per_episode:.3f} mins"
                            f"\tETA: {eta_completion:.3f} mins"
                        )

                        for k, v in curr_all_metrics.items():
                            self.logger.info(f"{k:30s}: {v / self.envs.num_envs:8.3f}")

                for i in range(n_envs):
                    if (
                            len(self.config.VIDEO_OPTION) > 0
                            or self.config.SAVE_STATISTICS_FLAG
                    ):
                        # episode ended
                        if masks[i].item() == 0:
                            episode_visualization_maps.append(rgb_frames[i][-1])
                            video_metrics = {}
                            for k in curr_all_metrics.keys():
                                video_metrics[k] = curr_all_metrics[k]

                            if self.config.QUALITATIVE:
                                video_dir = f"qualitatives/{self.config.EVAL_CKPT_PATH_DIR.split('/')[-2]}/{self.config.TASK_CONFIG.DATASET.DATA_PATH.split('/')[3]}_{self.config.TASK_CONFIG.DATASET.SPLIT}"
                                cv2.imwrite(
                                    f"qualitatives/{self.config.EVAL_CKPT_PATH_DIR.split('/')[-2]}/{self.config.TASK_CONFIG.DATASET.DATA_PATH.split('/')[3]}_{self.config.TASK_CONFIG.DATASET.SPLIT}/ep_{ep}_top_down_map_{ep_step}_areaseen={curr_area_seen_metrics['area_seen']}.jpg",
                                    topdown_to_image(infos[0]["top_down_map_exp"])[..., ::-1])
                                print("Top down map saved...")
                            else:
                                video_dir = self.config.VIDEO_DIR

                            if len(self.config.VIDEO_OPTION) > 0:
                                generate_video(
                                    video_option=self.config.VIDEO_OPTION,
                                    video_dir=video_dir,
                                    images=rgb_frames[i],
                                    episode_id=current_episodes[i].episode_id,
                                    checkpoint_idx=checkpoint_index,
                                    metrics=video_metrics,
                                    tb_writer=writer,
                                )

                                rgb_frames[i] = []

                        # episode continues
                        elif (
                                len(self.config.VIDEO_OPTION) > 0
                                or ep_step == self.config.T_EXP - 2
                        ):
                            if observations[i]['rgb'].shape[0] != observations[i]['rgb'].shape[1]:
                                observations[i]['rgb'] = batch['rgb'][i]
                            if observations[i]['depth'].shape[0] != observations[i]['depth'].shape[1]:
                                observations[i]['depth'] = batch['depth'][i]
                            frame = observations_to_image(
                                observations[i], infos[i], observation_size=300
                            )
                            # Add ego_map_gt to frame
                            ego_map_gt_i = asnumpy(batch["ego_map_gt"][i])  # (2, H, W)
                            ego_map_gt_i = convert_gt2channel_to_gtrgb(ego_map_gt_i)
                            ego_map_gt_i = cv2.resize(ego_map_gt_i, (300, 300))
                            frame = np.concatenate([frame, ego_map_gt_i], axis=1)
                            # Generate ANS specific visualizations
                            environment_layout = asnumpy(
                                ground_truth_states["environment_layout"][i]
                            )  # (2, H, W)
                            visible_occupancy = asnumpy(
                                ground_truth_states["visible_occupancy"][i]
                            )  # (2, H, W)
                            curr_gt_poses = gt_agent_poses_over_time[i]
                            anticipated_occupancy = asnumpy(
                                state_estimates["map_states"][i]
                            )  # (2, H, W)
                            curr_pred_poses = pred_agent_poses_over_time[i]

                            H = frame.shape[0]
                            visible_occupancy_vis = generate_topdown_allocentric_map(
                                environment_layout,
                                visible_occupancy,
                                curr_gt_poses,
                                thresh_explored=ans_cfg.thresh_explored,
                                thresh_obstacle=ans_cfg.thresh_obstacle,
                            )
                            visible_occupancy_vis = cv2.resize(
                                visible_occupancy_vis, (H, H)
                            )
                            anticipated_occupancy_vis = generate_topdown_allocentric_map(
                                environment_layout,
                                anticipated_occupancy,
                                curr_pred_poses,
                                thresh_explored=ans_cfg.thresh_explored,
                                thresh_obstacle=ans_cfg.thresh_obstacle,
                            )
                            anticipated_occupancy_vis = cv2.resize(
                                anticipated_occupancy_vis, (H, H)
                            )
                            anticipated_action_map = generate_topdown_allocentric_map(
                                environment_layout,
                                anticipated_occupancy,
                                curr_pred_poses,
                                zoom=False,
                                thresh_explored=ans_cfg.thresh_explored,
                                thresh_obstacle=ans_cfg.thresh_obstacle,
                            )
                            global_goals = self.ans_net.states["curr_global_goals"]
                            local_goals = self.ans_net.states["curr_local_goals"]
                            if global_goals is not None:
                                cX = int(global_goals[i, 0].item())
                                cY = int(global_goals[i, 1].item())
                                anticipated_action_map = cv2.circle(
                                    anticipated_action_map,
                                    (cX, cY),
                                    10,
                                    (255, 0, 0),
                                    -1,
                                )
                            if local_goals is not None:
                                cX = int(local_goals[i, 0].item())
                                cY = int(local_goals[i, 1].item())
                                anticipated_action_map = cv2.circle(
                                    anticipated_action_map,
                                    (cX, cY),
                                    10,
                                    (0, 255, 255),
                                    -1,
                                )
                            anticipated_action_map = cv2.resize(
                                anticipated_action_map, (H, H)
                            )

                            maps_vis = np.concatenate(
                                [
                                    visible_occupancy_vis,
                                    anticipated_occupancy_vis,
                                    anticipated_action_map,
                                    np.zeros_like(anticipated_action_map),
                                ],
                                axis=1,
                            )
                            if self.config.RL.ANS.overall_map_size == 2001 or self.config.RL.ANS.overall_map_size == 961:
                                if frame.shape[1] < maps_vis.shape[1]:
                                    diff = maps_vis.shape[1] - frame.shape[1]
                                    npad = ((0, 0), (0, diff), (0, 0))
                                    frame = np.pad(frame, pad_width=npad, mode='constant', constant_values=0)
                                elif frame.shape[1] > maps_vis.shape[1]:
                                    diff = frame.shape[1] - maps_vis.shape[1]
                                    npad = ((0, 0), (0, diff), (0, 0))
                                    frame = np.pad(maps_vis, pad_width=npad, mode='constant', constant_values=0)
                                frame = frame[:, :1200]
                                maps_vis = maps_vis[:, :1200]
                            frame = np.concatenate([frame, maps_vis], axis=0)
                            rgb_frames[i].append(frame)

                        if self.config.QUALITATIVE and (ep_step + 1) % 500 == 0:
                            os.makedirs(
                                f"qualitatives/{self.config.EVAL_CKPT_PATH_DIR.split('/')[-2]}/{self.config.TASK_CONFIG.DATASET.DATA_PATH.split('/')[3]}_{self.config.TASK_CONFIG.DATASET.SPLIT}",
                                exist_ok=True)
                            cv2.imwrite(
                                f"qualitatives/{self.config.EVAL_CKPT_PATH_DIR.split('/')[-2]}/{self.config.TASK_CONFIG.DATASET.DATA_PATH.split('/')[3]}_{self.config.TASK_CONFIG.DATASET.SPLIT}/ep_{ep}_top_down_map_{ep_step}_areaseen={curr_area_seen_metrics['area_seen']}.jpg",
                                topdown_to_image(infos[0]["top_down_map_exp"])[..., ::-1])
                            print("Top down map saved...")
                # done-if
            # done-for

            if self.config.SAVE_STATISTICS_FLAG:
                # Logging results individually per episode
                per_episode_metrics = {}
                for k, v in mapping_metrics.items():
                    per_episode_metrics["mapping/" + k] = v.metric_list
                for k, v in pose_estimation_metrics.items():
                    per_episode_metrics["pose_estimation/" + k] = v.metric_list
                for k, v in s2d2_metrics.items():
                    per_episode_metrics["s2d2/" + k] = v.metric_list

                stats = {}
                for k, v in per_episode_metrics.items():
                    stats[k] = {}
                    for t in v.keys():
                        stats[k][t] = v[t][-1]

                if not self.config.QUALITATIVE and len(self.config.VIDEO_OPTION) == 0:
                    stats["episode_statistics"] = episode_statistics[-1]
                    increm_per_episode_statistics.append(stats)

                    json.dump(increm_per_episode_statistics, open(increm_json_save_path, "w"), indent=4)

        num_frames_per_process = (
                (checkpoint_index + 1)
                * self.config.CHECKPOINT_INTERVAL
                * self.config.T_EXP
                / self.config.RL.PPO.num_global_steps
        )

        if checkpoint_index == 0:
            try:
                eval_ckpt_idx = self.config.EVAL_CKPT_PATH_DIR.split("/")[-1].split(
                    "."
                )[1]
                logger.add_filehandler(
                    f"{self.config.TENSORBOARD_DIR}/results_ckpt_final_{eval_ckpt_idx}.txt"
                )
            except:
                logger.add_filehandler(
                    f"{self.config.TENSORBOARD_DIR}/results_ckpt_{checkpoint_index}.txt"
                )
        else:
            logger.add_filehandler(
                f"{self.config.TENSORBOARD_DIR}/results_ckpt_{checkpoint_index}.txt"
            )

        self.logger.info(
            f"======= Evaluating over {number_of_eval_episodes} episodes ============="
        )

        self.logger.info(f"=======> Mapping metrics")
        for k, v in mapping_metrics.items():
            metric_all_times = v.get_metric()
            for kp in sorted(list(metric_all_times.keys())):
                vp = metric_all_times[kp]
                self.logger.info(f"{k}: {kp},{vp}")
            writer.add_scalar(
                f"mapping_evaluation/{k}", v.get_last_metric(), num_frames_per_process,
            )

        self.logger.info(f"=======> Pose-estimation metrics")
        for k, v in pose_estimation_metrics.items():
            metric_all_times = v.get_metric()
            for kp in sorted(list(metric_all_times.keys())):
                vp = metric_all_times[kp]
                self.logger.info(f"{k}: {kp},{vp}")
            writer.add_scalar(f"pose_estimation_evaluation/{k}", v.get_last_metric(), num_frames_per_process)

        self.logger.info(f"=======> S2D2 metrics")
        for k, v in s2d2_metrics.items():
            metric_all_times = v.get_metric()
            for kp in sorted(list(metric_all_times.keys())):
                vp = metric_all_times[kp]
                self.logger.info(f"{k}: {kp},{vp}")
            writer.add_scalar(f"s2d2/{k}", v.get_last_metric(), num_frames_per_process)

        if self.config.SAVE_STATISTICS_FLAG:
            # Logging results individually per episode
            per_episode_metrics = {}
            for k, v in mapping_metrics.items():
                per_episode_metrics["mapping/" + k] = v.metric_list
            for k, v in pose_estimation_metrics.items():
                per_episode_metrics["pose_estimation/" + k] = v.metric_list
            for k, v in s2d2_metrics.items():
                per_episode_metrics["s2d2/" + k] = v.metric_list

            per_episode_dict = {k: [ep_stats[k] for ep_stats in increm_per_episode_statistics] for k in
                                increm_per_episode_statistics[0].keys()}
            print(f"Evaluating over {len(per_episode_dict['episode_statistics'])} episodes!")
            for k in per_episode_dict.keys():
                if k != 'episode_statistics':
                    for key in per_episode_dict[k][0].keys():
                        metric_list = [elem[key] for elem in per_episode_dict[k]]
                        print(f"{k}: {key}, {np.array(metric_list).mean()}")

        if episode_number is not None:
            self._remove_tmp_dataset(episode_number, config)
        self.envs.close()

    def _create_tmp_dataset(self, episode_number, config):
        dataset_path = config.TASK_CONFIG.DATASET.DATA_PATH.replace("{split}", config.TASK_CONFIG.DATASET.SPLIT)
        with gzip.open(dataset_path, "rt") as fp:
            dataset = json.load(fp)
        tmp_dataset = {"episodes": [dataset["episodes"][episode_number]]}
        tmp_dataset_path = os.path.join('/'.join(dataset_path.split('/')[:-1]), f'{episode_number}.json.gz')
        with gzip.open(tmp_dataset_path, 'w') as fout:
            fout.write(json.dumps(tmp_dataset).encode('utf-8'))
        config.defrost()
        config.TASK_CONFIG.DATASET.DATA_PATH = tmp_dataset_path
        config.freeze()
        return config

    def _remove_tmp_dataset(self, episode_number, config):
        dataset_path = config.TASK_CONFIG.DATASET.DATA_PATH.replace("{split}", config.TASK_CONFIG.DATASET.SPLIT)
        tmp_dataset_path = os.path.join('/'.join(dataset_path.split('/')[:-1]), f'{episode_number}.json.gz')
        os.remove(tmp_dataset_path)
