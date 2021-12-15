#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import habitat
from habitat.config import Config
from habitat.core.dataset import Dataset
from habitat_baselines.common.baseline_registry import baseline_registry


@baseline_registry.register_env(name="ExpRLEnv")
class ExpRLEnv(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        self._rl_config = config.RL
        self._core_env_config = config.TASK_CONFIG

        self._previous_action = None
        self._episode_distance_covered = None
        super().__init__(self._core_env_config, dataset)

    def reset(self):
        self._previous_action = None

        observations = super().reset()

        return observations

    def step(self, *args, **kwargs):
        self._previous_action = kwargs["action"]
        return super().step(*args, **kwargs)

    def get_reward_range(self):
        return (
            -1.0,
            +1.0,
        )

    def get_reward(self, observations):
        reward = 0
        return reward

    def get_done(self, observations):
        done = False
        if self._env.episode_over:
            done = True
        return done

    def get_info(self, observations):
        metrics = self.habitat_env.get_metrics()
        episode_statistics = {
            "episode_id": self.habitat_env.current_episode.episode_id,
            "scene_id": self.habitat_env.current_episode.scene_id,
        }
        metrics["episode_statistics"] = episode_statistics
        return metrics

    @baseline_registry.register_env(name="NavRLEnv")
    class NavRLEnv(habitat.RLEnv):
        def __init__(self, config: Config, dataset: Optional[Dataset] = None):
            self._rl_config = config.RL
            self._core_env_config = config.TASK_CONFIG
            self._reward_measure_name = self._rl_config.REWARD_MEASURE
            self._success_measure_name = self._rl_config.SUCCESS_MEASURE

            self._previous_measure = None
            self._previous_action = None
            super().__init__(self._core_env_config, dataset)

        def reset(self):
            self._previous_action = None
            observations = super().reset()
            self._previous_measure = self._env.get_metrics()[
                self._reward_measure_name
            ]
            return observations

        def step(self, *args, **kwargs):
            self._previous_action = kwargs["action"]
            return super().step(*args, **kwargs)

        def get_reward_range(self):
            return (
                self._rl_config.SLACK_REWARD - 1.0,
                self._rl_config.SUCCESS_REWARD + 1.0,
            )

        def get_reward(self, observations):
            reward = self._rl_config.SLACK_REWARD

            current_measure = self._env.get_metrics()[self._reward_measure_name]

            reward += self._previous_measure - current_measure
            self._previous_measure = current_measure

            if self._episode_success():
                reward += self._rl_config.SUCCESS_REWARD

            return reward

        def _episode_success(self):
            return self._env.get_metrics()[self._success_measure_name]

        def get_done(self, observations):
            done = False
            if self._env.episode_over or self._episode_success():
                done = True
            return done

        def get_info(self, observations):
            return self.habitat_env.get_metrics()

        def get_metrics(self):
            metrics = self.habitat_env.get_metrics()
            return metrics
