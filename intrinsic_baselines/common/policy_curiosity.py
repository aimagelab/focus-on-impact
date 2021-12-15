import copy
import random

import numpy as np
import torch
from gym.spaces.box import Box
from habitat_baselines.rl.ddppo.policy import resnet
from habitat_baselines.rl.ddppo.policy.resnet_policy import ResNetEncoder
from habitat_baselines.rl.models.rnn_state_encoder import RNNStateEncoder
from habitat_baselines.rl.ppo.policy import Policy, Net
from torch import nn
from torchvision.models import resnet50

from intrinsic_baselines.common.models import SimpleCNN, Flatten


class CuriosityBaselinePolicy(Policy):
    def __init__(
            self,
            observation_space,
            action_space,
            hidden_size=512,
            eps_greedy=0,
    ):
        super().__init__(
            CuriosityBaselineNet(
                observation_space=observation_space,
                hidden_size=hidden_size,
            ),
            action_space.n,
        )
        self.eps_greedy = eps_greedy

    def act_curiosity(self, observations, rnn_hidden_states, prev_actions,
                      masks,
                      deterministic=False, type=0):
        features, rnn_hidden_states = self.net(observations,
                                               rnn_hidden_states,
                                               prev_actions,
                                               masks)
        distribution = self.action_distribution(features)
        value = self.critic(features)
        if random.random() < self.eps_greedy:
            action = torch.tensor(
                [[random.randrange(len(distribution.probs))]]).cuda()
        else:
            if deterministic:
                action = distribution.mode()
            else:
                action = distribution.sample()

        action_log_probs = distribution.log_probs(action)
        dist_entropy = distribution.entropy().mean()

        return value, action, action_log_probs, rnn_hidden_states, features

    def evaluate_actions_curiosity(self, observations, rnn_hidden_states,
                                   observations_features,
                                   last_three_f_a_m,
                                   prev_actions, masks, action, i=None):
        features, rnn_hidden_states = self.net(observations, rnn_hidden_states,
                                               prev_actions, masks)
        distribution = self.action_distribution(features)
        value = self.critic(features)

        action_log_probs = distribution.log_probs(action)
        distribution_entropy = distribution.entropy().mean()

        return value, action_log_probs, distribution_entropy, rnn_hidden_states, features

    def get_features(self, observations, rnn_hidden_states, prev_actions, masks):
        features, _ = self.net(observations,
                               rnn_hidden_states,
                               prev_actions,
                               masks)
        return features


class CuriosityBaselineNet(Net):
    r"""Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    def __init__(self, observation_space, hidden_size):
        super().__init__()
        self._hidden_size = hidden_size

        # Changed initial size of input to cnn
        cnn_space = copy.deepcopy(observation_space)
        if 'rgb' in observation_space.spaces.keys():
            cnn_space.spaces['rgb'] = Box(low=0, high=255, shape=(84, 84, 4),
                                          dtype='uint8')
        if 'depth' in observation_space.spaces.keys():
            cnn_space.spaces['depth'] = Box(low=0, high=255, shape=(84, 84, 4),
                                            dtype='float32')

        self.visual_encoder = SimpleCNN(cnn_space, hidden_size)
        for param in self.visual_encoder.parameters():
            param.requires_grad = False

        self.state_encoder = RNNStateEncoder(
            (0 if self.is_blind else self._hidden_size),
            self._hidden_size,
        )

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def forward(self, observations, rnn_hidden_states, prev_actions, masks, use_rnn=True):
        if not self.is_blind:
            perception_embed = self.visual_encoder(observations)
            x = [perception_embed]
        else:
            blind_observation = torch.zeros_like(observations)
            perception_embed = self.visual_encoder(blind_observation)
            x = [perception_embed]

        x = torch.cat(x, dim=1)

        if use_rnn:
            x, rnn_hidden_states = self.state_encoder(x, rnn_hidden_states, masks)
        else:
            rnn_hidden_states = None

        return x, rnn_hidden_states

    def to_ddp(self, device_ids, output_device):
        self.pg1 = torch.distributed.new_group(range(torch.distributed.get_world_size()))
        self.state_encoder.rnn = nn.parallel.DistributedDataParallel(self.state_encoder.rnn, device_ids=device_ids,
                                                                     output_device=output_device,
                                                                     process_group=self.pg1)


class ImpactBaselineNet(Net):
    r"""Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    def __init__(self, observation_space, hidden_size):
        super().__init__()
        self._hidden_size = hidden_size

        # Changed initial size of input to cnn
        cnn_space = copy.deepcopy(observation_space)
        if 'rgb' in observation_space.spaces.keys():
            cnn_space.spaces['rgb'] = Box(low=0, high=255, shape=(42, 42, 4),
                                          dtype='uint8')
        if 'depth' in observation_space.spaces.keys():
            cnn_space.spaces['depth'] = Box(low=0, high=255, shape=(42, 42, 4),
                                            dtype='float32')

        self.visual_encoder = SimpleCNN(cnn_space, hidden_size)
        for param in self.visual_encoder.parameters():
            param.requires_grad = False

        self.state_encoder = RNNStateEncoder(
            (0 if self.is_blind else self._hidden_size),
            self._hidden_size,
        )

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def forward(self, observations, rnn_hidden_states, prev_actions, masks, use_rnn=True):
        if not self.is_blind:
            perception_embed = self.visual_encoder(observations)
            x = [perception_embed]
        else:
            blind_observation = torch.zeros_like(observations)
            perception_embed = self.visual_encoder(blind_observation)
            x = [perception_embed]

        x = torch.cat(x, dim=1)

        if use_rnn:
            x, rnn_hidden_states = self.state_encoder(x, rnn_hidden_states, masks)
        else:
            rnn_hidden_states = None

        return x, rnn_hidden_states

    def to_ddp(self, device_ids, output_device):
        self.pg1 = torch.distributed.new_group(range(torch.distributed.get_world_size()))
        self.state_encoder.rnn = nn.parallel.DistributedDataParallel(self.state_encoder.rnn, device_ids=device_ids,
                                                                     output_device=output_device,
                                                                     process_group=self.pg1)


class CuriosityResNetPolicy(Policy):
    def __init__(
            self,
            observation_space,
            action_space,
            hidden_size=512,
            num_recurrent_layers=1,
            rnn_type="LSTM",
            resnet_baseplanes=32,
            backbone="resnet50",
            normalize_visual_inputs=False,
            rgb=True,
            four_obs_gru=False,
            eps_greedy=0,
    ):
        if rgb is True:
            if four_obs_gru is True:
                super().__init__(
                    CuriosityResNetRGBNet(
                        observation_space=observation_space,
                        action_space=action_space,
                        hidden_size=hidden_size,
                        num_recurrent_layers=num_recurrent_layers,
                        rnn_type=rnn_type,
                        backbone=backbone,
                        resnet_baseplanes=resnet_baseplanes,
                        normalize_visual_inputs=normalize_visual_inputs,
                        use_4obs_gru=True,
                    ),
                    action_space.n,
                )
            else:
                super().__init__(
                    CuriosityResNetRGBNet(
                        observation_space=observation_space,
                        action_space=action_space,
                        hidden_size=hidden_size,
                        num_recurrent_layers=num_recurrent_layers,
                        rnn_type=rnn_type,
                        backbone=backbone,
                        resnet_baseplanes=resnet_baseplanes,
                        normalize_visual_inputs=normalize_visual_inputs,
                    ),
                    action_space.n,
                )
        else:
            super().__init__(
                CuriosityResNetNet(
                    observation_space=observation_space,
                    action_space=action_space,
                    hidden_size=hidden_size,
                    num_recurrent_layers=num_recurrent_layers,
                    rnn_type=rnn_type,
                    backbone=backbone,
                    resnet_baseplanes=resnet_baseplanes,
                    normalize_visual_inputs=normalize_visual_inputs,
                ),
                action_space.n,
            )
        self.eps_greedy = eps_greedy

    def act_curiosity(self, observations, rnn_hidden_states,
                      observations_features, last_three_f_a_m,
                      rollouts_step, prev_actions, masks, deterministic=False,
                      type=0):

        features, rnn_hidden_states = self.net(observations,
                                               rnn_hidden_states,
                                               observations_features,
                                               last_three_f_a_m,
                                               rollouts_step,
                                               prev_actions,
                                               masks)

        distribution = self.action_distribution(features)

        print(distribution.probs)
        value = self.critic(features)

        if random.random() < self.eps_greedy:
            action = torch.randint_like(distribution.sample(), 0,
                                        distribution.probs.size(-1)).cuda()
        else:
            if deterministic:
                action = distribution.mode()
            else:
                action = distribution.sample()

        action_log_probs = distribution.log_probs(action)
        dist_entropy = distribution.entropy().mean()

        return value, action, action_log_probs, rnn_hidden_states, features

    def evaluate_actions_curiosity(self, observations, rnn_hidden_states,
                                   observations_features,
                                   last_three_f_a_m,
                                   prev_actions, masks, action, i=None):
        rollouts_step = None
        features, rnn_hidden_states = self.net(observations,
                                               rnn_hidden_states,
                                               observations_features,
                                               last_three_f_a_m,
                                               rollouts_step,
                                               prev_actions,
                                               masks,
                                               evaluate=True,
                                               env=i)

        distribution = self.action_distribution(features)
        value = self.critic(features)

        action_log_probs = distribution.log_probs(action)
        distribution_entropy = distribution.entropy().mean()

        return value, action_log_probs, distribution_entropy, rnn_hidden_states, features

    def resnet_features(self, observations):
        x = self.net.resnet_features(observations)
        return x

    def concatenate(self, observations):
        x = self.net.concatenate(observations)
        return x

    def get_features(self, observations, rnn_hidden_states,
                     observations_features, last_three_f_a_m,
                     rollouts_step, prev_actions, masks):
        features, _ = self.net(observations,
                               rnn_hidden_states,
                               observations_features,
                               last_three_f_a_m,
                               rollouts_step,
                               prev_actions,
                               masks)
        return features

    def get_value(self, observations, rnn_hidden_states, observations_features,
                  last_three_f_a_m,
                  rollouts_step, prev_actions, masks):
        features, _ = self.net(observations,
                               rnn_hidden_states,
                               observations_features,
                               last_three_f_a_m,
                               rollouts_step,
                               prev_actions,
                               masks)
        return self.critic(features)


class CuriosityResNetNet(Net):
    r"""Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    def __init__(
            self,
            observation_space,
            action_space,
            hidden_size,
            num_recurrent_layers,
            rnn_type,
            backbone,
            resnet_baseplanes,
            normalize_visual_inputs,
    ):
        super().__init__()

        self.prev_action_embedding = nn.Embedding(action_space.n + 1, 32)
        self._n_prev_action = 32

        self._hidden_size = hidden_size * 4

        rnn_input_size = self._n_prev_action * 4  # + self._n_tgt_embeding
        self.visual_encoder = ResNetEncoder(
            observation_space,
            baseplanes=resnet_baseplanes,
            ngroups=resnet_baseplanes // 2,
            make_backbone=getattr(resnet, backbone),
            normalize_visual_inputs=normalize_visual_inputs,
        )

        if not self.visual_encoder.is_blind:
            self.visual_fc = nn.Sequential(
                Flatten(),
                nn.Linear(
                    np.prod(self.visual_encoder.output_shape), hidden_size
                ),
                nn.ReLU(True),
            )

        self.state_encoder = RNNStateEncoder(
            input_size=self._hidden_size + rnn_input_size,
            hidden_size=self._hidden_size,
            rnn_type=rnn_type,
            num_layers=num_recurrent_layers,
        )

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def resnet_features(self, observations):
        if not self.is_blind:
            perception_embed = self.visual_encoder(observations)
            perception_embed = self.visual_fc(perception_embed)
            x = [perception_embed]
        else:
            blind_observation = torch.zeros_like(observations)
            perception_embed = self.visual_encoder(blind_observation)
            perception_embed = self.visual_fc(perception_embed)
            x = [perception_embed]
        return x

    def forward(self, current_features, rnn_hidden_states,
                resnet_features, last_three_f_a_m,
                rollout_step, rollouts_actions,
                rollouts_masks, evaluate=False):

        if not evaluate:
            features = [current_features]
            actions = [rollouts_actions[rollout_step]]
            masks = [rollouts_masks[rollout_step]]

            for i in range(1, 4):
                if rollout_step - i >= 0:
                    features.append(resnet_features[rollout_step - i])
                    actions.append(rollouts_actions[rollout_step - i])
                    masks.append(rollouts_masks[rollout_step - i])
                else:
                    features.append(last_three_f_a_m[0][rollout_step - i][0])
                    actions.append(last_three_f_a_m[1][rollout_step - i])
                    masks.append(last_three_f_a_m[2][rollout_step - i])
            features.reverse()
            actions.reverse()
            masks.reverse()

            prev_actions_list = []
            for i in range(len(actions)):
                prev_actions_list.append(self.prev_action_embedding(
                    ((actions[i].float() + 1) * masks[i]).long().squeeze(-1)
                ))

            for i in range(len(features)):
                features[i] = torch.cat(
                    (features[i], prev_actions_list[i]), dim=1)

            tensor_features = torch.squeeze(torch.stack(features), dim=1).view(
                1, -1).cuda()
            tensor_masks = torch.squeeze(masks[3]).cuda()

            tensor_features, rnn_hidden_states = self.state_encoder(
                tensor_features, rnn_hidden_states, tensor_masks)

            return tensor_features, rnn_hidden_states

        else:
            tensor_features = []
            tensor_masks = []

            for rollout_step, feature in enumerate(resnet_features):
                features = []
                actions = []
                masks = []

                for i in range(0, 4):
                    if rollout_step - i >= 0:
                        features.append(resnet_features[rollout_step - i])
                        actions.append(rollouts_actions[rollout_step - i])
                        masks.append(rollouts_masks[rollout_step - i])
                    else:
                        features.append(torch.zeros_like(
                            resnet_features[rollout_step - i]))
                        actions.append(torch.zeros_like(
                            rollouts_actions[rollout_step - i]))
                        masks.append(
                            torch.zeros_like(rollouts_masks[rollout_step - i]))

                features.reverse()
                actions.reverse()
                masks.reverse()

                prev_actions_list = []
                for i in range(len(actions)):
                    prev_actions_list.append(self.prev_action_embedding(
                        ((torch.unsqueeze(actions[i], 0).float() + 1) * masks[
                            i]).long().squeeze(
                            -1)
                    ))

                for i in range(len(features)):
                    features[i] = torch.cat(
                        (features[i].cpu(), prev_actions_list[i].cpu()), dim=1)

                tensor_features.append(
                    torch.stack(features).view(1, -1).cuda())
                tensor_masks.append(masks[3].cuda())

            tensor_features = torch.squeeze(
                torch.stack(tensor_features, dim=0))
            tensor_masks = torch.stack(tensor_masks, dim=0)

            tensor_features, rnn_hidden_states = self.state_encoder(
                tensor_features, rnn_hidden_states, tensor_masks)

            return tensor_features, rnn_hidden_states


class CuriosityResNetRGBNet(Net):
    r"""Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    def __init__(
            self,
            observation_space,
            action_space,
            hidden_size,
            num_recurrent_layers,
            rnn_type,
            backbone,
            resnet_baseplanes,
            normalize_visual_inputs,
            use_4obs_gru=False,
    ):
        super().__init__()

        self.prev_action_embedding = nn.Embedding(action_space.n + 1, 32)
        self._n_prev_action = 32

        self._hidden_size = hidden_size

        rnn_input_size = self._n_prev_action
        self.visual_encoder = ResNetEncoder(
            observation_space,
            baseplanes=resnet_baseplanes,
            ngroups=resnet_baseplanes // 2,
            make_backbone=getattr(resnet, backbone),
            normalize_visual_inputs=normalize_visual_inputs,
        )
        self.visual_encoder_depth = self.visual_encoder

        self.visual_encoder_depth.compression = nn.Identity()
        self.visual_encoder = None
        self.visual_encoder_rgb = resnet50(pretrained=True)
        self.visual_encoder_rgb.fc = torch.nn.AdaptiveAvgPool1d(
            1024)
        if use_4obs_gru is True:
            self.observations_encoder = torch.nn.GRU(
                input_size=self._hidden_size,
                hidden_size=self._hidden_size,
                num_layers=1)

        self.state_encoder = RNNStateEncoder(
            input_size=self._hidden_size * 4 + rnn_input_size,
            hidden_size=self._hidden_size,
            rnn_type=rnn_type,
            num_layers=num_recurrent_layers,
        )

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder_depth.is_blind and not hasattr(self,
                                                                  'visual_encoder_rgb')

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def resnet_features(self, observations):
        if not self.is_blind:
            x = {}
            if 'rgb' in observations.keys():
                rgb_embed = self.visual_encoder_rgb(observations['rgb'])
                x.update({'rgb': rgb_embed})
            if not self.visual_encoder_depth.is_blind:
                depth_embed = self.visual_encoder_depth(
                    {"depth": observations['depth']})
                depth_embed = nn.AdaptiveAvgPool2d((1, 1)).forward(depth_embed)
                depth_embed = depth_embed.view(depth_embed.size(0), -1)
                x.update({'depth': depth_embed})
        else:
            x = None
        return x

    def concatenate(self, observations):
        x = torch.cat([observations['depth'], observations['rgb']], dim=1)
        x = self.visual_fc(x)
        return x

    def forward(self, observations, rnn_hidden_states, resnet_features,
                last_three_f_a_m,
                rollout_step, rollouts_actions,
                rollouts_masks, evaluate=False, env=None):

        if not evaluate:
            features = [resnet_features[rollout_step]]
            actions = [rollouts_actions[rollout_step]]
            masks = [rollouts_masks[rollout_step]]

            prev_actions_list = []

            for i in range(len(actions)):
                prev_actions_list.append(self.prev_action_embedding(
                    ((actions[i].float() + 1) * masks[i]).long().squeeze(-1)
                ))

            tensor_masks = torch.stack(masks).squeeze(0)

            if hasattr(self, "observations_encoder"):
                for i in range(1, 4):
                    if rollout_step - i >= 0:
                        features.append(resnet_features[rollout_step - i])
                    else:
                        features.append(last_three_f_a_m[0][rollout_step - i])
                features.reverse()

                tensor_features = torch.stack(features, dim=0)
                output_gru, hidden_states_gru = self.observations_encoder(
                    tensor_features)
                input_rnn = output_gru[-1]

                tensor_features = torch.cat((input_rnn,
                                             torch.stack(
                                                 prev_actions_list).squeeze(
                                                 dim=0)), dim=1)
            else:
                tensor_features = torch.cat((torch.stack(features,
                                                         dim=0).squeeze(dim=0),
                                             torch.stack(
                                                 prev_actions_list).squeeze(
                                                 dim=0)), dim=1)

            tensor_features, rnn_hidden_states = self.state_encoder(
                tensor_features, rnn_hidden_states, tensor_masks)

            return tensor_features, rnn_hidden_states

        else:
            tensor_features = self.resnet_features(observations)
            tensor_features = torch.cat((tensor_features[
                                             'rgb'],
                                         tensor_features[
                                             'depth']), dim=1)
            tensor_masks = rollouts_masks

            prev_actions_list = []
            for i in range(len(rollouts_actions)):
                prev_actions_list.append(self.prev_action_embedding(
                    ((rollouts_actions[i].float() + 1) * rollouts_masks[
                        i]).long()
                ))

            if hasattr(self, "observations_encoder"):
                for rollout_step in range(len(rollouts_actions)):

                    features = []

                    for i in range(0, 4):
                        if rollout_step - i >= 0:
                            features.append(resnet_features[rollout_step - i])
                        else:
                            features.append(
                                last_three_f_a_m[0][rollout_step - i])
                    features.reverse()

                    features = torch.stack(features, dim=0)
                    output_gru, hidden_states_gru = self.observations_encoder(
                        features[:, env, :].unsqueeze(1))
                    input_rnn = output_gru[-1]

                    tensor_features.append(input_rnn)

                tensor_features = torch.stack(tensor_features).squeeze(1)
                tensor_features = torch.cat(
                    (tensor_features,
                     torch.stack(prev_actions_list).squeeze(1)),
                    dim=1)

            else:
                tensor_features = torch.cat(
                    (tensor_features,
                     torch.stack(prev_actions_list).squeeze(1)),
                    dim=1)

            x, rnn_hidden_states = self.state_encoder(
                tensor_features, rnn_hidden_states, tensor_masks)

            return x, rnn_hidden_states
