import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from intrinsic_baselines.common.policy_curiosity import CuriosityBaselineNet

EPS_PPO = 1e-5


class CuriosityModels(nn.Module):
    def __init__(
            self,
            clip_param,
            update_epochs,
            num_mini_batch,
            value_loss_coef,
            entropy_coef,
            lr=None,
            eps=None,
            max_grad_norm=None,
            use_clipped_value_loss=True,
            use_normalized_advantage=True,
            observation_spaces=None,
            fwd_model=None,
            inv_model=None,
            curiosity_beta=0.2,
            curiosity_lambda=0.1,
            curiosity_hidden_size=512,
            device='cpu',
            use_ddp=False,
    ):

        super().__init__()

        self.clip_param = clip_param
        self.update_epochs = update_epochs
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        'Curiosity modules'
        self.fwd_model = fwd_model
        self.inv_model = inv_model
        self.obs_encoder = CuriosityBaselineNet(observation_spaces, curiosity_hidden_size)
        self.curiosity_beta = curiosity_beta
        self.curiosity_lambda = curiosity_lambda

        self.optimizer = optim.Adam(
            [{'params': self.fwd_model.parameters()},
             {'params': self.inv_model.parameters()}],
            lr=lr, eps=eps)

        self.use_normalized_advantage = use_normalized_advantage
        self.device = device
        self.use_ddp = use_ddp

    def forward(self, *x):
        raise NotImplementedError

    def curiosity_loss(self, features, actions_batch, T, N):
        curr_states = features.view(T, N, -1)[:-1].view(
            (T - 1) * N, -1)
        next_states = features.view(T, N, -1)[1:].view(
            (T - 1) * N, -1)
        acts = actions_batch.view(T, N, -1)[:-1]
        acts_one_hot = torch.zeros(T - 1, N,
                                   self.fwd_model.n_actions).to(
            self.device)
        acts_one_hot.scatter_(2, acts, 1)
        acts_one_hot = acts_one_hot.view((T - 1) * N, -1)
        acts = acts.view(-1)
        # Forward prediction loss
        pred_next_states = self.fwd_model(curr_states.detach(),
                                          acts_one_hot)
        fwd_loss = 0.5 * F.mse_loss(pred_next_states,
                                    next_states.detach())
        # Inverse prediction loss
        pred_acts = self.inv_model(curr_states, next_states)
        inv_loss = F.cross_entropy(pred_acts, acts.long())

        return self.curiosity_beta * fwd_loss + (
                1 - self.curiosity_beta) * inv_loss, fwd_loss, inv_loss

    def update(self, rollouts, ans_cfg):

        fwd_loss_epoch = 0
        inv_loss_epoch = 0

        data_generator = rollouts.recurrent_generator_curiosity(
            self.num_mini_batch
        )
        env = 0

        for _ in range(self.update_epochs):
            for sample in data_generator:
                (
                    obs_batch,
                    recurrent_hidden_states_batch,
                    actions_batch,
                    prev_actions_batch,
                    masks_batch,
                    adv_targ,
                    T,
                    N
                ) = sample

                features, rnn_hidden_states = self.obs_encoder(obs_batch, recurrent_hidden_states_batch,
                                                               prev_actions_batch, masks_batch,
                                                               use_rnn=ans_cfg.CURIOSITY.use_curiosity_rnn)

                self.optimizer.zero_grad()

                curiosity_loss, fwd_loss, inv_loss = self.curiosity_loss(
                    features,
                    actions_batch, T, N)

                self.before_backward(curiosity_loss)
                curiosity_loss.backward()
                self.after_backward(curiosity_loss)

                self.before_step()
                self.optimizer.step()
                self.after_step()

                fwd_loss_epoch += fwd_loss.item()
                inv_loss_epoch += inv_loss.item()
                env += 1

        num_updates = self.update_epochs * self.num_mini_batch

        fwd_loss_epoch /= num_updates
        inv_loss_epoch /= num_updates

        return fwd_loss_epoch, inv_loss_epoch

    def before_backward(self, loss):
        pass

    def after_backward(self, loss):
        pass

    def before_step(self):
        nn.utils.clip_grad_norm_(
            self.fwd_model.parameters(), self.max_grad_norm
        )
        nn.utils.clip_grad_norm_(
            self.inv_model.parameters(), self.max_grad_norm
        )

    def after_step(self):
        pass

    def to_ddp(self, device_ids, output_device):
        if self.use_ddp:
            self.pg1 = torch.distributed.new_group(range(torch.distributed.get_world_size()))
            self.inv_model = nn.parallel.DistributedDataParallel(self.inv_model, device_ids=device_ids,
                                                                 output_device=output_device, process_group=self.pg1)
            self.fwd_model.to_ddp(device_ids, output_device)
            self.obs_encoder.to_ddp(device_ids, output_device)
