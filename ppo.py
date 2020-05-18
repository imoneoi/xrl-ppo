import torch
import numpy as np
from torch import nn
from batch import Batch

'''
    Batch: state, action, reward, next_state, done, dummy
'''

class PPO(nn.Module):
    def __init__(self, net, optimizer,
                gamma=0.99, gae_lambda=0.95,
                eps_clip=0.2, dual_clip=None, value_clip=True,
                max_grad_norm=0.5,
                vf_coef=0.5, ent_coef=0.001,
                batch_size=32,
                normalize=True):
        super().__init__()

        assert 0 <= gae_lambda <= 1, 'GAE lambda should be in [0, 1].'
        assert dual_clip is None or dual_clip > 1, \
            'Dual-clip PPO parameter should greater than 1.'

        self.net = net
        self.optimizer = optimizer

        self._gamma = gamma
        self._gae_lambda = gae_lambda

        self._eps_clip = eps_clip
        self._dual_clip = dual_clip
        self._value_clip = value_clip

        self._max_grad_norm = max_grad_norm

        self._vf_coef = vf_coef
        self._ent_coef = ent_coef

        self._batch_size = batch_size

        self._normalize = normalize

        self.__eps = np.finfo(np.float32).eps.item()

    def forward(self, state, sample=True, inference_policy=True, estimate_value=False):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float, device=self.net.device)

        policy, value = self.net(state, inference_policy, estimate_value)

        if sample:
            return policy.sample().cpu().numpy()

        return policy, value

    def calculate_gae(self, batch):
        m = self._gamma * (1.0 - np.array(batch.done))
        delta = batch.reward + m * np.roll(batch.value, -1, axis=0) - batch.value
        delta[batch.dummy] = 0

        m *= self._gae_lambda
        gae = 0.
        batch.adv = np.zeros_like(batch.reward)
        for i in range(len(batch.reward) - 1, -1, -1):
            gae = delta[i] + m[i] * gae
            batch.adv[i] = gae

        #calculate \hat{return} = \hat{v} + \hat{adv}(gae)
        batch.returns = batch.value + batch.adv

    def train(self, batch, repeat=1):
        #transfer state and action to tensors
        batch.state  = torch.tensor(batch.state,  dtype=torch.float, device=self.net.device)
        batch.action = torch.tensor(batch.action, dtype=torch.float, device=self.net.device)

        #normalize reward
        if self._normalize:
            #remove dummy items
            reward_clean = np.array(batch.reward)[np.logical_not(batch.dummy)]

            mean, std = np.mean(reward_clean), np.std(reward_clean)
            if std > self.__eps:
                batch.reward = (batch.reward - mean) / std

        #calculate log(p_{old}) and v
        value    = []
        logp_old = []
        with torch.no_grad():
            for b in batch.split(self._batch_size):
                b_policy, b_value = self(b.state, sample=False, inference_policy=True, estimate_value=True)
                b_logp_old = b_policy.log_prob(b.action).sum(-1)

                value.append(b_value)
                logp_old.append(b_logp_old)

        batch.value    = torch.cat(value,    dim=0).cpu().numpy()
        batch.logp_old = torch.cat(logp_old, dim=0)

        #calculate advantage
        self.calculate_gae(batch)

        #remove dummies set for advantage calculation
        batch = batch[np.logical_not(batch.dummy)]

        #transfer value adv return to tensor
        batch.adv     = torch.tensor(batch.adv,     dtype=torch.float, device=self.net.device)
        batch.returns = torch.tensor(batch.returns, dtype=torch.float, device=self.net.device)
        batch.value   = torch.tensor(batch.value,   dtype=torch.float, device=self.net.device)

        #normalize return and advantage
        if self._normalize:
            #normalize returns
            mean, std = batch.returns.mean(), batch.returns.std()
            if std > self.__eps:
                batch.returns = (batch.returns - mean) / std

            #normalize advantage
            batch.adv = batch.returns - batch.value

            mean, std = batch.adv.mean(), batch.adv.std()
            if std > self.__eps:
                batch.adv = (batch.adv - mean) / std

        #train model
        losses, clip_losses, vf_losses, ent_losses = [], [], [], []

        #TODO: CHECK MULTI ACTIONS OK!!!!!!!
        #TODO: Dual Clip Bug!!!!!!

        for _ in range(repeat):
            for b in batch.split(self._batch_size, shuffle=True):
                #forward prop policy and value
                policy, value = self(b.state, sample=False, inference_policy=True, estimate_value=True)

                ratio = (policy.log_prob(b.action).sum(-1) - b.logp_old).exp().float()
                surr1 = ratio * b.adv
                surr2 = ratio.clamp(
                    1. - self._eps_clip, 1. + self._eps_clip) * b.adv

                if self._dual_clip:
                    clip_loss = -torch.max(torch.min(surr1, surr2),
                                           self._dual_clip * b.adv).mean()
                else:
                    clip_loss = -torch.min(surr1, surr2).mean()

                clip_losses.append(clip_loss.item())

                if self._value_clip:
                    v_clip = b.value + (value - b.value).clamp(
                        -self._eps_clip, self._eps_clip)
                    vf1 = (b.returns - value).pow(2)
                    vf2 = (b.returns - v_clip).pow(2)
                    vf_loss = .5 * torch.max(vf1, vf2).mean()
                else:
                    vf_loss = .5 * (b.returns - value).pow(2).mean()

                vf_losses.append(vf_loss.item())

                ent_loss = policy.entropy().sum(-1).mean()
                ent_losses.append(ent_loss.item())

                loss = clip_loss + self._vf_coef * vf_loss - self._ent_coef * ent_loss
                losses.append(loss.item())

                # backprop
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self._max_grad_norm)
                self.optimizer.step()

        return {
            'loss': losses,
            'loss/clip': clip_losses,
            'loss/vf': vf_losses,
            'loss/ent': ent_losses,
        }