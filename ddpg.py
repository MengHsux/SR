import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from buffers import ReplayBuffer, Auxiliarybuffer
import os, math
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def weight_init(module):
    if isinstance(module, nn.Linear):
        fan_in = module.weight.size(-1)
        w = 1. / np.sqrt(fan_in)
        nn.init.uniform_(module.weight, -w, w)
        nn.init.uniform_(module.bias, -w, w)


WEIGHTS_FINAL_INIT = 3e-3
BIAS_FINAL_INIT = 3e-4
N1 = 256
N2 = 256


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, N1)
        self.ln1 = nn.LayerNorm(N1)
        self.l2 = nn.Linear(N1, N2)
        self.ln2 = nn.LayerNorm(N2)
        self.l3 = nn.Linear(N2, action_dim)
        self.max_action = max_action

        self.apply(weight_init)

    def forward(self, x):
        x = self.l1(x)
        x = self.ln1(x)
        x = F.relu(x)
        x = self.l2(x)
        x = self.ln2(x)
        x = F.relu(x)
        x = self.max_action * torch.tanh(self.l3(x))
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, N1)
        self.ln1 = nn.LayerNorm(N1)
        self.l2 = nn.Linear(N1, N2)
        self.ln2 = nn.LayerNorm(N2)
        self.l3 = nn.Linear(N2, 1)
        self.apply(weight_init)

        self.l4 = nn.Linear(state_dim + action_dim, N1)
        self.ln4 = nn.LayerNorm(N1)
        self.l5 = nn.Linear(N1, N2)
        self.ln5 = nn.LayerNorm(N2)
        self.l6 = nn.Linear(N2, 1)
        self.apply(weight_init)

    def forward(self, x, u):
        if len(x.shape) == 3:
            sa = torch.cat([x, u], 2)
        else:
            sa = torch.cat([x, u], 1)
        xp = self.l1(sa)

        x = self.ln1(xp)
        x = F.relu(x)
        x = self.l2(x)
        x = self.ln2(x)
        x = F.relu(x)
        x = self.l3(x)

        x1 = self.ln4(xp)
        x1 = F.relu(x1)
        x1 = self.l5(x1)
        x1 = self.ln5(x1)
        x1 = F.relu(x1)
        x1 = self.l6(x1)
        return x, x1

    def Q1(self, x, u):
        x = self.l1(torch.cat([x, u], 1))
        x = self.ln1(x)
        x = F.relu(x)
        x = self.l2(x)
        x = self.ln2(x)
        x = F.relu(x)
        x = self.l3(x)

        return x


class DDPG(object):
    def __init__(self, args, state_dim, action_dim, max_action, lr=1e-3, gamma=0.99,
                 tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2, device='cuda'):

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.1 * lr)

        self.action_gradient_steps = args.action_gradient_steps
        self.action_grad_norm = action_dim * args.ratio
        self.ac_grad_norm = args.ac_grad_norm

        critic = Critic

        self.critic = critic(state_dim, action_dim).to(device)
        self.critic_target = critic(state_dim, action_dim).to(device)
        self.copy_params(self.critic_target, self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.max_action = max_action
        self.discount = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.total_it = 0
        self.action_lr = args.lr
        self.lmbda = 0.75

        self.action_dim = action_dim

    def load_from(self, agent):
        self.actor.load_state_dict(agent.actor.state_dict())
        self.actor_target.load_state_dict(agent.actor_target.state_dict())
        self.critic.load_state_dict(agent.critic.state_dict())
        self.critic_target.load_state_dict(agent.critic_target.state_dict())

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action = self.actor(state)
        action = action.cpu().data.numpy().flatten()
        return action

    def copy_params(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def action_gradient(self, batch_size, auxiliarybuffer):
        states, best_actions, idxs = auxiliarybuffer.sample(batch_size)
        states = torch.FloatTensor(states).to(device)
        best_actions = torch.FloatTensor(best_actions).to(device)

        actions_optim = torch.optim.Adam([best_actions], lr=self.action_lr, eps=1e-5)
        states_optim = torch.optim.Adam([states], lr=self.action_lr, eps=1e-5)

        for i in range(self.action_gradient_steps):
            best_actions.requires_grad_(True)
            states.requires_grad_(True)

            q1, q2 = self.critic(states, best_actions)

            # Soft Clipped Double Q-learning
            target_Q = torch.min(q1, q2)
            loss = -target_Q

            actions_optim.zero_grad()
            states_optim.zero_grad()
            loss.backward(torch.ones_like(loss))
            actions_optim.step()
            states_optim.step()

            best_actions.requires_grad_(False)
            states.requires_grad_(False)
            best_actions.clamp_(-1., 1.)

        best_actions = best_actions.detach()
        states = states.detach()

        auxiliarybuffer.replace(idxs, best_actions.cpu().numpy())
        auxiliarybuffer.replace1(idxs, states.cpu().numpy())

        return states, best_actions

    def update(self, replay_buffer, auxiliarybuffer, batch_size=32):
        self.total_it += 1
        # Sample replay buffer
        x, y, u, r, d = replay_buffer.sample(batch_size)

        state = torch.FloatTensor(x).to(device)
        next_state = torch.FloatTensor(y).to(device)
        action = torch.FloatTensor(u).to(device)
        done = torch.FloatTensor(d).to(device)
        reward = torch.FloatTensor(r).to(device)

        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + ((1 - done) * self.discount * target_Q).detach()

        # Get current Q estimate
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = 0

        if self.total_it % self.policy_freq == 0:

            q1, q2 = self.critic(state, self.actor(state))
            actor_loss = -q1.mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Policy Training
            states, best_actions = self.action_gradient(batch_size, auxiliarybuffer)
            # Calculate the mean squared error between the actor output and the optimal action
            actor_pred = self.actor(states)  # Action predicted by the Actor network based on the state
            actor_loss1 = F.mse_loss(actor_pred, best_actions)  # MSE loss
            # ============================================================

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss1.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return actor_loss, critic_loss
