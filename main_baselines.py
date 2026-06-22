import argparse
from itertools import count
import os, sys, random
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
# from tensorboardX import SummaryWriter
import time

start_time = time.time()
from tqdm import tqdm

### personal libraries
import envs
from buffers import ReplayBuffer, Auxiliarybuffer
from ddpg import DDPG

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', type=str, default="Navigation2D-v1")
parser.add_argument('--output', type=str, default='output/navi_v1')
parser.add_argument('--tau', default=0.05, type=float, help='target smoothing coefficient')
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--gamma', default=0.99, type=float)
parser.add_argument('--capacity', default=10000, type=int)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--random_seed', default=47, type=int)
parser.add_argument('--exploration_noise', default=0.1, type=float)
parser.add_argument('--max_epochs', default=200, type=int)
parser.add_argument('--max_steps', default=100, type=int)
parser.add_argument('--update_iterations', default=10, type=int)
parser.add_argument('--num_tasks', type=int, default=10)
parser.add_argument('--finetune', action='store_true', default=True)
parser.add_argument("--no_hyper", action="store_true")  # use regular critic
parser.add_argument("--reverse_hyper", action="store_true")  # use reverse critic
parser.add_argument("--random_hyper", action="store_true")  # use radom critic
parser.add_argument('--device', type=str, default='cuda:0')

parser.add_argument("--beta_schedule", type=str, default="linear", metavar='S',
                    help="linear, cosine or vp")
parser.add_argument('--n_timesteps', type=int, default=100, metavar='N',
                    help='diffusion timesteps (default: 100)')
parser.add_argument('--noise_ratio', type=float, default=1.0, metavar='G',
                    help='noise ratio in sample process (default: 1.0)')
parser.add_argument('--diffusion_lr', type=float, default=0.0003, metavar='G',
                    help='diffusion learning rate (default: 0.0003)')
parser.add_argument('--action_gradient_steps', type=int, default=20, metavar='N',
                    help='action gradient steps (default: 20)')
parser.add_argument('--ratio', type=float, default=0.1, metavar='G',
                    help='the ratio of action grad norm to action_dim (default: 0.1)')
parser.add_argument('--ac_grad_norm', type=float, default=2.0, metavar='G',
                    help='actor and critic grad norm (default: 1.0)')

args = parser.parse_args()
print(args)

torch.manual_seed(args.random_seed)
np.random.seed(args.random_seed)
np.set_printoptions(precision=3)
print('seed = %d' % args.random_seed)

device = torch.device(args.device if torch.cuda.is_available() else 'cuda')
if not os.path.exists(args.output):
    os.makedirs(args.output)

env = gym.make(args.env_name).unwrapped
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
A_space = env.action_space
print('State dim: %d, action dim: %d, max action: %.3f' % (state_dim, action_dim, max_action))

### generate a sequence of 50 tasks to simulate the lifelong learning environment
tasks = np.random.uniform(-0.5, 0.5, size=(args.num_tasks, 2))


###################### Small Functions ##########################
def softmax_normalize(array, temperature=1.0):
    array = np.array(array).reshape(-1)
    array -= array.mean()
    array_exp = np.exp(array * temperature)
    array_exp /= array_exp.sum()
    return array_exp


def collect_samples(agent, env, epochs=1, random=False):
    transitions = []
    transitions_s = []
    for i in range(epochs):
        state = env.reset()
        for t in range(args.max_steps):
            if random:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state)
                action_noise = np.random.normal(0, args.exploration_noise * max_action,
                                                size=env.action_space.shape[0])
                action = (action + action_noise).clip(env.action_space.low, env.action_space.high)

            next_state, reward, done, info = env.step(action)
            transitions.append((state, next_state, action, reward, np.float(done)))
            transitions_s.append((state, action))
            state = np.copy(next_state)
            if done: break
    print('Num of collected samples: %d' % len(transitions))
    return transitions, transitions_s


def train_ddpg(agent, replay_buffer, auxiliarybuffer, env, reservoir_buffer=None):
    rewards_on = np.zeros(args.max_epochs)
    rewards_off = np.zeros(args.max_epochs)
    coordinates = np.zeros((args.max_steps, 2))

    action_dim = env.action_space.shape[0]
    for idx in tqdm(range(args.max_epochs)):
        state = env.reset()
        ep_r = 0.0
        for t in range(args.max_steps):
            action = agent.select_action(state)
            action_noise = np.random.normal(0, args.exploration_noise * max_action, size=action_dim)
            action = (action + action_noise).clip(env.action_space.low, env.action_space.high)

            next_state, reward, done, info = env.step(action)
            ep_r += reward
            replay_buffer.push((state, next_state, action, reward, np.float(done)))
            auxiliarybuffer.push(state, action)

            state = next_state
            if done: break
        rewards_on[idx] = ep_r

        for it in range(args.update_iterations):
            if reservoir_buffer is not None:
                agent.update(replay_buffer, auxiliarybuffer=auxiliarybuffer, reservoir_buffer=reservoir_buffer,
                             batch_size=args.batch_size)
            else:
                agent.update(replay_buffer, auxiliarybuffer, batch_size=args.batch_size)

        ### evaluate
        state = env.reset()
        ep_r = 0.0
        for t in range(args.max_steps):
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            ep_r += reward
            state = next_state
            if done: break
        rewards_off[idx] = ep_r

    return agent, replay_buffer, auxiliarybuffer, rewards_on, rewards_off


EPOCHS = 10
N_INIT = 5
ITERS_INIT = 200
########################### Main Function ############################
if args.finetune:
    ### some hyper-parameters
    replay_buffer = ReplayBuffer(max_size=args.capacity)
    auxiliarybuffer = Auxiliarybuffer(max_size=args.capacity)
    agent_init = DDPG(args, state_dim, action_dim, max_action, lr=args.lr, tau=args.tau,
                      gamma=args.gamma, device=device)

    ### training the agent for initialization using domain randomization
    replay_buffer.reset(max_size=args.capacity)
    auxiliarybuffer.reset(max_size=args.capacity)
    for idx in range(N_INIT):
        env.reset_task(tasks[idx])
        transitions, transitions_s = collect_samples(agent_init, env, epochs=EPOCHS, random=True)
        replay_buffer.batch_push(transitions)
        auxiliarybuffer.batch_push(transitions_s)

    print('\n===== Training the universal model =====')
    print('Buffer size: %d' % len(replay_buffer.storage))
    for it in tqdm(range(ITERS_INIT)):
        _ = agent_init.update(replay_buffer, auxiliarybuffer, batch_size=args.batch_size)

    rewards_on = np.zeros((len(tasks), args.max_epochs))
    rewards_off = np.zeros((len(tasks), args.max_epochs))
    env.reset_task(tasks[0])
    replay_buffer.reset(max_size=args.capacity)
    auxiliarybuffer.reset(max_size=args.capacity)

    print('\n=============== Baseline: Fine-tune ===============')
    agent = DDPG(args, state_dim, action_dim, max_action, lr=args.lr, tau=args.tau,
                 gamma=args.gamma, device=device)
    for period in range(args.num_tasks):
        print("\n========== Period %d ==========" % (period + 1))
        print('Task %d' % (period + 1), tasks[period])
        env.reset_task(tasks[period])
        replay_buffer.reset(max_size=args.capacity)
        auxiliarybuffer.reset(max_size=args.capacity)
        agent.load_from(agent_init)
        agent, replay_buffer, auxiliarybuffer, r_on, r_off = train_ddpg(agent, replay_buffer, auxiliarybuffer, env)
        rewards_on[period] = r_on
        rewards_off[period] = r_off

        np.save(os.path.join(args.output, 'rewards.npy'), rewards_off)
        print('Average reward:', np.mean(rewards_off[:period + 1], axis=1))
