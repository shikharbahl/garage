import pickle

import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc

from garage.algos.base import RLAlgorithm
from garage.envs.util import bounds, flat_dim
from garage.misc import logger
from garage.misc.overrides import overrides
from garage.tf.misc import tensor_utils
# from garage.tf.algos.td3 import Actor, Critic

__all__ = ["TD3"]


"""
Implementation of Twin Delayed Deep Deterministic Policy Gradient (TD3)
algorithm based on https://arxiv.org/pdf/1802.09477.pdf.
"""
class TD3(object):
	def __init__(self,
				#  state_dim,
				#  action_dim,
				#  max_action):
                 env,
                 actor,
                 critic1, # taking the minimum value between a pair of critics to limit overestimation
				 critic2,
                 n_epochs=500,
                 n_epoch_cycles=20,
                 n_rollout_steps=100,
                 n_train_steps=50,
                 reward_scale=1.,
                 batch_size=64,
                 target_update_tau=0.01,
                 discount=0.99,
                 actor_lr=1e-4,
                 critic_lr=1e-3,
                 actor_weight_decay=0,
                 critic_weight_decay=0,
                 replay_buffer_size=int(1e6),
                 min_buffer_size=10000,
                 exploration_strategy=None,
                 plot=False,
                 pause_for_plot=False,
                 actor_optimizer=None,
                 critic_optimizer=None):
        self.env = env

        self.observation_dim = flat_dim(env.observation_space)
        self.action_dim = flat_dim(env.action_space)
        self.action_bound = env.action_space.high

        self.actor = actor
        self.critic = critic
        self.n_epochs = n_epochs
        self.n_epoch_cycles = n_epoch_cycles
        self.n_rollout_steps = n_rollout_steps
        self.n_train_steps = n_train_steps
        self.reward_scale = reward_scale
        self.batch_size = batch_size
        self.tau = target_update_tau
        self.discount = discount
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.actor_weight_decay = actor_weight_decay
        self.critic_weight_decay = critic_weight_decay
        self.replay_buffer_size = replay_buffer_size
        self.min_buffer_size = min_buffer_size
        self.es = exploration_strategy
        self.plot = plot
        self.pause_for_plot = pause_for_plot
        self.actor_optimizer = actor_optimizer
        self.critic_optimzier = critic_optimizer

		self.actor = actor
		self.actor_target =

###
		self.actor = Actor(state_dim, action_dim, max_action)
		self.actor_target = Actor(state_dim, action_dim, max_action)
		self.actor_target.load_state_dict(self.actor.state_dict())
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

		self.critic = Critic(state_dim, action_dim)
		self.critic_target = Critic(state_dim, action_dim)
		self.critic_target.load_state_dict(self.critic.state_dict())
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

		if torch.cuda.is_available():
			self.actor = self.actor.cuda()
			self.actor_target = self.actor_target.cuda()
			self.critic = self.critic.cuda()
			self.critic_target = self.critic_target.cuda()

		self.criterion = nn.MSELoss()
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.max_action = max_action




	def select_action(self, state):
		state = var(torch.FloatTensor(state.reshape(-1, self.state_dim)), volatile=True)
		return self.actor(state).cpu().data.numpy().flatten()

	@overrides
	def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):

		for it in range(iterations):

			# Sample replay buffer
			x, y, u, r, d = replay_buffer.sample(batch_size)
			state = var(torch.FloatTensor(x))
			action = var(torch.FloatTensor(u))
			next_state = var(torch.FloatTensor(y), volatile=True)
			done = var(torch.FloatTensor(1 - d))
			reward = var(torch.FloatTensor(r))

			# Select action according to policy and add clipped noise
			noise = np.clip(np.random.normal(0, policy_noise, size=(batch_size, self.action_dim)), -noise_clip, noise_clip)
			next_action = self.actor_target(next_state) + var(torch.FloatTensor(noise))
			next_action = next_action.clamp(-self.max_action, self.max_action)

			# Q target = reward + discount * min_i(Qi(next_state, pi(next_state)))
			target_Q1, target_Q2 = self.critic_target(next_state, next_action)
			target_Q = torch.min(target_Q1, target_Q2)
			target_Q = reward + (done * discount * target_Q)
			target_Q.volatile = False

			# Get current Q estimates
			current_Q1, current_Q2 = self.critic(state, action)

			# Compute critic loss
			critic_loss = self.criterion(current_Q1, target_Q) + self.criterion(current_Q2, target_Q)

			# Optimize the critic
			self.critic_optimizer.zero_grad()
			critic_loss.backward()
			self.critic_optimizer.step()

			# Delayed policy updates
			if it % policy_freq == 0:

				# Compute actor loss
				Q1, Q2 = self.critic(state, self.actor(state))
				actor_loss = -Q1.mean()

				# Optimize the actor
				self.actor_optimizer.zero_grad()
				actor_loss.backward()
				self.actor_optimizer.step()

				# Update the frozen target models
				for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
					target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

				for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
					target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


	def save(self, filename, directory):
		torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
		torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))


	def load(self, filename, directory):
		self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
		self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))
