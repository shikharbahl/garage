import numpy as np
import tensorflow as tf

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import utils

__all__ = ["Actor", "Critic"]

"""
Relevant: https://github.com/openai/baselines/blob/master/baselines/ddpg/models.py
"""
class Model(object):
    def __init__(self, name):
        self.name = name

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

    @property
    def trainable_vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    @property
    def perturbable_vars(self):
        return [var for var in self.trainable_vars if 'LayerNorm' not in var.name]

# class Actor(nn.Module):
# 	def __init__(self, state_dim, action_dim, max_action):
# 		super(Actor, self).__init__()

# 		self.l1 = nn.Linear(state_dim, 400)
# 		self.l2 = nn.Linear(400, 300)
# 		self.l3 = nn.Linear(300, action_dim)

# 		self.max_action = max_action


# 	def forward(self, x):
# 		x = F.relu(self.l1(x))
# 		x = F.relu(self.l2(x))
# 		x = self.max_action * F.tanh(self.l3(x))
# 		return x
class Actor(Model):
    def __init__(self, nb_actions, name='actor', layer_norm=True):
        super(Actor, self).__init__(name=name)
        self.nb_actions = nb_actions
        self.layer_norm = layer_norm

    def __call__(self, obs, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            x = obs
            x = tf.layers.dense(x, 64)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)

            x = tf.layers.dense(x, 64)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)

            x = tf.layers.dense(x, self.nb_actions, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            x = tf.nn.tanh(x)
        return x

class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim, 400)
		self.l2 = nn.Linear(400, 300)
		self.l3 = nn.Linear(300, 1)

		# Q2 architecture
		self.l4 = nn.Linear(state_dim + action_dim, 400)
		self.l5 = nn.Linear(400, 300)
		self.l6 = nn.Linear(300, 1)


	def forward(self, x, u):
		x1 = F.relu(self.l1(torch.cat([x, u], 1)))
		x1 = F.relu(self.l2(x1))
		x1 = self.l3(x1)

		x2 = F.relu(self.l4(torch.cat([x, u], 1)))
		x2 = F.relu(self.l5(x2))
		x2 = self.l6(x2)

		return x1, x2
