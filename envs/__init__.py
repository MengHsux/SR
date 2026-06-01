#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 15:52:14 2018

@author: qiutian
"""
from gym.envs.registration import register

register(
    'Navigation2D-v1',
    entry_point='envs.navigation:Navigation2DEnvV1',
    max_episode_steps=1000
)

register(
    'Navigation2D-v2',
    entry_point='envs.navigation:Navigation2DEnvV2',
    max_episode_steps=1000
)

register(
    'Navigation2D-v3',
    entry_point='envs.navigation:Navigation2DEnvV3',
    max_episode_steps=1000
)
