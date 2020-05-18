import torch
import numpy as np
from torch import nn
from batch import Batch

class Trainer:
    def __init__(self, policy, collector):
        self.policy = policy
        self.collector = collector

    def train(self, sample_repeats, n_collect_episodes=0, n_collect_steps=0):
        