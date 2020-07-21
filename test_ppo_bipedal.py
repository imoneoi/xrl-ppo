import torch
import numpy as np

from torch.utils.tensorboard import SummaryWriter

from vectorenv.dummy import VectorEnv
from vectorenv.subproc import SubprocVectorEnv
from collector import Collector
from network import Network
from ppo import PPO

from test.test_envs.bipedal_walker import BipedalWalkerHardcore
from gym.wrappers.time_limit import TimeLimit

import time
from tqdm import tqdm

G_parallel_envs = 32

#C_avg_runs = 10
C_avg_runs = 1
C_save_epoch = 100

H_steps_per_iteration = 2048
H_batch_size = 2048
H_repeat = 10
H_lr = 2.5e-4

H_num_epochs = 50000
H_test_episodes = 64

def train_policy(seed, device):
    #logger
    summary_writer = SummaryWriter(log_dir="logs/ppo_bipedal_seed_" + str(seed))

    tqdm.write("Train with device " + device + " seed " + str(seed))

    #construct envs
    class BipedalWalkerHardcoreWrapper(object):
        def __init__(self, env, action_repeat=3, reward_scale=5, act_noise=0.3):
            self._env = env
            self.action_repeat = action_repeat
            self.reward_scale = reward_scale
            self.act_noise = act_noise

        def __getattr__(self, name):
            return getattr(self._env, name)

        def step(self, action):
            #add action noise
            action += self.act_noise * (-2 * np.random.random(4) + 1)

            r = 0.0
            for _ in range(self.action_repeat):
                obs_, reward_, done_, info_ = self._env.step(action)
                r = r + reward_

                if done_:
                    break

            #Scale reward
            return obs_, self.reward_scale * r, done_, info_

    #construct envs
    def MakeEnv():
        return BipedalWalkerHardcoreWrapper(TimeLimit(BipedalWalkerHardcore(), max_episode_steps=2000))

    def IsStop(reward):
        return reward >= 300 * 5

    train_env = SubprocVectorEnv([MakeEnv for _ in range(G_parallel_envs)])
    test_env  = SubprocVectorEnv([MakeEnv for _ in range(G_parallel_envs)])

    #seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_env.seed(seed)
    test_env.seed(seed)

    #construct policy
    device = torch.device(device)
    net = Network((24), (4), 1.0, device).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=H_lr)
    policy = PPO(net, optimizer, batch_size=H_batch_size)

    #construct collector
    train_collector = Collector(train_env, policy)
    test_collector  = Collector(test_env,  policy)

    total_duration = 0

    #train policy
    global_step = 0
    for epoch in range(H_num_epochs):
        #collect experience
        start_time = time.time()

        train_collector.clear_buffer()
        result = train_collector.collect(n_step=H_steps_per_iteration)

        collect_time = time.time() - start_time

        #sample experience
        start_time = time.time()
        batch = train_collector.get_experience()
        sample_time = time.time() - start_time

        #train model
        start_time = time.time()

        train_metric = policy.train(batch, H_repeat)

        train_time = time.time() - start_time

        total_duration += collect_time + sample_time + train_time
        tqdm.write("Collect Time: {:.2f}s / Sample Time: {:.2f}s / Train time: {:.2f}s".format(collect_time, sample_time, train_time))

        #increase global step
        global_step += result["n/st"]

        #write logs
        for k, v in train_metric.items():
            summary_writer.add_scalar(k, np.mean(v), global_step)
        for k, v in result.items():
            summary_writer.add_scalar(k, v, global_step)

        tqdm.write(str(result))

        #save model
        if not (epoch % C_save_epoch):
            torch.save(net.state_dict(), "checkpoints/ppo_bipedal_hardcore_step_" + str(epoch) + "_seed_" + str(seed))

        #need to stop?
        if IsStop(result["rew"]):
            #test
            test_result = test_collector.collect(n_episode=H_test_episodes)

            if IsStop(test_result["rew"]):
                break

    train_env.close()
    test_env.close()

    #visualize result
    render_env = VectorEnv([MakeEnv for _ in range(1)])
    render_collector  = Collector(render_env, policy)
    render_collector.collect(n_episode=1, render=True)

    return total_duration

seed = 2500
device = "cuda:0"

#Average all runs
run_time = []
for run_id in tqdm(range(C_avg_runs)):
    torch.cuda.empty_cache()

    run_time.append(train_policy(int(str(seed) + str(run_id)), device))

tqdm.write("{}s +/- {}s".format(np.mean(run_time), np.std(run_time)))