import torch
import numpy as np

from vectorenv.dummy import VectorEnv
from vectorenv.subproc import SubprocVectorEnv
from collector import Collector
from network import Network
from ppo import PPO

import gym

import time
from tqdm import tqdm

#C_avg_runs = 10
C_avg_runs = 1

H_num_epochs = 2500
H_steps_per_iteration = 2048
H_test_episodes = 10
H_repeat = 10
H_lr = 2.5e-4

def train_policy(seed):
    #construct envs
    def MakeEnv():
        return gym.make("BipedalWalker-v3")

    def IsStop(reward):
        return reward >= 300

    train_env = SubprocVectorEnv([MakeEnv for _ in range(16)])
    test_env  = SubprocVectorEnv([MakeEnv for _ in range(16)])

    #seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_env.seed(seed)
    test_env.seed(seed)

    #construct policy
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = Network((24), (4), 1.0, device).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=H_lr)
    policy = PPO(net, optimizer)

    #construct collector
    train_collector = Collector(train_env, policy)
    test_collector  = Collector(test_env,  policy)

    total_duration = 0

    #train policy
    for _ in range(H_num_epochs):
        start_time = time.time()

        #collect experience
        train_collector.clear_buffer()
        result = train_collector.collect(n_step=H_steps_per_iteration)
        batch = train_collector.get_experience()

        #train model
        train_metric = policy.train(batch, H_repeat)

        total_duration += time.time() - start_time

        avg_metric = {}
        for k, v in train_metric.items():
            avg_metric[k] = np.mean(v)

        tqdm.write(str(result))
        tqdm.write(str(avg_metric))

        #need to stop?
        if IsStop(result["rew"]):
            #test
            test_result = test_collector.collect(n_episode=H_test_episodes)

            if IsStop(test_result["rew"]):
                break

    train_env.close()
    test_env.close()

    #save model
    torch.save(net.state_dict(), "checkpoints/bipedal")

    #visualize result
    render_env = VectorEnv([MakeEnv for _ in range(1)])
    render_collector  = Collector(render_env, policy)
    render_collector.collect(n_episode=1, render=True)

    return total_duration

#Average all runs
run_time = []
for run_id in tqdm(range(C_avg_runs)):
    torch.cuda.empty_cache()

    run_time.append(train_policy(run_id))

tqdm.write("{}s +/- {}s".format(np.mean(run_time), np.std(run_time)))