import torch
import numpy as np

from vectorenv.dummy import VectorEnv
from collector import Collector
from network import Network
from ppo import PPO

from test.test_envs.cartpole_continous import CartPoleContinousEnv
from gym.wrappers.time_limit import TimeLimit
import gym

import time
from tqdm import tqdm

C_avg_runs = 10
#C_avg_runs = 1

H_num_epochs = 10000
H_episodes_per_step = 1
H_test_episodes = 100
H_repeat = 2

def train_policy(seed):
    #construct envs
    def MakeEnv():
        return gym.make("Pendulum-v0")

    def IsStop(reward):
        return reward >= -250

    train_env = VectorEnv([MakeEnv for _ in range(16)])
    test_env  = VectorEnv([MakeEnv for _ in range(100)])

    #seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_env.seed(seed)
    test_env.seed(seed)

    #construct policy
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = Network((3), (1), 2.0, device).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
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
        #result = train_collector.collect(n_episode=H_episodes_per_step)
        result = train_collector.collect(n_step=4096)
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
    torch.save(net.state_dict(), "checkpoints/ppo_pendulum")

    #visualize result
    # render_env = VectorEnv([MakeEnv for _ in range(1)])
    # render_collector  = Collector(render_env, policy)
    # render_collector.collect(n_episode=1, render=True)

    return total_duration

#Average all runs
run_time = []
for run_id in tqdm(range(C_avg_runs)):
    torch.cuda.empty_cache()

    run_time.append(train_policy(run_id))

tqdm.write("{}s +/- {}s".format(np.mean(run_time), np.std(run_time)))