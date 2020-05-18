import torch
import numpy as np

from batch import Batch
#import time

class Collector:
    def __init__(self, env, policy):
        self.env = env
        self.env_num = len(env)

        self.policy = policy

        self.clear_buffer()
        self.reset_env()

    def clear_buffer(self):
        self.buffers = [Batch(
            state=[],
            action=[],
            reward=[],
            next_state=[],
            done=[]
        ) for _ in range(self.env_num)]

    def reset_env(self):
        #reset env
        self._obs = self.env.reset()
        self._act = self._rew = self._done = self._info = []

        self.reward = np.zeros(self.env_num)
        self.length = np.zeros(self.env_num)

    def collect(self, n_step=0, n_episode=0, render=False):
        assert n_step or n_episode, "Must specify maximum steps or maximum epochs (or both)"

        rewards = []
        lengths = []

        total_steps = 0
        total_episodes = 0

        while True:
            #infer policy in all environments
            with torch.no_grad():
                actions = self.policy(self._obs)

            #step all environments
            obs_cur = self._obs
            self._obs, self._rew, self._done, self._info = self.env.step(actions)

            total_steps += self.env_num

            #increase env statistics
            for i in range(self.env_num):
                self.reward[i] += self._rew[i]
                self.length[i] += 1

            #push all buffers
            for i in range(self.env_num):
                self.buffers[i].state.append(obs_cur[i])
                self.buffers[i].action.append(actions[i])
                self.buffers[i].reward.append(self._rew[i])
                self.buffers[i].next_state.append(self._obs[i])
                self.buffers[i].done.append(self._done[i])

            #reset environments that are done
            done_ids = []
            for i in range(self.env_num):
                if self._done[i]:
                    done_ids.append(i)

                    #record reward and length
                    total_episodes += 1
                    rewards.append(self.reward[i])
                    lengths.append(self.length[i])

                    self.reward[i], self.length[i] = 0, 0

            if done_ids:
                self._obs = self.env.reset(done_ids)

            #render envs
            if render:
                self.env.render()
                #time.sleep(0.01)

            #check return condition
            if (n_step and total_steps >= n_step) or (n_episode and total_episodes >= n_episode):
                break

        return {
            'n/ep': total_episodes,
            'n/st': total_steps,
            'rew': np.mean(rewards),
            'len': np.mean(lengths),
        }

    #Concatenate all buffers to one large batch
    def get_experience(self, insert_dummy=True):
        batch = Batch(dummy=[])

        for i in range(self.env_num):
            #append to total batch
            batch.append(self.buffers[i])

            #set dummy key
            if insert_dummy:
                length = len(self.buffers[i])
                
                batch.dummy += [False] * length

                #append dummy item at the end if not done
                if length:
                    if not batch[-1].done:
                        dummy_item = batch[-1:]

                        dummy_item.dummy = [True]
                        dummy_item.done = [True]
                        dummy_item.state = dummy_item.next_state

                        batch.append(dummy_item)

        return batch.to_numpy()

    # Environment Related
    def seed(self, seed=None):
        """Reset all the seed(s) of the given environment(s)."""
        if hasattr(self.env, 'seed'):
            return self.env.seed(seed)

    def render(self, **kwargs):
        """Render all the environment(s)."""
        if hasattr(self.env, 'render'):
            return self.env.render(**kwargs)

    def close(self):
        """Close the environment(s)."""
        if hasattr(self.env, 'close'):
            self.env.close()