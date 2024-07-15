import random
import numpy as np

class ReplayMemory:
    def __init__(self, capacity, seed, task_count, buffer_size=10, epsilon=0.1):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0

        self.done_task = []
        self.task_rewards = {i: [] for i in range(task_count)}
        self.previous_scores = {i: None for i in range(task_count)}
        self.task_count = task_count
        self.buffer_size = buffer_size
        self.epsilon = epsilon

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def push_task(self, episode_score, task_id):
        if self.previous_scores[task_id] is not None:
            reward_change = episode_score - self.previous_scores[task_id]
            self.task_rewards[task_id].append(reward_change)
            if len(self.task_rewards[task_id]) > self.buffer_size:
                self.task_rewards[task_id].pop(0)
        self.previous_scores[task_id] = episode_score

    def sample_task(self):
        sampled_rewards = {}
        for task_id in range(self.task_count):
            if len(self.task_rewards[task_id]) > 0:
                sampled_rewards[task_id] = random.choice(self.task_rewards[task_id])
            else:
                sampled_rewards[task_id] = 1  # 기본 보상

        if sampled_rewards:
            selected_task = max(sampled_rewards, key=lambda x: abs(sampled_rewards[x]))
            
            # Create one-hot distribution
            one_hot_dist = [0] * self.task_count
            one_hot_dist[selected_task] = 1
            
            # Mix with uniform distribution
            final_dist = [(1 - self.epsilon) * p + self.epsilon / self.task_count for p in one_hot_dist]
            
            # Sample from the final distribution
            selected_task = random.choices(range(self.task_count), weights=final_dist)[0]
            
            return selected_task, sampled_rewards[selected_task]
        else:
            return random.choice(range(self.task_count)), 1

    def __len__(self):
        return len(self.buffer)