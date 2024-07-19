import random
import numpy as np

class ReplayMemory:
    def __init__(self, capacity, seed, task_count, buffer_size=10):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.task_count = task_count
        self.buffer_size = buffer_size
        self.task_buffers = {i: [] for i in range(task_count)}
        self.previous_scores = {i: None for i in range(task_count)}
        self.done_task = []

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def push_task(self, task_id, score):
        if self.previous_scores[task_id] is not None:
            reward = score - self.previous_scores[task_id]
            self.task_buffers[task_id].append(reward)
            if len(self.task_buffers[task_id]) > self.buffer_size:
                self.task_buffers[task_id].pop(0)
        self.previous_scores[task_id] = score
    
    def sample_task(self):
        available_tasks = [task for task in range(self.task_count) if task not in self.done_task]
        
        if not available_tasks:
            return None, None  # 모든 작업이 완료된 경우

        sampled_rewards = {}
        for task_id in available_tasks:
            if len(self.task_buffers[task_id]) > 0:
                sampled_rewards[task_id] = random.choice(self.task_buffers[task_id])
            else:
                sampled_rewards[task_id] = 1  # 기본 보상

        # Choose task based on absolute value of sampled rewards
        if sampled_rewards:
                selected_task = max(sampled_rewards, key=lambda x: abs(sampled_rewards[x]))
                return selected_task, sampled_rewards[selected_task]
        else:
            return random.choice(available_tasks), 1

    def __len__(self):
        return len(self.buffer)