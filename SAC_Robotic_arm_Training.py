# SAC_Robotic_arm_Training.py

import datetime
import numpy as np
import itertools
import torch
from sac import SAC
from replay_memory import ReplayMemory
import Config
import Env
import time
import os
import sys
import gc
import csv
import wandb

wandb.init(project='Refer3')
wandb.run.name = 'SAC_Robotic_Arm_Refer3'
wandb.run.save()

def Run_Training():
    gc.enable()
    env = Env.Ned2_control()
    env.reset()

    agent = SAC(12, 3, Config)

    memory = ReplayMemory(Config.replay_size, Config.seed, task_count=5)  # 작업 수를 5로 설정

    total_numsteps = 0
    updates = 0
    episode_success = []

    now = datetime.datetime.now()
    folderName = ('{}_SAC_{}_{}_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), Config.Current_Data_Selection_Ratio, Config.lr,
                                                                Config.Clustering_K, Config.Is_Clearing_Memory, Config.Success_Standard))

    success_rate = 0.0
    success_rate_list =[]

    Is_Success_each_level = [[] for _ in range(Config.Clustering_K)]
    success_rate_each_level = [[] for _ in range(Config.Clustering_K)]

    for i_episode in itertools.count(1):
        episode_reward = 0
        episode_steps = 0
        done = False

        # 샘플링 알고리즘을 사용하여 다음 작업 선택
        task_id, sampled_reward = memory.sample_task()

        env.set_task(task_id)  # 환경에 선택된 작업 설정
        env.reset()
        state = env.get_state()

        while not done:
            if Config.start_steps > total_numsteps:
                action = np.random.uniform(-1, 1, size=3).tolist()
            else:
                action = agent.select_action(state)

            env.action(action)
            time.sleep(Config.time_sleep_interval)
            next_state, reward, done, success = env.observation()
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward

            mask = 1 if episode_steps == Config.max_episode_steps else float(not done)
            memory.push(state, action, reward, next_state, mask)  # 선택된 작업 ID 포함

            state = next_state

            if len(memory) > Config.batch_size:
                for i in range(Config.updates_per_step):
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, Config.batch_size, updates)
                    updates += 1

            if done:
                episode_success.append(success)

                success_rate = np.sum(episode_success[-min(10, len(episode_success)):])/10.0
                success_rate_list.append(success_rate)

                filePath = './models/' + folderName

                if not os.path.isdir(filePath):
                    os.makedirs(filePath)

                torch.save({
                    'model': agent.policy.state_dict(),
                    'optimizer': agent.policy_optim.state_dict()
                }, "./models/" + folderName + "/model_"+str(i_episode)+".tar")


        Is_Success_each_level[env.Level_Of_Point].append(success)
        wandb.log({f'IsSuccess/Is_Success_{task_id}':success}, step=i_episode)

        for i in range(Config.Clustering_K):
            if (len(Is_Success_each_level[i]) > 0)  and (task_id == i):
                level_sr = np.sum(Is_Success_each_level[i][-min(10, len(Is_Success_each_level[i])):])/10.0
                success_rate_each_level[i].append(level_sr)
                wandb.log({f'SuccessRate/Success_rate_{i}':level_sr}, step=i_episode)

                moving_avg_sr = np.sum(success_rate_each_level[i][-min(5, len(success_rate_list)):])/5.0

                if moving_avg_sr >= 0.9:
                    memory.done_task.append(i)

        if total_numsteps > Config.num_steps:
            break

        memory.push_task(episode_reward, task_id)


        wandb.log({'Success_rate':success_rate}, step=i_episode)
        wandb.log({'Score':episode_reward}, step=i_episode)

        print("Current_Level: {}, Episode: {}, total numsteps: {}, episode steps: {}, reward: {}, Replay Memory length: {}".format(env.Level_Of_Point, i_episode, total_numsteps, episode_steps, round(episode_reward, 2), len(memory)))
        gc.collect()

        try:
            wandb.log({'critic_1_loss':critic_1_loss}, step=i_episode)
            wandb.log({'alpha':alpha}, step=i_episode)
        except:
            pass

        if(len(memory.done_task) == Config.Clustering_K):
            break


if __name__ == "__main__":
    Run_Training()
