import numpy as np
import cv2
import os
import string
import random
import matplotlib.pyplot as plt


def animate(imgs):
    for idx, img in enumerate(imgs):
        cv2.imwrite(f"./game_res/{idx}.jpg", img)
    # length = len(imgs) // 4 + 1
    # plt.figure(figsize=(8*4, 16*length))
    # for idx, img in enumerate(imgs):
    #     plt.subplot(length, 4, idx+1)
    #     plt.imshow(img)
    # plt.show()

def interact(env, agents, steps):
    # reset our env
    obs = env.reset(seed=41)
    np.random.seed(0)
    imgs = []
    step = 0
    # Note that as the environment has two phases, we also keep track a value called 
    # `real_env_steps` in the environment state. The first phase ends once `real_env_steps` is 0 and used below

    # iterate until phase 1 ends
    print("prepare stage")
    while env.state.real_env_steps < 0:
        if step >= steps: break
        actions = {}
        for player in env.agents:
            o = obs[player]
            a = agents[player].early_setup(step, o)
            actions[player] = a
        step += 1
        obs, rewards, dones, infos = env.step(actions)
        print("step:", step)
        print("rewards:", rewards)
        print("dones:", dones)
        print("infos:", infos)
        print("actions:", actions)
        # print(rewards, dones, infos)
        imgs += [env.render("rgb_array", width=640, height=640)]
    done = False
    while not done:
        if step >= steps: break
        actions = {}
        for player in env.agents:
            o = obs[player]
            a = agents[player].act(step, o)
            actions[player] = a
        step += 1
        obs, rewards, dones, infos = env.step(actions)
        print("step:", step)
        print("rewards:", rewards)
        print("dones:", dones)
        print("infos:", infos)
        print("actions:", actions)
        imgs += [env.render("rgb_array", width=640, height=640)]
        done = dones["player_0"] and dones["player_1"]
    animate(imgs)
    # return animate(imgs)