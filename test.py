#!/usr/bin/env python
from __future__ import print_function

import sys, gym, time, pickle, torch
import matplotlib.pyplot as plt
plt.style.use("ggplot")

#
# Test yourself as a learning agent! Pass environment name as a command-line argument, for example:
# 
# python3 test.py CartPole-v1
#
env = gym.make('LunarLander-v2' if len(sys.argv)<2 else sys.argv[1])

if not hasattr(env.action_space, 'n'):
    raise Exception('Keyboard agent only supports discrete action spaces')
ACTIONS = env.action_space.n
SKIP_CONTROL = 0    # Use previous control decision SKIP_CONTROL times, that's how you
                    # can test what skip is still usable.

human_agent_action = 0
human_wants_restart = False
human_sets_pause = False
breakPoint = False
episode_num = 1

def key_press(key, mod):
    global human_agent_action, human_wants_restart, human_sets_pause
    #Press enter : restart, Press space : pause
    if key==0xff0d: human_wants_restart = True
    if key==32: human_sets_pause = not human_sets_pause
    a = int( key - ord('0') )
    if a <= 0 or a >= ACTIONS: return
    human_agent_action = a

def key_release(key, mod):
    global human_agent_action
    a = int( key - ord('0') )
    if a <= 0 or a >= ACTIONS: return
    if human_agent_action == a:
        human_agent_action = 0

env.render()
env.unwrapped.viewer.window.on_key_press = key_press
env.unwrapped.viewer.window.on_key_release = key_release
graph =  {
            'episode_num': [],
            'reward': [],
            'state' : [],
            }
def rollout(env):
    #rollout : 자료르 뽑아내다.
    global human_agent_action, human_wants_restart, human_sets_pause, episode_num, breakPoint
    human_wants_restart = False
    max_episode = 20
    total_reward = 0
    total_timesteps = 0
    sim_iter = 50
    for i in range(max_episode):
        print("++"*30)
        print("Episode " + str(i+1) + " learner test start")
        print("++"*30)
        graph['episode_num'].append(i+1)
        #load model
        PATH = 'model/learner_'+str(i+1)
        model = torch.load(PATH)
        reward = 0
        for j in range(sim_iter):
            obser = env.reset()
            skip = 0
            total_reward = 0
            total_timesteps = 0
            while 1:
                if not skip:
                    # print("taking action {}".format(human_agent_action))
                    # print("state {}".format(obser))
                    obser = obser[None,...]
                    te_obser = torch.from_numpy(obser).float()
                    
                    learner_action, _ = model.predict(te_obser)
                    learner_action = int(learner_action)
                    
                    a = learner_action
                    total_timesteps += 1
                    skip = SKIP_CONTROL
                else:
                    skip -= 1

                obser, r, done, info = env.step(a)
                # results['reward'].append(r)
                # if r != 0:
                    # print("reward %0.3f" % r)
                total_reward += r
                window_still_open = env.render()
                if window_still_open==False: return False
                if done: break
                if human_wants_restart: break
                while human_sets_pause:
                    env.render()
                    time.sleep(0.1)
                time.sleep(0.1)
            print("\t ["+ str(j+1) +"] times : timesteps %i reward %0.2f" % (total_timesteps, total_reward))
            reward += total_reward
        reward = reward/sim_iter
        print("[Finish!!] Episode %i mean of reward : %0.2f" % (i+1, reward))
        graph['reward'].append(reward)
    X = graph['episode_num']
    Y = graph['reward']
    plt.xlabel("episode_num")
    plt.ylabel("reward")
    plt.plot(X, Y, "*")
    line = plt.plot(X, Y)
    plt.ylim(0.0, 600)
    breakPoint = True
    plt.show()


print("ACTIONS={}".format(ACTIONS))
print("Press keys 1 2 3 ... to take actions 1 2 3 ...")
print("No keys pressed is taking action 0")

while 1:
    window_still_open = rollout(env)
    if breakPoint : break
    if window_still_open==False: break