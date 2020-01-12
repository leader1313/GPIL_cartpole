#!/usr/bin/env python
from __future__ import print_function

import sys, gym, time, pickle

#
# Test yourself as a learning agent! Pass environment name as a command-line argument, for example:
# 
# python3 keyboard_agent.py CartPole-v1
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
results =  {'pos': [],
            'vel': [],
            'angle': [],
            'a_vel': [],
            'action': [],
            'reward': [],
            'state' : [],
            }
initialize = {}
max_episode = 20
def rollout(env):
    #rollout : 자료르 뽑아내다.
    global human_agent_action, human_wants_restart, human_sets_pause, episode_num, max_episode
    human_wants_restart = False
    obser = env.reset()
    skip = 0
    total_reward = 0
    total_timesteps = 0
    
    while 1:
        if not skip:
            # print("taking action {}".format(human_agent_action))
            # print("state {}".format(obser))
            results['pos'].append(obser[0])
            results['vel'].append(obser[1])
            results['angle'].append(obser[2])
            results['a_vel'].append(obser[3])
            results['state'].append(obser)
            results['action'].append(human_agent_action)
            a = human_agent_action
            total_timesteps += 1
            skip = SKIP_CONTROL
        else:
            skip -= 1

        obser, r, done, info = env.step(a)
        results['reward'].append(r)
        # if r != 0:
            # print("reward %0.3f" % r)
        total_reward += r
        window_still_open = env.render()
        if window_still_open==False: return False
        if total_reward > (10*episode_num) :
                with open('data/supervisor_demo'+str(episode_num)+'.pickle', 'wb') as handle:
                    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
                episode_num += 1
            
        if total_reward > (10*max_episode) : done = True
        if done: 
            if episode_num < max_episode :
                episode_num = 1
                #results initialization
            for i in results :
                results[i] = []
            break
        if human_wants_restart: break
        while human_sets_pause:
            env.render()
            time.sleep(0.1)
        time.sleep(0.1)
    print("[]timesteps %i reward %0.2f" % (total_timesteps, total_reward))

print("ACTIONS={}".format(ACTIONS))
print("Press keys 1 2 3 ... to take actions 1 2 3 ...")
print("No keys pressed is taking action 0")

while 1:
    window_still_open = rollout(env)
    # if episode_num >= max_episode : break
    if window_still_open==False: break