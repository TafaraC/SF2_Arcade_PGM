import stable_retro
import pandas as pd
import random
from collections import deque 
env = stable_retro.make(game='StreetFighterIISpecialChampionEdition-Genesis-v0', use_restricted_actions=stable_retro.Actions.FILTERED, render_mode=None)
data = []
obs = env.reset()
done = False
i = 0


#[MK,LK,MODE,START,U,D,L,R,HK,MP,LP,HP]
   
'''[boolean, action] boolean determines if action is a macro or not
    If macro each element of the list must be fed into action queue, otherwise
    just act
    '''
Actions = {"Block_Left": [False,[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]],
    "Block_Right": [False,[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]],
    "Crouch_Block_Left": [False,[0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]],
    "Crouch_Block_Right": [False,[0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]],
    "Walk_Right": [False,[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]],
    "Walk_Left": [False,[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]],
    "Vertical_Jump": [False,[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]],
    "Jump_Right": [False,[0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]],
    "Jump_Left": [False,[0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0]],
    "Light_Punch": [False,[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]],
    "Medium_Punch": [False,[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]],
    "Heavy_Punch": [False,[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]],
    "Light_Kick": [False,[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]],
    "Medium_Kick": [False,[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
    "Heavy_Kick": [False,[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]],
    "Throw_Left": [False,[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]],
    "Throw_Right": [False,[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1]],
    "Fireball_Right": [True,[[0,0,0,0,0,1,0,0,0,0,0,0],
		[0,0,0,0,0,1,1,0,0,0,0,0],
		[0,0,0,0,0,0,1,0,0,0,0,1]]],
    "Fireball_Left": [True,[[0,0,0,0,0,1,0,0,0,0,0,0],
		[0,0,0,0,0,1,0,1,0,0,0,0],
		[0,0,0,0,0,0,0,1,0,0,0,1]]],
    "Shoryuken_Right": [True,[[0,0,0,0,0,0,1,0,0,0,0,0],
		    [0,0,0,0,0,1,0,0,0,0,0,0],
		    [0,0,0,0,0,1,1,0,0,0,0,1]]],
    "Shoryuken_Left": [True,[[0,0,0,0,0,0,0,1,0,0,0,0],
		    [0,0,0,0,0,1,0,0,0,0,0,0],
		    [0,0,0,0,0,1,0,1,0,0,0,1]]],
    "Tatsumaki_Right": [True,[[0,0,0,0,0,1,0,0,0,0,0,0],
		    [0,0,0,0,0,1,0,1,0,0,0,0],
		    [1,0,0,0,0,0,0,1,0,0,0,0]]],
    "Tatsumaki_Left": [True,[[0,0,0,0,0,1,0,0,0,0,0,0],
		    [0,0,0,0,0,1,1,0,0,0,0,0],
		    [1,0,0,0,0,0,1,0,0,0,0,0]]]
                
}
    
def select_actions(state_info):
    return random.choice(list(Actions.values()))

action_queue = deque()

for game in range(1):
    while not done:
        if done:
            obs=env.reset()
        #env.render()
        action = select_actions(obs)
        #print(action)
        if action[0]: # If it's a macro, add all steps to the queue
            for step in action[1]:
                action_queue.append(step)
        else: # If it's a single action, just add it to the queue
            action_queue.append(action[1])
        current_action = action_queue.popleft() if action_queue else [0]*12 # Default to no action if queue is empty
        #print(current_action)
        info = env.step(current_action)
        print(info[4])
        i+=1
        if(i>(2)):
            done =True
        

        
