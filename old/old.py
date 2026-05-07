i run this script:
import stable_retro
import numpy as np
from collections import deque

# 'ALL' ensures your 12-slot list matches the MultiBinary action space
env = stable_retro.make(
    game='StreetFighterIISpecialChampionEdition-Genesis-v0' ,players=1, use_restricted_actions=stable_retro.Actions.ALL
   
)

# MultiBinary action: [B, A, MODE, START, UP, DOWN, LEFT, RIGHT, C, Y, X, Z]
# Index 10 is  (Light Punch), Index 4 is 'UP'
ACTIONS = {"Block": [False,[[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]]],#left then right relative to player orientation
    "Crouch_Block": [False,[[0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]]], #left then right relative to player orientation
    "Crouch_Block_Right": [False,[0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]],
    "Walk_Forward": [False,[[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]]], #right then left, relative to player orientation
    "Walk_Left": [False,[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]],
    "Vertical_Jump": [False,[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]],
    "Jump_Forward": [False,[[0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],[0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0]]], #right then left relative to player orientation
    "Jump_Back": [False,[[0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],[0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]]], #left then right relative to player orientation
    "Jump_Right": [False,[0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]],
    "Jump_Left": [False,[0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0]],
    "Light_Punch": [False,[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]],
    "Medium_Punch": [False,[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]],
    "Heavy_Punch": [False,[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]],
    "Light_Kick": [False,[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]],
    "Medium_Kick": [False,[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
    "Heavy_Kick": [False,[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]],
    
    "Throw_Forward": [False,[[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]]],#left then right side input relative to player orientation
    "Throw_Back": [False,[[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1]]], #left then right side input relative to player orientation

    "Fireball": [True,[[[0,0,0,0,0,1,0,0,0,0,0,0],
		[0,0,0,0,0,1,0,1,0,0,0,0],
		[0,0,0,0,0,0,0,1,0,0,0,1]],[[0,0,0,0,0,1,0,0,0,0,0,0],
		[0,0,0,0,0,1,1,0,0,0,0,0],
		[0,0,0,0,0,0,1,0,0,0,0,1]]]], #left then right side relative to player orientation
  
    "Shoryuken": [True,[[[0,0,0,0,0,0,0,1,0,0,0,0],
		    [0,0,0,0,0,1,0,0,0,0,0,0],
		    [0,0,0,0,0,1,0,1,0,0,0,1]],[[0,0,0,0,0,0,1,0,0,0,0,0],
		    [0,0,0,0,0,1,0,0,0,0,0,0],
		    [0,0,0,0,0,1,1,0,0,0,0,1]]]], #left then right side input relative to player orientation
   
    "Tatsumaki": [True,[[[0,0,0,0,0,1,0,0,0,0,0,0],
		    [0,0,0,0,0,1,1,0,0,0,0,0],
		    [1,0,0,0,0,0,1,0,0,0,0,0]],[[0,0,0,0,0,1,0,0,0,0,0,0],
		    [0,0,0,0,0,1,0,1,0,0,0,0],
		    [1,0,0,0,0,0,0,1,0,0,0,0]]]], #left then right side input relative to player orientation
    "Neutral": [False,[0,0,0,0,0,0,0,0,0,0,0,0]]    
}
        #for tatsumaki,shoryuken, fireball and  throws, the first element represents whether or not the action is a macro, and the second element is a list of lists where the first element is the input for right side actions while the second element is for left side actions
        

# Reset returns (obs, info)
obs, info = env.reset()
done = False
i = 0
action_queue = deque()

for game in range(1):
     while not done and i < 1000:
          #select random action from actions dictionary
        if done:
            obs, info = env.reset()
        if action_queue: #if there are actions in the queue, pop the next one and act
            next_action = action_queue.popleft()
            obs, reward, done, info = env.step(next_action)
        else: #if no actions in the queue, select a new action
            action = random.choice(list(ACTIONS.values()))
            if action[0]: # If it's a macro, add all steps to the queue
                for step in action[1]:
                    action_queue.append(step)
            else: # If it's a single action, just add it to the queue
                action_queue.append(action[1])
            obs, reward, done, info = env.step(action_queue.popleft())
        i += 1

          

         
       
      
