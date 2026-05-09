import stable_retro
import pandas as pd
from collections import deque

class SF2_Experience_Builder:
    def __init__(self):
        self.env = stable_retro.make(game='StreetFighterIISpecialChampionEdition-Genesis-v0', use_restricted_actions=stable_retro.Actions.FILTERED)
        self.action_queue = deque() #used for macro inputs, if an action is a macro, its input sequence will be added to the queue and executed across frames, otherwise the action will just be executed and queue will remain empty
        self.history = [] #stores state, action, reward tuples for each step of an episode, used for training data generation
        self.is_active_play = False #whether or not the agent is currently in an active play (not in a loading screen or between rounds)
        self.last_hp = 176 #hp of player when action began used for reward calculation, initialized to max hp at start of episode
        self.last_enemy_hp = 176   #hp of opponent when action began used for reward calculation, initialized to max hp at start of episode
        self.last_p_x = 205
        self.last_e_x = 307
        self.current_action = Block
        self.active = False #whether agent is in the recovery of an action or not
        # Action definitions [MK, LK, MODE, START, U, D, L, R, HK, MP, LP, HP]
        self.ACTIONS={"Block": [False,[[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]]],#left then right relative to player orientation
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
        

    def get_semantic_state(self, info):
        """Discretizes raw RAM into ID nodes based on tested bounds [cite: 1, 3, 4]"""
        recovering = info.get("self_active",0)
        if recovering == 10 or recovering == 12: active =True
        else: active = False
        dist = info.get('distance', 187)
        p_x = info.get('play_x', 205)
        p_y = info.get('play_y', 20)
        e_x = info.get('cpu_x', 307)
        e_y = info.get('cpu_y', 20)
    
        
        # Attack Range Discretization 
        if dist < 20: range_state = "CLOSE"
        elif dist < 43: range_state = "MEDIUM"
        elif dist < 120: range_state = "LARGE"
        else: range_state = "FULLSCREEN"
        
        if info.get('active_enemy')==0: enemy_attacking = False
        else: enemy_attacking = True 
        
        if e_x<self.last_e_x: movement = "APPROACHING"
        elif e_x>self.last_e_x: movement = "RECEDING"
        else: movement = "STATIONARY"
        return {
            "range": range_state,
            "player_airborne": p_y >= 50,
            "opponent_airborne": e_y >=50, 
            "player_on_right": p_x > e_x, 
            "player_in_corner": (p_x < 100 and p_x<e_x) or (p_x>e_x and p_x > 410), # [cite: 4]
            "enemy_in_corner" : (e_x < 100 and e_x<p_x) or (e_x > 410 and e_x > p_x), # [cite: 4]
            "enemy_attacking": enemy_attacking, # [cite: 4]
            "enemy_movement": movement, 
            "active": active
        }

    #used after an action has been completed and player is no longer actively in an animation
    def calculate_utility(self, info):
        current_hp = info.get('health', 0)
        current_enemy_hp = info.get('enemy_health', 0)
        
        u_offense = self.last_enemy_hp - current_enemy_hp # Positive if enemy loses HP [cite: 6]
        u_penalty = current_hp - self.last_hp # Negative if player loses HP 
        
        # Defense Bonus: Opponent attacks but player takes no damage 
        u_defense = 20 if info.get('active_enemy') and u_penalty == 0 else 0
        
        self.last_hp, self.last_enemy_hp = current_hp, current_enemy_hp
        return u_offense + u_defense + u_penalty

   

# --- Data Generation ---
agent = SF2_Experience_Builder()
obs, info = agent.env.reset()


    
    