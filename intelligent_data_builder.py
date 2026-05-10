import stable_retro
import pandas as pd
import random
import pickle
from collections import deque
import time
import os

class SF2_Boss_Refiner:
    def __init__(self, brain_file="uni_ryu_brain.pkl"):
        self.env = stable_retro.make(
            game='StreetFighterIISpecialChampionEdition-Genesis-v0', 
            use_restricted_actions=stable_retro.Actions.FILTERED,
            render_mode=None,
            obs_type=stable_retro.Observations.RAM
        )
        self.action_queue = deque() 
        self.history = [] 
        self.is_active_play = False 
        
        # Stats tracking
        self.last_hp = 176 
        self.last_enemy_hp = 176   
        self.last_p_x = 205
        self.last_e_x = 307
        
        # Round tracking
        self.last_matches_won = 0
        self.last_enemy_matches_won = 0

        # Exploration Toggles
        self.selection_counter = 0  # To track the 2 vs 2 rhythm
        
        # TARGET THE Losing enemies 
        self.opponents = ["ryu","e_honda", "sagat", "m_bison"]
        self.current_opponent_index = 0
        
        # Load the Brain for the "Smart" moves
        print(f"Loading {brain_file} for tactical guidance...")
        try:
            with open(brain_file, "rb") as f:
                brain_data = pickle.load(f)
                self.policy = brain_data["policy"]
                self.state_keys = brain_data["state_keys"]
        except FileNotFoundError:
            print("ERROR: No brain file found. Please train first.")
            exit()

        # Action Definitions (Full list from your original)
        self.ACTIONS = {
            "Block": [False,[[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]]],
            "Crouch_Block": [False,[[0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]]], 
            "Walk_Forward": [False,[[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]]], 
            "Walk_Left": [False,[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]],
            "Vertical_Jump": [False,[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]],
            "Jump_Forward": [False,[[0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],[0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0]]], 
            "Jump_Back": [False,[[0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],[0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]]], 
            "Light_Punch": [False,[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]],
            "Medium_Punch": [False,[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]],
            "Heavy_Punch": [False,[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]],
            "Light_Kick": [False,[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]],
            "Medium_Kick": [False,[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
            "Heavy_Kick": [False,[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]],
            "Throw_Forward": [False,[[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]]],
            "Throw_Back": [False,[[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1]]], 
            "Fireball": [True,[[[0,0,0,0,0,1,0,0,0,0,0,0], [0,0,0,0,0,1,0,1,0,0,0,0], [0,0,0,0,0,0,0,1,0,0,0,1]],
                               [[0,0,0,0,0,1,0,0,0,0,0,0], [0,0,0,0,0,1,1,0,0,0,0,0], [0,0,0,0,0,0,1,0,0,0,0,1]]]], 
            "Shoryuken": [True,[[[0,0,0,0,0,0,0,1,0,0,0,0], [0,0,0,0,0,1,0,0,0,0,0,0], [0,0,0,0,0,1,0,1,0,0,0,1]],
                                [[0,0,0,0,0,0,1,0,0,0,0,0], [0,0,0,0,0,1,0,0,0,0,0,0], [0,0,0,0,0,1,1,0,0,0,0,1]]]], 
            "Tatsumaki": [True,[[[0,0,0,0,0,1,0,0,0,0,0,0], [0,0,0,0,0,1,1,0,0,0,0,0], [1,0,0,0,0,0,1,0,0,0,0,0]],
                                [[0,0,0,0,0,1,0,0,0,0,0,0], [0,0,0,0,0,1,0,1,0,0,0,0], [1,0,0,0,0,0,0,1,0,0,0,0]]]], 
            "Neutral": [False,[0,0,0,0,0,0,0,0,0,0,0,0]],
            "Down_LP": [False,[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]],
            "Down_MP": [False,[0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0]],
            "Down_HP": [False,[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]],
            "Down_LK": [False,[0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]],
            "Down_MK": [False,[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]],
            "Down_HK": [False,[0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0]]     
        }
        self.ATTACK_ACTIONS = set([k for k in self.ACTIONS.keys() if "Punch" in k or "Kick" in k or "Throw" in k or k in ["Fireball", "Shoryuken", "Tatsumaki"]])

    def get_semantic_state(self, info):
        # ... (Same semantic logic as your original script) ...
        recovering = info.get("self_active", 0)
        active = True if recovering in [10, 12] else False
        dist = info.get('distance', 187)
        p_x, p_y = info.get('play_x', 205), info.get('play_y', 20)
        e_x, e_y = info.get('cpu_x', 307), info.get('cpu_y', 20)
        
        range_state = "FULLSCREEN"
        if dist < 20: range_state = "CLOSE"
        elif dist < 43: range_state = "MEDIUM"
        elif dist < 120: range_state = "LARGE"
        
        return {
            "opponent": self.opponents[self.current_opponent_index],
            "prev_action_rewarded": False, # Updated in loop
            "prev_action_was_attack": False, # Updated in loop
            "range": range_state,
            "player_airborne": p_y >= 50,
            "opponent_airborne": e_y >= 50, 
            "player_on_right": p_x > e_x, 
            "player_in_corner": (p_x < 100 and p_x < e_x) or (p_x > e_x and p_x > 410),
            "enemy_in_corner": (e_x < 100 and e_x < p_x) or (e_x > 410 and e_x > p_x),
            "enemy_attacking": bool(info.get('active_p2', 0) != 0),
            "enemy_movement": "APPROACHING" if e_x < self.last_e_x else "RECEDING" if e_x > self.last_e_x else "STATIONARY", 
            "active": active
        }

    def calculate_utility(self, info):
        current_hp = info.get('health', 0)
        current_enemy_hp = info.get('enemy_health', 0)
        u_offense = self.last_enemy_hp - current_enemy_hp 
        u_penalty = current_hp - self.last_hp 
        u_defense = 20 if info.get('active_p2') and u_penalty == 0 else 0
        self.last_hp, self.last_enemy_hp = current_hp, current_enemy_hp
        return u_offense + u_defense + u_penalty

    def run_data_collection(self, runs_per_opponent=500):
        for opp_idx, opponent_name in enumerate(self.opponents):
            self.current_opponent_index = opp_idx
            
            for match in range(runs_per_opponent):
                print(f"REFINING: {opponent_name.upper()} | MATCH {match + 1}/{runs_per_opponent}")
                self.env.load_state(opponent_name)
                self.env.reset()
                _, _, _, _, info = self.env.step(self.ACTIONS["Neutral"][1])
                
                self.is_active_play = False
                self.action_queue.clear()
                current_state_record = None
                current_action_name = None
                
                # Reset score tracking
                self.last_matches_won = info.get('matches_won', 0)
                self.last_enemy_matches_won = info.get('enemy_matches_won', 0)
                
                done = False
                while not done:
                    p_x = info.get('play_x', 205)
                    m_won = info.get('matches_won', 0)
                    e_won = info.get('enemy_matches_won', 0)
                    
                    # Round Reset Logic
                    if m_won != self.last_matches_won or e_won != self.last_enemy_matches_won:
                        self.is_active_play = False
                        self.action_queue.clear()
                        self.last_matches_won, self.last_enemy_matches_won = m_won, e_won
                    elif p_x != 205 and not self.is_active_play:
                        self.is_active_play = True
                        self.last_hp = info.get('health', 176)
                        self.last_enemy_hp = info.get('enemy_health', 176)

                    if not self.is_active_play:
                        current_input = [0] * 12
                    else:
                        semantic_state = self.get_semantic_state(info)
                        
                        if not semantic_state["active"] and len(self.action_queue) == 0:
                            # 1. Record Outcome of previous action
                            if current_state_record is not None:
                                utility = self.calculate_utility(info)
                                self.history.append((current_state_record, current_action_name, utility))
                                semantic_state["prev_action_rewarded"] = bool(utility > 0)
                                semantic_state["prev_action_was_attack"] = current_action_name in self.ATTACK_ACTIONS

                            # 2. SELECT NEXT ACTION (The 2 vs 2 Logic)
                            if self.selection_counter < 2:
                                # TACTICAL CHOICE (PGM)
                                state_tuple = tuple(semantic_state[k] for k in self.state_keys)
                                if state_tuple in self.policy:
                                    current_action_name = self.policy[state_tuple]["best_action"]
                                else:
                                    current_action_name = "Crouch_Block" # Fallback if state is new
                            else:
                                # RANDOM CHOICE (Exploration)
                                current_action_name = random.choice(list(self.ACTIONS.keys()))
                            
                            # Increment counter and reset after 4 moves
                            self.selection_counter = (self.selection_counter + 1) % 4
                            
                            current_state_record = semantic_state
                            
                            # Queue it up
                            action_data = self.ACTIONS[current_action_name]
                            side = 1 if semantic_state["player_on_right"] else 0
                            if action_data[0]: # Macro
                                for f in action_data[1][side]: self.action_queue.append(f)
                            else: # Simple
                                inputs = action_data[1]
                                self.action_queue.append(inputs[side] if isinstance(inputs[0], list) else inputs)

                        current_input = self.action_queue.popleft() if self.action_queue else self.ACTIONS["Neutral"][1]

                    _, _, terminated, truncated, info = self.env.step(current_input)
                    done = terminated or truncated or m_won == 2 or e_won == 2
                    self.last_e_x = info.get('cpu_x', 307)

            # SAVE AFTER EACH CHARACTER (Append mode is inside save_data)
            self.save_data()

        self.env.close()

    def save_data(self):
        if not self.history: return
        filename = "sf2_experience_data_full_runs.csv"
        
        state_keys = list(self.history[0][0].keys())
        df = pd.DataFrame([ [s[k] for k in state_keys] + [a, u] for s, a, u in self.history ], 
                          columns=state_keys + ["Action", "Utility"])
        
        file_exists = os.path.isfile(filename)
        df.to_csv(filename, mode='a', index=False, header=not file_exists)
        print(f"Appended {len(self.history)} rows for current boss.")
        self.history = []

if __name__ == "__main__":
    builder = SF2_Boss_Refiner()
    builder.run_data_collection(runs_per_opponent=1000) # Targeted refinement