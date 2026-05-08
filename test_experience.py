import stable_retro
import pandas as pd
import random
from collections import deque
import time

class SF2_Experience_Builder:
    def __init__(self):
        # Disabled rendering and enabled RAM observations for maximum speed
        self.env = stable_retro.make(
            game='StreetFighterIISpecialChampionEdition-Genesis-v0', 
            use_restricted_actions=stable_retro.Actions.FILTERED,
            #render_mode=None,
            obs_type=stable_retro.Observations.RAM
        )
        self.action_queue = deque() 
        self.history = [] 
        self.is_active_play = False 
        
        # Tracking Variables
        self.last_hp = 176 
        self.last_enemy_hp = 176   
        self.last_p_x = 205
        self.last_e_x = 307
        
        # Memory and Matchup Tracking
        self.last_action_name = "Neutral"
        self.last_action_rewarded = False
        self.opponents = [
            "guile","ken","chun_li","zangief","dhalsim","ryu","e_honda","blanka","balrog","vega","sagat","m_bison"
        ]
        self.current_opponent_index = 0
        
        self.current_action_name = None
        self.current_state = None
        
        # Action definitions [MK, LK, MODE, START, U, D, L, R, HK, MP, LP, HP]
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
        
        self.ATTACK_ACTIONS = {
            "Light_Punch", "Medium_Punch", "Heavy_Punch", "Light_Kick", "Medium_Kick", 
            "Heavy_Kick", "Throw_Forward", "Throw_Back", "Fireball", "Shoryuken", 
            "Tatsumaki", "Down_LP", "Down_MP", "Down_HP", "Down_LK", "Down_MK", "Down_HK"
        }

    def get_semantic_state(self, info):
        recovering = info.get("self_active", 0)
        active = True if recovering in [10, 12] else False
        
        dist = info.get('distance', 187)
        p_x = info.get('play_x', 205)
        p_y = info.get('play_y', 20)
        e_x = info.get('cpu_x', 307)
        e_y = info.get('cpu_y', 20)
    
        if dist < 20: range_state = "CLOSE"
        elif dist < 43: range_state = "MEDIUM"
        elif dist < 120: range_state = "LARGE"
        else: range_state = "FULLSCREEN"
        
        enemy_attacking = bool(info.get('active_p2', 0) != 0) 
        
        if e_x < self.last_e_x: movement = "APPROACHING"
        elif e_x > self.last_e_x: movement = "RECEDING"
        else: movement = "STATIONARY"
        
        # Expanded semantic state with more information
        return {
            "opponent": self.opponents[self.current_opponent_index],
            "prev_action_rewarded": self.last_action_rewarded,
            "prev_action_was_attack": self.last_action_name in self.ATTACK_ACTIONS,
            "range": range_state,
            "player_airborne": p_y >= 50,
            "opponent_airborne": e_y >= 50, 
            "player_on_right": p_x > e_x, 
            "player_in_corner": (p_x < 100 and p_x < e_x) or (p_x > e_x and p_x > 410),
            "enemy_in_corner": (e_x < 100 and e_x < p_x) or (e_x > 410 and e_x > p_x),
            "enemy_attacking": enemy_attacking,
            "enemy_movement": movement, 
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

    def queue_action(self, action_name, player_on_right):
        action_data = self.ACTIONS[action_name]
        is_macro = action_data[0]
        inputs = action_data[1]
        side_idx = 1 if player_on_right else 0
        
        if is_macro:
            for frame_input in inputs[side_idx]:
                self.action_queue.append(frame_input)
        else:
            if isinstance(inputs[0], list): 
                self.action_queue.append(inputs[side_idx])
            else: 
                self.action_queue.append(inputs)

    def run_data_collection(self, runs_per_opponent=5, max_steps_per_run=1000000000):
        """Loops through the entire roster, fighting each opponent n times sequentially."""
        start_time = time.time()
        
        for opp_idx, opponent_name in enumerate(self.opponents):
            self.current_opponent_index = opp_idx
            
            for match in range(runs_per_opponent):
                print(f"\n======================================")
                print(f"FIGHTING: {opponent_name} | MATCH {match + 1} OF {runs_per_opponent}")
                print(f"======================================")
                
                # Load specific opponent state
                self.env.load_state(opponent_name)
                obs = self.env.reset()
                _, _, _, _, info = self.env.step(self.ACTIONS["Neutral"][1])
                
                self.is_active_play = False
                self.action_queue.clear()
                done = False
                
                self.last_action_name = "Neutral"
                self.last_action_rewarded = False
                
                for step in range(max_steps_per_run):
                    p_x = info.get('play_x', 205)
                    health = info.get('health', 176)
                    enemy_health = info.get('enemy_health', 176)
                    
                    matches_won = info.get('matches_won', 0)
                    enemy_matches_won = info.get('enemy_matches_won', 0)
                    continue_timer = info.get('continuetimer', 0)
                    
                    # Match condition logic: break if either side wins 2 rounds
                    if matches_won == 2 or enemy_matches_won == 2 or continue_timer > 0 or done:
                        break 

                    # --- ACTIVE PLAY LOGIC ---
                    if health <= 0 or enemy_health <= 0:
                        self.is_active_play = False
                        self.action_queue.clear()
                    elif p_x != 205 and not self.is_active_play:
                        self.is_active_play = True
                        self.last_hp = health
                        self.last_enemy_hp = enemy_health

                    # --- ACTION SELECTION & MEMORY ---
                    if not self.is_active_play:
                        current_input = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
                    else:
                        semantic_state = self.get_semantic_state(info)
                        
                        if not semantic_state["active"] and len(self.action_queue) == 0:
                            
                            # 1. UPDATE MEMORY FIRST
                            if self.current_state is not None:
                                utility = self.calculate_utility(info)
                                self.history.append((self.current_state, self.current_action_name, utility))
                                
                                self.last_action_rewarded = bool(utility > 0)
                                self.last_action_name = self.current_action_name
                            
                            # 2. SET NEW STATE
                            self.current_state = semantic_state
                            
                            # 3. SELECT AND QUEUE ACTION
                            self.current_action_name = random.choice(list(self.ACTIONS.keys()))
                            self.queue_action(self.current_action_name, semantic_state["player_on_right"])

                        if len(self.action_queue) > 0:
                            current_input = self.action_queue.popleft()
                        else:
                            current_input = self.ACTIONS["Neutral"][1]

                    # --- STEP ENVIRONMENT ---
                    _, _, terminated, truncated, info = self.env.step(current_input)
                    done = terminated or truncated
                    
                    self.last_e_x = info.get('cpu_x', 307)
                    self.last_p_x = info.get('play_x', 205)
               
        end_time = time.time()
        total_seconds = end_time - start_time
        minutes = int(total_seconds // 60)
        seconds = int(total_seconds % 60)
    
        print("\n" + "="*30)
        print(f"TRAINING COMPLETE!")
        total_runs = len(self.opponents) * runs_per_opponent
        print(f"Total time for {total_runs} matches: {minutes}m {seconds}s")
        print(f"Average time per match: {total_seconds / total_runs:.2f} seconds")
        print("="*30)

        self.env.close()
        self.save_data()

    def save_data(self):
        """Flattens the dictionary state and saves everything cleanly to a CSV."""
        if not self.history:
            print("No history collected. Skipping save.")
            return

        # Extract state keys dynamically
        state_keys = list(self.history[0][0].keys())
        
        # Flatten the history rows
        flat_history = []
        for state_dict, action, utility in self.history:
            row = [state_dict[k] for k in state_keys] + [action, utility]
            flat_history.append(row)

        columns = state_keys + ["Action", "Utility"]
        df = pd.DataFrame(flat_history, columns=columns)
        
        filename = "sf2_experience_data_full_runs.csv"
        df.to_csv(filename, index=False)
        print(f"\nAll runs completed. Collected {len(self.history)} total interactions.")
        print(f"Data cleanly flattened and saved to {filename}.")

if __name__ == "__main__":
    agent = SF2_Experience_Builder()
    agent.run_data_collection(runs_per_opponent=1) # Adjust as needed