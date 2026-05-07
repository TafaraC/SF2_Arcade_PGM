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
            render_mode=None,
            obs_type=stable_retro.Observations.RAM,
            state='sagat'
        )
        self.action_queue = deque() 
        self.history = [] 
        self.is_active_play = False 
        self.last_hp = 176 
        self.last_enemy_hp = 176   
        self.last_p_x = 205
        self.last_e_x = 307
        self.current_action_name = "Neutral"
        self.current_state = None
        self.active = False 
        
        # Action definitions [MK, LK, MODE, START, U, D, L, R, HK, MP, LP, HP]
        self.ACTIONS={
            "Block": [False,[[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]]],
            "Crouch_Block": [False,[[0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]]], 
            "Crouch_Block_Right": [False,[0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]],
            "Walk_Forward": [False,[[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]]], 
            "Walk_Left": [False,[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]],
            "Vertical_Jump": [False,[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]],
            "Jump_Forward": [False,[[0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],[0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0]]], 
            "Jump_Back": [False,[[0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],[0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]]], 
            "Jump_Right": [False,[0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]],
            "Jump_Left": [False,[0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0]],
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
            "Neutral": [False,[0,0,0,0,0,0,0,0,0,0,0,0]]    
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
        
        return {
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

    def run_arcade_mode(self, n_runs=5, max_steps_per_run=1000000000):
        """Repeats a full arcade run n times, resetting on win (12 enemies) or loss (2 round drops)."""
        done = False
        
        start_time = time.time()
        
        for run in range(n_runs):
            print(f"\n======================================")
            print(f"STARTING ARCADE RUN {run + 1} OF {n_runs}")
            print(f"======================================")
            
            obs = self.env.reset()
            # Unpack 5 elements (Gymnasium API)
            _, _, _, _, info = self.env.step(self.ACTIONS["Neutral"][1])
            
            self.is_active_play = False
            self.action_queue.clear()
            
            opponents_defeated = 0
            last_matches_won = 0
            
            for step in range(max_steps_per_run):
                p_x = info.get('play_x', 205)
                health = info.get('health', 176)
                enemy_health = info.get('enemy_health', 176)
                
                current_matches_won = info.get('matches_won', 0)
                current_enemy_matches_won = info.get('enemy_matches_won', 0)
                continue_timer = info.get('continuetimer', 0)
                
                # --- ARCADE RUN WIN/LOSS LOGIC ---
                if current_matches_won > last_matches_won:
                    if current_matches_won == 2:
                        opponents_defeated += 1
                        print(f"--> Opponent Defeated! Total so far: {opponents_defeated} / 12")
                
                last_matches_won = current_matches_won
                
                if opponents_defeated >= 12:
                    print(f"*** ARCADE RUN {run + 1} SUCCESS! Ryu beat all 12 opponents! ***")
                    break 
                
                if current_enemy_matches_won == 2 or continue_timer > 0 or done:
                    print(f"*** ARCADE RUN {run + 1} FAILED. Ryu lost to opponent {opponents_defeated + 1}. ***")
                    break 

                # --- ACTIVE PLAY LOGIC ---
                if health <= 0 or enemy_health <= 0:
                    self.is_active_play = False
                    self.action_queue.clear()
                elif p_x != 205 and not self.is_active_play:
                    self.is_active_play = True
                    self.last_hp = health
                    self.last_enemy_hp = enemy_health

                # --- ACTION SELECTION ---
                if not self.is_active_play:
                    current_input = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
                else:
                    semantic_state = self.get_semantic_state(info)
                    
                    if not semantic_state["active"] and len(self.action_queue) == 0:
                        if self.current_state is not None:
                            utility = self.calculate_utility(info)
                            self.history.append((self.current_state, self.current_action_name, utility))
                        
                        self.current_state = semantic_state
                        self.current_action_name = random.choice(list(self.ACTIONS.keys()))
                        self.queue_action(self.current_action_name, semantic_state["player_on_right"])

                    if len(self.action_queue) > 0:
                        current_input = self.action_queue.popleft()
                    else:
                        current_input = self.ACTIONS["Neutral"][1]

                # --- STEP ENVIRONMENT ---
                # Unpack all 5 elements from the Gymnasium step()
                _, _, terminated, truncated, info = self.env.step(current_input)
                done = terminated or truncated
                
                # Update memory tracking
                self.last_e_x = info.get('cpu_x', 307)
                self.last_p_x = info.get('play_x', 205)
           
        end_time = time.time()
        total_seconds = end_time - start_time
        minutes = int(total_seconds // 60)
        seconds = int(total_seconds % 60)
    
        print("\n" + "="*30)
        print(f"TRAINING COMPLETE!")
        print(f"Total time for {n_runs} runs: {minutes}m {seconds}s")
        print(f"Average time per run: {total_seconds / n_runs:.2f} seconds")
        print("="*30)

        self.env.close()
        
        # Save aggregated history across all runs
        df = pd.DataFrame(self.history, columns=["State", "Action", "Utility"])
        df.to_csv("sf2_experience_data_full_runs.csv", index=False)
        print(f"\nAll runs completed. Collected {len(self.history)} total interactions.")

# --- Data Generation ---
if __name__ == "__main__":
    agent = SF2_Experience_Builder()
    agent.run_arcade_mode(n_runs=50)