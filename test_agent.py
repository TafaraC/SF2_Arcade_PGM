import stable_retro
import pickle
from collections import deque
import time

class Trained_SF2_Agent:
    def __init__(self, brain_file="ryu_brain.pkl"):
        # render_mode="human" so we can watch Ryu fight
        self.env = stable_retro.make(
            game='StreetFighterIISpecialChampionEdition-Genesis-v0', 
            use_restricted_actions=stable_retro.Actions.FILTERED,
            render_mode="human" 
        )
        
        self.action_queue = deque()
        self.is_active_play = False
        self.last_hp = 176
        self.last_enemy_hp = 176
        self.last_p_x = 205
        self.last_e_x = 307
        
        # New Tracking Variables for Expanded State
        self.last_action_name = "Neutral"
        self.last_action_rewarded = False
        self.current_opponent_name = "guile"
        self.last_matches_won = 0
        self.last_enemy_matches_won = 0
        self.opponents = [
            "guile", "ken", "chun_li", "zangief", "dhalsim", "ryu", 
            "e_honda", "blanka", "balrog", "vega", "sagat", "m_bison"
        ]
        
        # Load the trained policy (The "Brain")
        print("Loading Ryu's Brain...")
        try:
            with open(brain_file, "rb") as f:
                brain_data = pickle.load(f)
                self.policy = brain_data["policy"]
                self.state_keys = brain_data["state_keys"]
            print(f"Successfully loaded brain with {len(self.policy)} known states.")
        except FileNotFoundError:
            print(f"ERROR: Could not find {brain_file}. Please run the training script first!")
            exit()
            
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
        """Translates raw memory into the exact format expected by the Brain."""
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
            "opponent": self.current_opponent_name,
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
        """Used to determine if the previous move was a good one."""
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

    def watch_agent_play(self, matches_per_opponent=1):
        """Loops through the requested opponents so Ryu fights the whole roster."""
        for opponent in self.opponents:
            self.current_opponent_name = opponent
            
            for match in range(matches_per_opponent):
                print(f"\n--- FIGHTING: {opponent.upper()} | Match {match + 1} ---")
                self.env.load_state(opponent)
                obs = self.env.reset()
                time.sleep(2)
                _, _, _, _, info = self.env.step(self.ACTIONS["Neutral"][1])
                
                self.last_matches_won = info.get('matches_won', 0)
                self.last_enemy_matches_won = info.get('enemy_matches_won', 0)
                
                self.is_active_play = False
                self.action_queue.clear()
                self.last_action_name = "Neutral"
                self.last_action_rewarded = False
                done = False
                
                while not done:
                    p_x = info.get('play_x', 205)
                    health = info.get('health', 176)
                    enemy_health = info.get('enemy_health', 176)
                    
                    matches_won = info.get('matches_won', 0)
                    enemy_matches_won = info.get('enemy_matches_won', 0)
                    continue_timer = info.get('continuetimer', 0)
                    
                    # Match win/loss condition
                    if matches_won == 2 or enemy_matches_won == 2 or continue_timer > 0 or done:
                        for i in range(480):
                            self.env.step(self.ACTIONS["Neutral"][1])
                        
                        
                        break 

                    # --- ACTIVE PLAY LOGIC ---
                    #round ends and win counter changes for either char
                    if matches_won != self.last_matches_won or enemy_matches_won != self.last_enemy_matches_won:
                        self.is_active_play = False
                        self.action_queue.clear()
                        self.last_matches_won = matches_won
                        self.last_enemy_matches_won = enemy_matches_won


                    elif p_x != 205 and not self.is_active_play:
                        self.is_active_play = True
                        self.last_hp = health
                        self.last_enemy_hp = enemy_health
                    
                    # --- DECISION SELECTION FROM INFLUENCE DIAGRAM ---
                    if not self.is_active_play:
                        current_input = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0] 
                    else:
                        semantic_state = self.get_semantic_state(info)
                        
                        if not semantic_state["active"] and len(self.action_queue) == 0:
                            
                            # 1. Update utility of the PREVIOUS action so we know if it was rewarded
                            utility = self.calculate_utility(info)
                            self.last_action_rewarded = bool(utility > 0)
                            
                            # Update the dictionary so it matches the current reality
                            semantic_state["prev_action_rewarded"] = self.last_action_rewarded
                            
                            # 2. Format state using keys saved in the brain to ensure perfect matching
                            current_state_tuple = tuple(semantic_state[k] for k in self.state_keys)
                            
                            # 3. Look up the expected utilities
                            if current_state_tuple in self.policy:
                                chosen_action = self.policy[current_state_tuple]["best_action"]
                            else:
                                # State not seen in training data. Default to safe blocking.
                                chosen_action = "Crouch_Block" 
                                
                            self.queue_action(chosen_action, semantic_state["player_on_right"])
                            self.last_action_name = chosen_action

                        # Pull next frame of the selected move
                        if len(self.action_queue) > 0:
                            current_input = self.action_queue.popleft()
                        else:
                            current_input = self.ACTIONS["Neutral"][1]

                    # Step Environment
                    _, _, terminated, truncated, info = self.env.step(current_input)
                    done = terminated or truncated
                    
                    self.last_e_x = info.get('cpu_x', 307)
                    self.last_p_x = info.get('play_x', 205)
                    
                    # Render limit (~60fps)
                    #time.sleep(1/480.0)

        self.env.close()

if __name__ == "__main__":
    trained_agent = Trained_SF2_Agent()
    trained_agent.watch_agent_play(matches_per_opponent=1)