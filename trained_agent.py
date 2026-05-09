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
        
        # Load the trained Influence Diagram policy (The "Brain")
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
            
        # Full Action Dictionary [MK, LK, MODE, START, U, D, L, R, HK, MP, LP, HP]
        self.ACTIONS = {
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
        """Translates raw memory into the chance nodes for the Influence Diagram."""
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

    def queue_action(self, action_name, player_on_right):
        """Translates the selected decision node into game inputs."""
        action_data = self.ACTIONS[action_name]
        is_macro = action_data[0]
        inputs = action_data[1]
        
        # Determine side for directional inputs
        side_idx = 1 if player_on_right else 0
        
        if is_macro:
            for frame_input in inputs[side_idx]:
                self.action_queue.append(frame_input)
        else:
            if isinstance(inputs[0], list): 
                self.action_queue.append(inputs[side_idx])
            else: 
                self.action_queue.append(inputs)

    def watch_agent_play(self, n_games=3):
        """Main loop letting the trained agent fight."""
        for game in range(n_games):
            print(f"\n--- Starting Arcade Run {game + 1} ---")
            obs = self.env.reset()
            info = self.env.step(self.ACTIONS["Neutral"][1])[4]
            
            self.is_active_play = False
            self.action_queue.clear()
            done = False
            
            while not done:
                p_x = info.get('play_x', 205)
                health = info.get('health', 176)
                enemy_health = info.get('enemy_health', 176)
                
                # --- ACTIVE PLAY LOGIC ---
                if health <= 0 or enemy_health <= 0:
                    self.is_active_play = False
                    self.action_queue.clear()
                elif p_x != 205 and not self.is_active_play:
                    self.is_active_play = True
                
                # --- DECISION SELECTION FROM INFLUENCE DIAGRAM ---
                if not self.is_active_play:
                    # Transition/Round Start Default
                    current_input = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0] 
                else:
                    semantic_state = self.get_semantic_state(info)
                    
                    if not semantic_state["active"] and len(self.action_queue) == 0:
                        
                        # 1. Format the current state exactly as the training keys expect
                        current_state_tuple = tuple(semantic_state[k] for k in self.state_keys)
                        
                        # 2. Look up the expected utilities
                        if current_state_tuple in self.policy:
                            # Select the optimal policy learned from the diagram
                            chosen_action = self.policy[current_state_tuple]["best_action"]
                        else:
                            # State not seen in training data. Default to safe play.
                            chosen_action = "Crouch_Block" 
                            
                        self.queue_action(chosen_action, semantic_state["player_on_right"])

                    # 3. Pull next frame of the selected move
                    if len(self.action_queue) > 0:
                        current_input = self.action_queue.popleft()
                    else:
                        current_input = self.ACTIONS["Neutral"][1]

                # Step Environment
                info = self.env.step(current_input)[4]
                
                # Update history for next frame's momentum calculation
                self.last_e_x = info.get('cpu_x', 307)
                self.last_p_x = info.get('play_x', 205)
                
                # Render limit (~60fps)
                time.sleep(1/120.0)

        self.env.close()

if __name__ == "__main__":
    trained_agent = Trained_SF2_Agent()
    trained_agent.watch_agent_play(n_games=3)