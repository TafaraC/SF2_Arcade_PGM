import pandas as pd
import pickle
import os
#script learns best actions regardless of opponent 
def train_universal_brain(filename="sf2_experience_data_original.csv"):
    print(f"Loading experience data from {filename}...")
    
    if not os.path.exists(filename):
        print(f"ERROR: {filename} not found.")
        return

    # Load the dataset
    df = pd.read_csv(filename)

    # --- THE CORE CHANGE ---
    # We explicitly drop the 'opponent' column so the model 
    # generalizes across the entire roster.
    if 'opponent' in df.columns:
        df = df.drop(columns=['opponent'])
        print("Dropped 'opponent' column to focus on universal tactics.")

    # 1. Identify State Columns
    # Every remaining column except 'Action' and 'Utility' is a part of the "State"
    state_keys = [col for col in df.columns if col not in ['Action', 'Utility']]
    print(f"Features used for universal decision making: {state_keys}")

    # 2. Calculate Expected Utility
    # This averages the utility of actions across ALL opponents.
    # e.g., 'Fireball' at 'LARGE' range now has one average score for the whole game.
    print("Calculating universal average utilities...")
    utility_table = df.groupby(state_keys + ['Action'])['Utility'].mean().reset_index()

    # 3. Create the Optimal Policy
    # Find the best action for each unique combination of semantic variables
    print("Finding the globally optimal actions...")
    idx = utility_table.groupby(state_keys)['Utility'].idxmax()
    optimal_policy_df = utility_table.loc[idx]

    # 4. Convert to Dictionary (The Brain)
    policy = {}
    for _, row in optimal_policy_df.iterrows():
        state_tuple = tuple(row[k] for k in state_keys)
        policy[state_tuple] = {
            "best_action": row['Action'],
            "expected_utility": row['Utility']
        }

    # 5. Save the Brain
    # Note: We save the state_keys so the Trained_Agent knows NOT to look for opponent data
    brain_data = {
        "policy": policy,
        "state_keys": state_keys
    }

    with open("uni_ryu_brain.pkl", "wb") as f:
        pickle.dump(brain_data, f)

    print(f"\nSUCCESS: Universal Training Complete.")
    print(f"Unique situational states mapped: {len(policy)}")
    print("Ryu will now use the same fundamental logic against every fighter.")

if __name__ == "__main__":
    train_universal_brain()