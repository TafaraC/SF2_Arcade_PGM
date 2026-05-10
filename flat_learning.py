import pandas as pd
import pickle
import os

def train_flat_brain(filename="sf2_experience_data_original.csv"):
    print(f"Loading experience data from {filename}...")
    
    if not os.path.exists(filename):
        print(f"ERROR: {filename} not found. You need to collect data first!")
        return

    # Load the dataset
    df = pd.read_csv(filename)

    # 1. Identify State Columns
    # We treat every column except 'Action' and 'Utility' as part of the "State"
    state_keys = [col for col in df.columns if col not in ['Action', 'Utility']]
    print(f"Features used for decision making: {state_keys}")

    # 2. Calculate Expected Utility
    # We group by the state and the action to find the average reward for that move
    print("Calculating average utilities...")
    utility_table = df.groupby(state_keys + ['Action'])['Utility'].mean().reset_index()

    # 3. Create the Optimal Policy
    # For every unique state, we find the row with the highest Utility
    print("Finding the best actions for each state...")
    idx = utility_table.groupby(state_keys)['Utility'].idxmax()
    optimal_policy_df = utility_table.loc[idx]

    # 4. Convert to Dictionary (The Brain)
    # This allows the agent to look up a state tuple and get an action instantly
    policy = {}
    for _, row in optimal_policy_df.iterrows():
        state_tuple = tuple(row[k] for k in state_keys)
        policy[state_tuple] = {
            "best_action": row['Action'],
            "expected_utility": row['Utility']
        }

    # 5. Save the Brain
    brain_data = {
        "policy": policy,
        "state_keys": state_keys
    }

    with open("ryu_brain.pkl", "wb") as f:
        pickle.dump(brain_data, f)

    print(f"Success! Trained on {len(policy)} unique states.")
    print("Saved as ryu_brain.pkl")

if __name__ == "__main__":
    train_flat_brain()