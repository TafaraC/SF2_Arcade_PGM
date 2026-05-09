import pandas as pd
import pickle

def train_hierarchical_brain():
    filename = "sf2_experience_data_full_runs.csv"
    print(f"Loading data from {filename}...")
    
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"ERROR: {filename} not found.")
        return

    # 1. Define the hierarchy
    parent_node = 'opponent'
    # All other columns except 'Action' and 'Utility' are the children of 'opponent'
    semantic_vars = [col for col in df.columns if col not in [parent_node, 'Action', 'Utility']]
    
    # 2. Grouping by the full hierarchy to find the max expected utility
    # This effectively treats 'opponent' as the primary context for every state/action pair
    full_state_cols = [parent_node] + semantic_vars
    
    print("Analyzing action utilities across the hierarchy...")
    utility_table = df.groupby(full_state_cols + ['Action'])['Utility'].mean().reset_index()

    # 3. Extract the optimal policy
    # For every unique combination of (Opponent + Semantic States), find the best action
    idx = utility_table.groupby(full_state_cols)['Utility'].idxmax()
    optimal_policy_df = utility_table.loc[idx]

    brain = {}
    for _, row in optimal_policy_df.iterrows():
        # The key is the full state tuple (Opponent, Range, Airborne, etc.)
        state_tuple = tuple(row[col] for col in full_state_cols)
        
        brain[state_tuple] = {
            "best_action": row['Action'],
            "expected_utility": row['Utility']
        }

    # Save metadata so the agent knows the 'Opponent' comes first in the tuple
    brain_data = {
        "policy": brain,
        "state_keys": full_state_cols,
        "hierarchy": {
            "root": parent_node,
            "children": semantic_vars
        }
    }

    with open("ryu_brain.pkl", "wb") as f:
        pickle.dump(brain_data, f)

    print(f"Hierarchical Training Complete. States mapped: {len(brain)}")

if __name__ == "__main__":
    train_hierarchical_brain()