import pandas as pd
import pickle

def train_brain():
    filename = "sf2_experience_data_full_runs.csv"
    print(f"Loading data from {filename}...")
    
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"ERROR: {filename} not found. Run the data collection script first!")
        return

    # Dynamically grab all state columns (everything except Action and Utility)
    state_cols = [col for col in df.columns if col not in ['Action', 'Utility']]

    print("Calculating Expected Utilities...")
    # Group by all state variables + Action to find the mean utility for each (State, Action) pair
    utility_table = df.groupby(state_cols + ['Action'])['Utility'].mean().reset_index()

    print("Extracting optimal policy...")
    # Find the row index of the highest utility action for each unique state
    idx = utility_table.groupby(state_cols)['Utility'].idxmax()
    optimal_policy_df = utility_table.loc[idx]

    brain = {}
    
    for _, row in optimal_policy_df.iterrows():
        # Build the state tuple directly from the flat columns
        state_tuple = tuple(row[col] for col in state_cols)
        
        brain[state_tuple] = {
            "best_action": row['Action'],
            "expected_utility": row['Utility']
        }

    brain_data = {
        "policy": brain,
        "state_keys": state_cols
    }

    with open("ryu_brain.pkl", "wb") as f:
        pickle.dump(brain_data, f)

    print("\n" + "="*40)
    print(f"TRAINING COMPLETE!")
    print(f"Analyzed {len(df)} total experiences.")
    print(f"Learned optimal actions for {len(brain)} unique states.")
    print("Brain saved to ryu_brain.pkl!")
    print("="*40)

if __name__ == "__main__":
    train_brain()