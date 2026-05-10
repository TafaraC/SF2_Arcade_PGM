import pandas as pd
import ast
import pickle

def train_influence_diagram(csv_path):
    print("Loading data...")
    df = pd.read_csv(csv_path)
    
    # 1. Unpack the stringified dictionary in the 'State' column into separate columns
    print("Parsing states...")
    # ast.literal_eval safely turns the string "{'range': 'CLOSE', ...}" back into a dict
    df['State'] = df['State'].apply(ast.literal_eval)
    state_df = pd.json_normalize(df['State'])
    
    # Combine the unpacked states with the Action and Utility columns
    df = pd.concat([state_df, df[['Action', 'Utility']]], axis=1)
    
    # Drop the 'active' column as it's always False when a decision is made
    if 'active' in df.columns:
        df = df.drop(columns=['active'])
        
    # Get a list of the actual chance nodes (state variables)
    state_columns = [col for col in df.columns if col not in ['Action', 'Utility']]
    
    # 2. Calculate Expected Utility (Average Utility) for every State-Action pair
    print("Calculating Expected Utilities...")
    # We group by all state variables PLUS the action, and calculate the mean of the Utility
    expected_utility_table = df.groupby(state_columns + ['Action'])['Utility'].mean().reset_index()
    
    # 3. Extract the Optimal Policy (The Action with the Max Utility for each State)
    print("Extracting optimal policy...")
    # Sort by Utility descending, then drop duplicate states keeping the top one
    optimal_policy_df = expected_utility_table.sort_values('Utility', ascending=False).drop_duplicates(subset=state_columns, keep='first')
    
    # 4. Convert to a dictionary for extremely fast lookups during gameplay
    policy_dict = {}
    for _, row in optimal_policy_df.iterrows():
        # Create a tuple of the state values to act as the dictionary key
        state_tuple = tuple(row[col] for col in state_columns)
        policy_dict[state_tuple] = {
            "best_action": row['Action'],
            "expected_utility": row['Utility']
        }
        
    print(f"Training complete! Learned optimal actions for {len(policy_dict)} unique states.")
    return policy_dict, state_columns

# Run the training
policy, state_keys = train_influence_diagram("sf2_experience_data_original.csv")
with open("ryu_brain.pkl", "wb") as f:
    pickle.dump({"policy": policy, "state_keys": state_keys}, f)
print("Brain saved to ryu_brain.pkl!")