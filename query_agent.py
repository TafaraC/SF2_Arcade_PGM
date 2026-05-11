import pickle
import pandas as pd

def query_agent(brain_file="ryu_brain.pkl", opponent_filter=None):
    """Displays the full CEU table with all semantic columns."""
    try:
        with open(brain_file, "rb") as f:
            brain_data = pickle.load(f)
            policy = brain_data["policy"]
            state_keys = brain_data["state_keys"]
    except FileNotFoundError:
        print(f"ERROR: {brain_file} not found.")
        return None, None

    # Only generate the DataFrame and print if we aren't skipping the table
    rows = []
    for state_tuple, decision in policy.items():
        row = dict(zip(state_keys, state_tuple))
        row["Best_Action"] = decision["best_action"]
        row["Exp_Utility"] = round(decision["expected_utility"], 2)
        rows.append(row)

    df = pd.DataFrame(rows)
    if opponent_filter:
        df = df[df['opponent'] == opponent_filter]

    if not df.empty:
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print("\n" + "="*120)
        print(f"FULL HIERARCHICAL CPD TABLE (Filter: {opponent_filter or 'None'})")
        print("-" * 120)
        sort_order = ['opponent', 'range'] if 'range' in df.columns else ['opponent']
        print(df.sort_values(sort_order).to_string(index=False))
        print("="*120)
    else:
        print(f"\n[Table View Skipped or No Data for {opponent_filter}]")
    
    return policy, state_keys

def query_custom_state(policy, state_keys):
    """Prompts the user for state inputs to query the brain."""
    print("\n--- BRAIN QUERY MODE ---")
    print("Enter values for each state variable. (True/False or category names)")
    
    user_input_tuple = []
    for key in state_keys:
        val = input(f"Enter value for '{key}': ").strip()
        
        # Simple type conversion for boolean/numeric entries
        if val.lower() in ['true', 't', '1']: val = True
        elif val.lower() in ['false', 'f', '0']: val = False
        
        user_input_tuple.append(val)
    
    state_query = tuple(user_input_tuple)
    
    print("\nQuerying Brain...")
    if state_query in policy:
        result = policy[state_query]
        print(f"MATCH FOUND!")
        print(f"Recommended Action: {result['best_action']}")
        print(f"Expected Utility:   {round(result['expected_utility'], 2)}")
    else:
        print("RESULT: No matching state found in current experience data.")
        print("Ryu would likely default to 'Crouch_Block' in this unknown scenario.")

if __name__ == "__main__":
    # 1. Load Data first so it's always available for Querying
    brain_file = "ryu_brain.pkl"
    try:
        with open(brain_file, "rb") as f:
            data = pickle.load(f)
            policy_dict = data["policy"]
            keys = data["state_keys"]
    except FileNotFoundError:
        print(f"ERROR: {brain_file} not found.")
        exit()

    # 2. Ask if user wants the full table view
    choice = input("View CPD Table? (Enter opponent name, 'all' for full table, or 'n' to skip to query): ").strip().lower()
    
    if choice != 'n':
        # If they didn't say 'n', show the table (filtered or full)
        filter_val = None if choice == 'all' or choice == "" else choice
        query_agent(opponent_filter=filter_val)
    
    # 3. Always allow querying as long as data is loaded
    while True:
        do_query = input("\nWould you like to query a specific custom state? (y/n): ").lower()
        if do_query == 'y':
            query_custom_state(policy_dict, keys)
        else:
            break