import pandas as pd

# Read the CSV file
df = pd.read_csv("grid_search_results/symmetry_results_2024-04-10-22_wl.csv")

# Define the constants
iterations = [1, 2, 3]
probability_constants = [0.3, 0.6, 0.9]


# Prepare a list for storing the results
results = []

# Loop through every combination of division and probability constants
for iteration in iterations:
    for prob_const in probability_constants:
        # Filter the DataFrame for the current combination
        filtered_df = df[(df['iterations'] == iteration) & (df['probability_constant'] == prob_const)]
        
        # Define the conditions for graph types and vertex counts
        conditions = [
            ("ER", 50), ("ER", 100),
            ("BA", 60), ("BA", 100),
            ("SBM", 60), ("SBM", 100),
        ]
        
        # For each condition, calculate the average energy and store the results
        for graph_type, vertex_count in conditions:
            condition_df = filtered_df[(filtered_df['graph_type'] == graph_type) & (filtered_df['vertex_count'] == vertex_count)]
            avg_energy = condition_df['energy'].mean()
            results.append({
                'iteration': iteration,
                'probability': prob_const,
                'graph_type': graph_type,
                'vertex_count': vertex_count,
                'average_energy': avg_energy
            })

# Convert the results into a DataFrame and write to a CSV file
results_df = pd.DataFrame(results)
results_df.to_csv("analyzed_grid_search_results/wl.csv", index=False)
