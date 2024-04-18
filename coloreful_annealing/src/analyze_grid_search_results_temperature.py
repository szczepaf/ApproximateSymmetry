import pandas as pd

# Read the CSV file
df = pd.read_csv("grid_search_results/symmetry_results_for_temperature_2024-04-05-22_sa_temperature.csv")

# Define the constants
temp_max = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
temp_min = [0.0001, 0.001, 0.01, 0.1, 1]


# Prepare a list for storing the results
results = []

# Loop through every combination of division and probability constants
for t in temp_max:
    for s in temp_min:
        # Filter the DataFrame for the current combination
        filtered_df = df[(df['temp_max'] == t) & (df['temp_min'] == s)]
        
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
                'temp_max': t,
                'temp_min': s,
                'graph_type': graph_type,
                'vertex_count': vertex_count,
                'average_energy': avg_energy
            })

# Convert the results into a DataFrame and write to a CSV file
results_df = pd.DataFrame(results)
results_df.to_csv("analyzed_grid_search_results/temperatures.csv", index=False)
