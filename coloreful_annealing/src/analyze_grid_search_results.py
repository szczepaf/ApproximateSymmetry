import pandas as pd

# Read the CSV file
df = pd.read_csv("symmetry_results_2024-03-27-14_eigenvector_validation.csv")

# Define the constants
#division_constants = [0.01, 0.1, 0.2, 1, 10]
#probability_constants = [0.001, 0.01, 0.2, 0.1, 1]
division_constants = [0.2]
probability_constants = [0.001]

# Prepare a list for storing the results
results = []

# Loop through every combination of division and probability constants
for div_const in division_constants:
    for prob_const in probability_constants:
        # Filter the DataFrame for the current combination
        filtered_df = df[(df['division_constant'] == div_const) & (df['probability_constant'] == prob_const)]
        
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
                'division_constant': div_const,
                'probability_constant': prob_const,
                'graph_type': graph_type,
                'vertex_count': vertex_count,
                'average_energy': avg_energy
            })

# Convert the results into a DataFrame and write to a CSV file
results_df = pd.DataFrame(results)
results_df.to_csv("analyzed_results_eigenvector_validation.csv", index=False)
