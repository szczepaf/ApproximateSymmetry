import pandas as pd

# Read the CSV file
df = pd.read_csv("grid_search_results/symmetry_results_alpha_2024-04-08-12_page_twostep.csv")


alphas = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]



# Prepare a list for storing the results
results = []

# Loop through every combination of division and probability constants
for a in alphas:
    # Filter the DataFrame for the current combination
    filtered_df = df[(df['alpha'] == a)]
        
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
            'alpha': a,
            'graph_type': graph_type,
            'vertex_count': vertex_count,
            'average_energy': avg_energy
        })

# Convert the results into a DataFrame and write to a CSV file
results_df = pd.DataFrame(results)
results_df.to_csv("analyzed_grid_search_results/alphas.csv", index=False)
