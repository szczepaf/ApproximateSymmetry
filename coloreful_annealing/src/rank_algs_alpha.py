import pandas as pd

# Assuming 'data' is your DataFrame loaded from 'analyzed_results.csv'
data = pd.read_csv('analyzed_grid_search_results/alphas.csv')

# Define the categories
categories = [
    ("ER", 50), ("ER", 100),
    ("BA", 60), ("BA", 100),
    ("SBM", 60), ("SBM", 100),
]

# Initialize a dictionary to keep track of the scores and energy totals
scores = {}
energy_totals = {}

# Rank and score each combination within each category
for graph_type, vertex_count in categories:
    category_df = data[(data['graph_type'] == graph_type) & (data['vertex_count'] == vertex_count)]
    category_df = category_df.sort_values(by='average_energy', ascending=True)
    for rank, row in enumerate(category_df.itertuples(), 1):
        params = (row.alpha)
        # Update scores
        if params in scores:
            scores[params] += rank
            energy_totals[params]['total'] += row.average_energy
            energy_totals[params]['count'] += 1
        else:
            scores[params] = rank
            energy_totals[params] = {'total': row.average_energy, 'count': 1}

# Calculate the average energy for each parameter combination
average_energies = {params: totals['total'] / totals['count'] for params, totals in energy_totals.items()}

# Convert scores and average energies to a list of tuples and sort by total score
sorted_scores_and_energies = sorted([(params, scores[params], average_energies[params]) for params in scores], key=lambda x: x[1])

# For output and further analysis, convert the best parameters, their scores, and average energies to a DataFrame
best_params_df = pd.DataFrame(sorted_scores_and_energies, columns=['Parameters', 'Total Score', 'Average Energy'])

# Optionally, you can print or save this DataFrame to a CSV for review
print(best_params_df)
# save the df to a csv file "ranked_params_{alg_type}.csv"
best_params_df.to_csv(f"analyzed_grid_search_results/ranked_alphas.csv", index=False)
