import pandas as pd

# Assuming 'data' is your DataFrame loaded from 'analyzed_results.csv'
data = pd.read_csv('analyzed_results_pagerank_twostep.csv')

# Define the categories
categories = [
    ("ER", 50), ("ER", 100),
    ("BA", 60), ("BA", 100),
    ("SBM", 60), ("SBM", 100),
]

# Initialize a dictionary to keep track of the scores
scores = {}

# Rank and score each combination within each category
for graph_type, vertex_count in categories:
    category_df = data[(data['graph_type'] == graph_type) & (data['vertex_count'] == vertex_count)]
    category_df = category_df.sort_values(by='average_energy', ascending=True)
    for rank, row in enumerate(category_df.itertuples(), 1):
        params = (row.division_constant, row.probability_constant)
        if params in scores:
            scores[params] += rank
        else:
            scores[params] = rank

# Convert scores to a list of tuples and sort by total score
sorted_scores = sorted(scores.items(), key=lambda x: x[1])



# For output and further analysis, convert the best parameters and their scores to a DataFrame
best_params_df = pd.DataFrame(sorted_scores, columns=['Parameters', 'Total Score'])

# Optionally, you can print or save this DataFrame to a CSV for review
print(best_params_df)
# save the df to a csv file "ranked_params_{alg_type}.csv"
best_params_df.to_csv(f"ranked_params_pagerank_twostep.csv", index=False)
