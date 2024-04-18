from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('comparisons/measurement_of_distances_from_LR_permutation.csv')

# Define your 'k' and 'l' values as arrays
k_values = np.array(df['k'].unique())
l_values = np.array(df['l'].unique())

k_values = np.sort(k_values)
l_values = np.sort(l_values)


versions_energy = ['sa_s', 'sa_eigenvector_s', 'sa_pagerank_s', 'sa_betweenness_s']
versions_hamming_distance = ['hamming_distance_sa', 'hamming_distance_eigenvector', 'hamming_distance_pagerank', 'hamming_distance_betweenness']


# Set up a grid for the heatmap
grid_shape = (len(k_values), len(l_values))

# Function to create heatmap data
def create_heatmap_data_S_differences(df, k_values, l_values, value_column):
    heatmap_data = np.zeros(grid_shape)
    for i, k in enumerate(k_values):
        for j, l in enumerate(l_values):
            condition = (df['k'] == k) & (df['l'] == l)
            if df[condition].empty:
                heatmap_data[i, j] = np.nan
            else:
                heatmap_data[i, j] = df.loc[condition, value_column].mean() - df.loc[condition, 'energy_with_perfect_LR_permutation'].mean()
    return heatmap_data

def create_heatmap_data_hamming_distances(df, k_values, l_values, value_column):
    heatmap_data = np.zeros(grid_shape)
    for i, k in enumerate(k_values):
        for j, l in enumerate(l_values):
            condition = (df['k'] == k) & (df['l'] == l)
            if df[condition].empty:
                heatmap_data[i, j] = np.nan
            else:
                heatmap_data[i, j] = df.loc[condition, value_column].mean()
    return heatmap_data

# Column names for the different versions

for version in versions_energy:
    plt.figure()
    data = create_heatmap_data_S_differences(df, k_values, l_values, version)
    plt.imshow(data, cmap='hot_r', aspect='auto', interpolation='none')
    
    # Customize the plot to match the provided image
    plt.colorbar(label='AFP: S(A) - S(LR)')
    plt.xlabel('#(swaps)')
    plt.ylabel('#(Rewired edges)')
    plt.title(f'AFP: S(A) - S(LR), version: {version}')
    
    # Set the ticks to match k_values and l_values
    plt.xticks(np.arange(len(l_values)), l_values)
    plt.yticks(np.arange(len(k_values)), k_values)
    
    # Uncomment the next line to display the figure
    plt.show()
    




# now for the second part
for version in versions_hamming_distance:
    plt.figure()
    data = create_heatmap_data_S_differences(df, k_values, l_values, version)
    plt.imshow(data, cmap='hot', aspect='auto', interpolation='none')
    
    # Customize the plot to match the provided image
    plt.colorbar(label='AFP: distance from optimum')
    plt.xlabel('#(swaps)')
    plt.ylabel('#(Rewired edges)')
    plt.title(f'AFP: hamming distance from LR, {version}')
    
    # Set the ticks to match k_values and l_values
    plt.xticks(np.arange(len(l_values)), l_values)
    plt.yticks(np.arange(len(k_values)), k_values)
    
    # Uncomment the next line to display the figure
    plt.show()
