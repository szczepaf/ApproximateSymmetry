import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def prepare_plot_data(df, vertex_count, k_value):
    # Initialize the plot data structure
    plot_data = {'block_count': [], 'Energy (S(A))': [], 'Algorithm': []}
    
    # Filter the dataframe for the given vertex count and k value
    filtered_df = df[(df['vertex_count'] == vertex_count) & (df['block_count'] == k_value)]
    
    # Automatically determine algorithm names from column names
    algorithms = [col.replace('_energy', '') for col in df.columns if '_energy' in col]
    
    # Iterate over each algorithm
    for algorithm in algorithms:
        energy_column = f'{algorithm}_energy'
        
        # Append the data for each energy value to the plot_data dict
        for energy in filtered_df[energy_column]:
            plot_data['block_count'].append(k_value)
            plot_data['Energy (S(A))'].append(energy)
            plot_data['Algorithm'].append(algorithm.capitalize())
                
    return pd.DataFrame(plot_data)

def plot_violin(df_plot, title):
    plt.figure(figsize=(12, 6))
    sns.violinplot(x='block_count', y='Energy (S(A))', hue='Algorithm', data=df_plot, split=False)
    plt.title(title)
    plt.xlabel('block_count')
    plt.ylabel('Energy (S(A))')
    plt.legend(title='Algorithm')
    plt.tight_layout()
    plt.show()

def main():
    # Load your dataframe here
    df_BA = pd.read_csv('comparisons/comparison_of_centralities_2_SBM.csv')
    vertex_counts = df_BA['vertex_count'].unique()  # Automatically handle different vertex counts

    for vertex_count in vertex_counts:
        k_values = df_BA[df_BA['vertex_count'] == vertex_count]['block_count'].unique()
        for k_value in k_values:
            plot_df = prepare_plot_data(df_BA, vertex_count, k_value)
            plot_violin(plot_df, f'SBM Graphs with {vertex_count} Vertices, block_count={k_value}')

if __name__ == "__main__":
    main()
