import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def prepare_plot_data(df, vertex_count):
    # Initialize the plot data structure
    plot_data = {'dimension': [], 'Energy (S(A))': [], 'Algorithm': []}
    
    # Filter the dataframe for the given vertex count
    filtered_df = df[df['vertex_count'] == vertex_count]
    
    # Automatically determine algorithm names from column names
    algorithms = [col.replace('_energy', '') for col in df.columns if '_energy' in col]
    
    # Iterate over each combination of k-value and algorithm
    for k in filtered_df['dimension'].unique():
        for algorithm in algorithms:
            energy_column = f'{algorithm}_energy'
            
            # Append the data for each energy value to the plot_data dict
            for energy in filtered_df[filtered_df['dimension'] == k][energy_column]:
                plot_data['dimension'].append(k)
                plot_data['Energy (S(A))'].append(energy)
                plot_data['Algorithm'].append(algorithm.capitalize())
                
    return pd.DataFrame(plot_data)

def plot_violin(df_plot, title):
    plt.figure(figsize=(12, 6))
    sns.violinplot(x='dimension', y='Energy (S(A))', hue='Algorithm', data=df_plot, split=False)
    plt.title(title)
    plt.xlabel('dimension')
    plt.ylabel('Energy (S(A))')
    plt.legend(title='Algorithm')
    plt.tight_layout()
    plt.show()

def main():
    # Load your dataframe here
    df_BA = pd.read_csv('comparisons/comparison_of_centralities_2_grid.csv')
    vertex_counts = df_BA['vertex_count'].unique()  # Automatically handle different vertex counts

    for vertex_count in vertex_counts:
        plot_df = prepare_plot_data(df_BA, vertex_count)
        plot_violin(plot_df, f'grid Graphs with {vertex_count} Vertices')

if __name__ == "__main__":
    main()