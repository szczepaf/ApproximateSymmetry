import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel
import numpy as np


def prepare_plot_data(df, vertex_count):
    plot_data = {'block_count': [], 'Energy (S(A))': [], 'Algorithm': []}
    
    filtered_df = df[df['vertex_count'] == vertex_count]

    block_counts = [2, 3, 5] 
    algorithms = ['zero', 'new', 'half']
    
    for b in block_counts:
        for algorithm in algorithms:
            energy_column = f'{algorithm}_energy'
            for energy in filtered_df[filtered_df['block_count'] == b][energy_column]:
                plot_data['block_count'].append(f'{b}')
                plot_data['Energy (S(A))'].append(energy)
                plot_data['Algorithm'].append(f'{algorithm.capitalize()} SA')
                
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
    
def cohens_d(group1, group2):
    s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    pooled_std = np.sqrt((s1 ** 2 + s2 ** 2) / 2)
    
    d = (np.mean(group1) - np.mean(group2)) / pooled_std
    return d

def main():
    df_SBM = pd.read_csv('comparison_of_original_and_new_sym_def_SBM.csv')
    vertex_counts = [50, 100, 150]

    for vertex_count in vertex_counts:
        plot_df = prepare_plot_data(df_SBM, vertex_count)
        plot_violin(plot_df, f'SBM Graphs with {vertex_count} Vertices')
        
    # Calculate p-values and Cohen's D for Zero SA vs. New SA and New SA vs. Half SA
    block_counts = [2, 3, 5]
    comparisons = [('zero', 'new'), ('half', 'new')]
    for comparison in comparisons:
        p_values_df = pd.DataFrame(index=vertex_counts, columns=[f'{b}' for b in block_counts])
        cohens_d_df = pd.DataFrame(index=vertex_counts, columns=[f'{b}' for b in block_counts])
        
        for vertex_count in vertex_counts:
            for b in block_counts:
                df_filtered = df_SBM[(df_SBM['vertex_count'] == vertex_count) & (df_SBM['block_count'] == b)]
                stat, p_value = ttest_rel(df_filtered[f'{comparison[0]}_energy'], df_filtered[f'{comparison[1]}_energy'])
                p_values_df.loc[vertex_count, f'{b}'] = p_value
                
                d = cohens_d(df_filtered[f'{comparison[0]}_energy'], df_filtered[f'{comparison[1]}_energy'])
                cohens_d_df.loc[vertex_count, f'{b}'] = d
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(p_values_df.astype(float), annot=True, cmap='viridis', cbar_kws={'label': 'p-value'})
        plt.title(f'Paired t-test p-values ({comparison[0].capitalize()} SA vs. {comparison[1].capitalize()} SA)')
        plt.xlabel('block_count')
        plt.ylabel('# Nodes')
        plt.show()
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cohens_d_df.astype(float), annot=True, cmap='coolwarm', center=0, cbar_kws={'label': "Cohen's d"})
        plt.title(f"Cohen's d ({comparison[0].capitalize()} SA vs. {comparison[1].capitalize()} SA)")
        plt.xlabel('block_count')
        plt.ylabel('# Nodes')
        plt.show()

    

if __name__ == "__main__":
    main()