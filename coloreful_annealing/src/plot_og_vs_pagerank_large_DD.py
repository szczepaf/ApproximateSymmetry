import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel
import numpy as np

def prepare_plot_data(df, vertex_count, k):
    plot_data = {'probability': [], 'Energy (S(A))': [], 'Algorithm': []}
    
    filtered_df = df[(df['vertex_count'] == vertex_count) & (df['probability'] == k)]
    algorithms = ['pagerank', 'og']
    
    for algorithm in algorithms:
        energy_column = f'{algorithm}_energy'
        for energy in filtered_df[energy_column]:
            plot_data['probability'].append(f'{k}')
            plot_data['Energy (S(A))'].append(energy)
            plot_data['Algorithm'].append(f'{algorithm.capitalize()} SA')
                
    return pd.DataFrame(plot_data)

def plot_violin(df_plot, title):
    plt.figure(figsize=(12, 6))
    sns.violinplot(x='probability', y='Energy (S(A))', hue='Algorithm', data=df_plot, split=False)
    plt.title(title)
    plt.xlabel('probability')
    plt.ylabel('Energy (S(A))')
    plt.legend(title='Algorithm')
    plt.tight_layout()
    plt.show()

def cohens_d_function(group1, group2):
    s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    pooled_std = np.sqrt((s1 ** 2 + s2 ** 2) / 2)
    
    d = (np.mean(group1) - np.mean(group2)) / pooled_std
    return d

def main():
    df_BA = pd.read_csv('comparisons/comparisons_of_pagerank_vs_og_on_large_DD.csv')
    vertex_counts = [300, 500]
    probabilities = [0.05, 0.1, 0.3]

    for vertex_count in vertex_counts:
        for p in probabilities:
            plot_df = prepare_plot_data(df_BA, vertex_count, p)
            plot_violin(plot_df, f'BA Graphs with {vertex_count} Vertices, k={p}')

    comparisons = [('pagerank', 'og')]
    for comparison in comparisons:
        p_values_df = pd.DataFrame(index=vertex_counts, columns=[f'{k}' for k in probabilities])
        cohens_d_df = pd.DataFrame(index=vertex_counts, columns=[f'{k}' for k in probabilities])
        
        for vertex_count in vertex_counts:
            for p in probabilities:
                df_filtered = df_BA[(df_BA['vertex_count'] == vertex_count) & (df_BA['probability'] == p)]
                stat, p_value = ttest_rel(df_filtered[f'{comparison[0]}_energy'], df_filtered[f'{comparison[1]}_energy'])
                p_values_df.loc[vertex_count, f'{p}'] = p_value
                
                cohens_d = cohens_d_function(df_filtered[f'{comparison[0]}_energy'], df_filtered[f'{comparison[1]}_energy'])
                cohens_d_df.loc[vertex_count, f'{p}'] = cohens_d
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(p_values_df.astype(float), annot=True, cmap='viridis', cbar_kws={'label': 'p-value'})
        plt.title(f'Paired t-test p-values ({comparison[0].capitalize()} SA vs. {comparison[1].capitalize()} SA)')
        plt.xlabel('probability')
        plt.ylabel('# Nodes')
        plt.show()
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cohens_d_df.astype(float), annot=True, cmap='coolwarm', center=0, cbar_kws={'label': "Cohen's d"})
        plt.title(f"Cohen's d ({comparison[0].capitalize()} SA vs. {comparison[1].capitalize()} SA)")
        plt.xlabel('probability')
        plt.ylabel('# Nodes')
        plt.show()

if __name__ == "__main__":
    main()
