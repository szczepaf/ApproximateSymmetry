import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel
import numpy as np



def prepare_plot_data(df, vertex_count):
    plot_data = {'Edge Density': [], 'Energy (S(A))': [], 'Algorithm': []}
    
    filtered_df = df[df['vertex_count'] == vertex_count]
    edge_densities = [0.1, 0.2, 0.3, 0.4, 0.5]
    algorithms = ['zero', 'new', 'half']
    
    for edge_density in edge_densities:
        for algorithm in algorithms:
            energy_column = f'{algorithm}_energy'
            for energy in filtered_df[filtered_df['edge_density'] == edge_density][energy_column]:
                plot_data['Edge Density'].append(edge_density)
                plot_data['Energy (S(A))'].append(energy)
                plot_data['Algorithm'].append(f'{algorithm.capitalize()} SA')
                
    return pd.DataFrame(plot_data)

def plot_violin(df_plot, title):
    plt.figure(figsize=(12, 6))
    sns.violinplot(x='Edge Density', y='Energy (S(A))', hue='Algorithm', data=df_plot, split=False)
    plt.title(title)
    plt.xlabel('Edge Density')
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

    df_ER = pd.read_csv('comparison_of_original_and_new_sym_def_ER.csv')
    edge_densities = [0.1, 0.2, 0.3, 0.4, 0.5]
    vertex_counts = [20, 50, 100]

    small_plot_df = prepare_plot_data(df_ER, 20)
    medium_plot_df = prepare_plot_data(df_ER, 50)
    large_plot_df = prepare_plot_data(df_ER, 100)

    plot_violin(small_plot_df, 'Small ER Graphs')
    plot_violin(medium_plot_df, 'Medium ER Graphs')
    plot_violin(large_plot_df, 'Large ER Graphs')

    p_values_df = pd.DataFrame(index=vertex_counts, columns=edge_densities)
    for vertex_count in vertex_counts:
        for edge_density in edge_densities:
            df_filtered = df_ER[(df_ER['vertex_count'] == vertex_count) & (df_ER['edge_density'] == edge_density)]
            stat, p_value = ttest_rel(df_filtered['zero_energy'], df_filtered['new_energy'])
            p_values_df.loc[vertex_count, edge_density] = p_value

    p_values_df.index = p_values_df.index.map(str)
    p_values_df.columns = p_values_df.columns.map(str)

    plt.figure(figsize=(8, 6))
    sns.heatmap(p_values_df.astype(float), annot=True, cmap='viridis', cbar_kws={'label': 'p-value'})
    plt.title('Paired t-test p-values (Zero SA vs. New SA)')
    plt.xlabel('Edge Density')
    plt.ylabel('# Nodes')
    plt.show()

    p_values_df2 = pd.DataFrame(index=vertex_counts, columns=edge_densities)
    for vertex_count in vertex_counts:
        for edge_density in edge_densities:
            df_filtered = df_ER[(df_ER['vertex_count'] == vertex_count) & (df_ER['edge_density'] == edge_density)]
            stat, p_value = ttest_rel(df_filtered['half_energy'], df_filtered['new_energy'])
            p_values_df2.loc[vertex_count, edge_density] = p_value

    p_values_df2.index = p_values_df2.index.map(str)
    p_values_df2.columns = p_values_df2.columns.map(str)

    plt.figure(figsize=(8, 6))
    sns.heatmap(p_values_df2.astype(float), annot=True, cmap='viridis', cbar_kws={'label': 'p-value'})
    plt.title('Paired t-test p-values (Half SA vs. New SA)')
    plt.xlabel('Edge Density')
    plt.ylabel('# Nodes')
    plt.show()

    cohens_d_df = pd.DataFrame(index=vertex_counts, columns=edge_densities)
    cohens_d_df2 = pd.DataFrame(index=vertex_counts, columns=edge_densities)
    
    for vertex_count in vertex_counts:
        for edge_density in edge_densities:
            df_filtered = df_ER[(df_ER['vertex_count'] == vertex_count) & (df_ER['edge_density'] == edge_density)]
            
            d = cohens_d(df_filtered['zero_energy'], df_filtered['new_energy'])
            d2 = cohens_d(df_filtered['half_energy'], df_filtered['new_energy'])
            cohens_d_df.loc[vertex_count, edge_density] = d
            cohens_d_df2.loc[vertex_count, edge_density] = d2

    cohens_d_df.index = cohens_d_df.index.map(str)
    cohens_d_df.columns = cohens_d_df.columns.map(str)
    cohens_d_df2.index = cohens_d_df2.index.map(str)
    cohens_d_df2.columns = cohens_d_df2.columns.map(str)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cohens_d_df.astype(float), annot=True, cmap='coolwarm', center=0, cbar_kws={'label': "Cohen's d"})
    plt.title("Cohen's d (Zero SA vs. New SA)")
    plt.xlabel('Edge Density')
    plt.ylabel('# Nodes')
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.heatmap(cohens_d_df2.astype(float), annot=True, cmap='coolwarm', center=0, cbar_kws={'label': "Cohen's d"})
    plt.title("Cohen's d (Half SA vs. New SA)")
    plt.xlabel('Edge Density')
    plt.ylabel('# Nodes')
    plt.show()

    

if __name__ == "__main__":
    main()
