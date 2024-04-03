Current graphs used for Grid Search: graphs_for_grid_search_26_03.csv. Do NOT change this file, as it is a standard for grid search comparisons.
The directory comparisons is used for storing data that compares the performance of several versions of SA and can than be used to plot the results with the scripts "plot_og_vs...".
Python scripts:

annealer.py: is the ABC used for SA
sa.py: original SA without any modification
new_sa.py: SA working with the new symmetry definition.
_naive: works with swapping the two vertices if they are similar
_twostep: chooses with probability the best vertex for swap.
_onestep: chooses from all pairs of vertices with probability (inefficient).

grid_search.py: uses always the same graphs (graphs_for_grid_search_26_03.csv) and iterates over possible division constants to find the best parameters.
analyze_grid_search_results.py: sums the symmetries for different parameters of the different versions analyzed in grid search.
rank_algs.py: ranks the parameters in term of cummulative symmetry computed in the grid search.
