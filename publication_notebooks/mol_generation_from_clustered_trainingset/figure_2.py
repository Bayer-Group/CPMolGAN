from scipy import spatial, stats
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
import pandas as pd
import os
import cpmolgan.visualization as vi


def sort_centroids( model, ref_centroid , metric="euclidean"):
    """
    Exchanges centroids such that they are sorted by increasing distance to ref centroid"
    """
    centroids = model.cluster_centers_    
    centroid_distances = spatial.distance.cdist( centroids , centroids , metric=metric)
    D_ref = centroid_distances[ref_centroid,:] # distance relative to the ref cluster
    sorted_idx = np.argsort(D_ref)
    temp = centroids.copy()
    for i, idx in enumerate(sorted_idx):
        centroids[i,:]= temp[idx,:]
    model.cluster_centers_ = centroids 
    return model

def plot_distributions_comparison_sets( comparison_sets , knns_data, K, k_col, sim_col, ref_output_file='', 
                          xticks=None, yticks=None, xlims=None, ylims=None, xlabel=''):
    """
    Plots distributions of distances between sets of generate molecules presented in Figure 2c 
    """
    # Plot results
    for i, comparison_sets in enumerate(comparison_sets):
        plot_df = knns_data.loc[ knns_data[k_col].isin(range(0,K+1)) ].reset_index(drop=True)
        selected_idxs, colors = [], []
        for comp_set in comparison_sets:
            cluster_l, cluster_r = comp_set
            print(cluster_l, cluster_r)
            selected_idxs.append( (plot_df["cluster_l"]==cluster_l) & (plot_df["cluster_r"]==cluster_r) )
            colors.append(vi.cluster_colors_dict[cluster_r])

        fig = plt.figure(num=None, figsize=(1.7, 1), dpi=300, facecolor='w', edgecolor='k')
        sns.set_style(style='white') 
        ax = plt.subplot(1,1,1)
        if len(xlabel)==0:
            xlabel = sim_col.replace('KNNs_','').replace('_',' ')
        vi.plot_distributions_group2group( ax, plot_df, sim_col, selected_idxs , title="", xlims=xlims, ylims=xlims, colors=colors, 
                                        xlabel= xlabel, font_size=4)
        ax.tick_params(axis='both', which='major', labelsize=3.5, pad=0)

        if xticks: ax.set_xticks(xticks)
        if yticks: ax.set_yticks(yticks)     
        plt.tight_layout()
        plt.show()

        if ref_output_file: fig.savefig(ref_output_file.replace('IDX',str(i)))


def load_knns_comparison_sets(comparison_sets, knn_path, knns_ref_file):
    """
    Load nearest neighbors data from generate molecules presented in Figure 2c
    """
    # get unique sets, to not repeat cmparisons appearing in tow branches
    unique_sets = []
    for s in comparison_sets:
        if not(s in unique_sets): unique_sets.append(s)

    # Read data
    knns_data = pd.DataFrame()
    for comp_set in unique_sets:
        cluster_l , cluster_r = comp_set
        filename =knns_ref_file.replace("CLUSTER1", cluster_l).replace("CLUSTER2",cluster_r)
        temp_knns =  pd.read_csv( os.path.join(knn_path,filename), index_col=0)
        temp_knns["cluster_l"] = cluster_l
        temp_knns["cluster_r"] = cluster_r
        knns_data = pd.concat( [knns_data, temp_knns ])
    knns_data = knns_data.reset_index(drop=True)    

    return knns_data

def correct_same_cluster_nn(knns_data, k_col):
    """ Removes first neighbors between same clusters since they always have simmilairty=1 """
    same_cluster_idx = knns_data['cluster_l']==knns_data['cluster_r']
    bad_nn_idx = same_cluster_idx & ( knns_data[k_col]==0 )
    knns_data = knns_data.loc[bad_nn_idx ==False ].reset_index(drop=True)
    knns_data.loc[ same_cluster_idx, k_col] = knns_data.loc[ same_cluster_idx, k_col].values -1
    return knns_data

