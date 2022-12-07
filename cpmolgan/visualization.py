import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import seaborn as sns 



cluster_colors_dict = {
'Cluster0': [0, 0, 0, 1.0],
 'Cluster1': [0.17254902, 0.62745098, 0.17254902, 1.0],  # green
 'Cluster2': [0.89019608, 0.46666667, 0.76078431, 1.0],  # pink
 'Cluster3': [0.09019608, 0.74509804, 0.81176471, 1.0],   # intense cyan
 'Cluster4': [0.3, 0.3, 0.3],  # dark grey
 'Cluster5': [0.12156863, 0.46666667, 0.70588235, 1.0], # blue
 'Cluster6': [0.96862745, 0.71372549, 0.82352941, 1.0],   # baby pink 
 'Cluster7': [1.0, 0.49803922, 0.05490196, 1.0],        # orange
 'Cluster8': [0.58039216, 0.40392157, 0.74117647, 1.0],  # Dark purple
 'Cluster9': [0.54901961, 0.3372549, 0.29411765, 1.0],  # Dark brown 
 'Cluster10': [0.77254902, 0.69019608, 0.83529412, 1.0],  # ligth purple
 'Cluster11': [0.7372549, 0.74117647, 0.13333333, 1.0],  # yellow green
 'Cluster12': [1.0, 0.73333333, 0.47058824, 1.0],  # baby orange
 'Cluster13': [0.85882353, 0.85882353, 0.55294118, 1.0],  # ligth yellow
 'Cluster14': [0.68235294, 0.78039216, 0.90980392, 1.0],  # baby blue
 'Cluster15': [0.83921569, 0.15294118, 0.15686275, 1.0],  # red
 'Cluster16': [0.76862745, 0.61176471, 0.58039216, 1.0],  # ligth brown|
 'Cluster17': [0.61960784, 0.85490196, 0.89803922, 1.0],   # ligth Cyan
 'Cluster18': [0.55, 0.55, 0.55, 1.0], # ligth gray
 'Cluster19': [0.59607843, 0.8745098, 0.54117647, 1.0]   # baby green 
}

gene_colors_dict = {'TP53': [0.17254902, 0.62745098, 0.17254902, 1], # green
 'BRCA1': [1.        , 0.49803922, 0.05490196, 1], # orange
 'HSPA5': [0.83921569, 0.15294118, 0.15686275, 1], # red
 'NFKB1': [0.12156863, 0.46666667, 0.70588235, 1], # blue 
 'CREBBP': [0.58039216, 0.40392157, 0.74117647, 1], # purple
 'STAT1': [0.54901961, 0.3372549 , 0.29411765, 1], #brown
 'STAT3': [0.89019608, 0.46666667, 0.76078431, 1], # pink
 'HIF1A': [0.09019608, 0.74509804, 0.81176471, 1.0],   # intense cyan
 'NFKBIA': [0.7372549 , 0.74117647, 0.13333333, 1], # yellow
 'DMSO': [0, 0, 0, 1],
 'DMSO_train': [0.5, 0.5, 0.5, 1]
}
# [1.0, 0.59607843, 0.58823529, 1.0],         # salmon


def get_diverse_colors(Ncolors):  
    if Ncolors <= 10: 
        colors=plt.cm.tab10(np.linspace(0,1,10))[0:Ncolors]
    elif Ncolors <= 20: 
        colors=plt.cm.tab20(np.linspace(0,1,20))[0:Ncolors]
    else:
        Nreps = int(np.ceil(Ncolors/20))
        colors = plt.cm.tab20(np.linspace(0,1,20))[0:Ncolors]
        colors = np.tile( colors, (Nreps,1) )[0:Ncolors]
    return colors

def plot_projections_2(df, color_col, x='umap_x', y='umap_y', color_map={}, x_lims=[None,None], y_lims=[None,None], c_lims=[None,None],alpha=0.5, 
                     outside_legend=True, legend_ncol=1, figsize=(9,8), markersize = 1, legend_fontsize=14, legend_markerscale=5, grid=True, 
                     cmap="plasma", title=''):

    fig = plt.figure(figsize=figsize)
    
    # If continuous variable use a color map
    if (np.array(df[color_col].values).dtype==int)|((np.array(df[color_col].values).dtype==float)):
        plt.scatter( df[x].values, df[y].values, s=markersize,c=df[color_col],cmap=cmap, vmin=c_lims[0],vmax=c_lims[1], alpha=alpha)
        plt.colorbar()
    # If categorical variable, Ppot the color group with more samples first
    else:
        color_groups = df[color_col].value_counts().sort_values(ascending=False).index
        for color_group in color_groups:
            plot_df = df.loc[ df[color_col]==color_group] 
            plt.plot( plot_df[x].values, plot_df[y].values, '.', markersize=markersize, color=color_map[color_group], alpha=alpha, label=color_group)
        if outside_legend:
            plt.legend( markerscale=legend_markerscale, fontsize=legend_fontsize, loc='center left', bbox_to_anchor=(1, 0.5), ncol=legend_ncol)
        else:
            plt.legend( markerscale=legend_markerscale, fontsize=legend_fontsize,ncol=legend_ncol)
    plt.xlim(x_lims)
    plt.ylim(y_lims)
    plt.xlabel(x,fontsize=14)
    plt.ylabel(y,fontsize=14)
    plt.title(title,fontsize=14)
    if grid:
        plt.grid(b=True, which='major', color='0.65', linestyle='-')
    plt.tight_layout()
    plt.show()

def drawMST(centroid_distances, cluster_colors, figsize = (8,8), draw_numbers =True, node_size=400, font_size=10, show=False, title ='' ):
    Nclusters = centroid_distances.shape[0]
    # Create a fully conected  graph object form centroid distances 
    G   = nx.Graph(centroid_distances)
    mst = nx.minimum_spanning_tree(G) 
    # re-format labels as a dictionary (needed by nx.draw_networkx_labels)
    labels = dict()
    for c in range(Nclusters): labels[c] = str(c)
    # Plot tree 
    fig, ax = plt.subplots(num=None, figsize=figsize, facecolor='w', edgecolor='k')
    nodes_pos = nx.kamada_kawai_layout(mst)  # positions for all nodes
    nx.draw_networkx_nodes( mst, nodes_pos, nodelist=mst.nodes, node_color=cluster_colors, alpha=1, node_size=node_size)
    nx.draw_networkx_edges( mst, nodes_pos, edgelist=mst.edges, edge_color='k', alpha=1)
    if draw_numbers:
        text = nx.draw_networkx_labels( mst, nodes_pos, labels=labels, font_size=font_size)
        for _,t in text.items():
            t.set_rotation(0)
    plt.title(title,fontsize=15)
    plt.xticks([])
    plt.yticks([])
    if show:
        plt.show()
    return fig

def plot_distributions_group2group( ax, df, plot_col, idx_list=None, title="", legends=[], colors=[], 
                                 ylims=[], xlims=[], xlabel=None, cumulative=False, font_size=6):
    for i in range(len(idx_list)):
        x = df.loc[ idx_list[i],plot_col].values
        kwargs = {'cumulative': cumulative,'linewidth': 0.5}
        if colors:
            sns.distplot(x, hist=True, hist_kws=kwargs, kde_kws=kwargs, color=list(colors[i][0:3]))   
        else:
            sns.distplot(x, hist=True, norm_hist=True, kde =True, kde_kws=kwargs)   
    ax.set_title(title, size = font_size)
    if xlabel: ax.set_xlabel(xlabel, size = font_size)
    else: ax.set_xlabel('Similarity', size = font_size)
    ax.set_ylabel('Density', size= font_size)
    if ylims: ax.set_ylim(ylims)
    if xlims: ax.set_xlim(xlims)
    ax.tick_params(axis='both', labelsize=font_size)


"""
def plot_hit_counts(ref_label, cond_labels, data,  dataset="Excape", ecfp_sim_thresh="", y_lims=[0,5], figsize=(15,5), title=""):
       
    if dataset.lower()=="excape":
        ref_col = "Gene_Symbol_Excape"
        cond_col = "Gene_Symbol_conditioned"
        title_str = "Excape "
    elif dataset.lower()=="lincs":
        ref_col = "moa_lincs"
        cond_col = "moa_conditioned"
        title_str = "LINCS "
        
    # Create figure
    plt.figure(figsize=figsize)
    ax = plt.subplot(1,2,1)
    
    cond_colors = get_diverse_colors(len(cond_labels))
    for x, cond_label in enumerate(cond_labels):
        idx = ( data[ref_col]==ref_label ) & ( data[cond_col]==cond_label)
        if idx.sum()==0:
            y, std = 0, 0
        else:
            y , std =  data.loc[ idx, "mean"].values, data.loc[ idx, "std"].values
        ax.bar( x, y, color=cond_colors[x,0:3])
        ax.errorbar( x , y, std, color='k')

    ax.set_xticks(np.array(range(len(cond_labels))) -0.1 )
    ax.set_xticklabels(cond_labels, rotation=75, fontsize=14)
    ax.set_ylabel("No Cpds \n (simmilarity > "+str(ecfp_sim_thresh)+" )", fontsize=18)
    if y_lims:
        ax.set_ylim(y_lims)
    #ax.set_title(title +"\n"+ title_str+ref_label, fontsize=14)
    ax.grid()
    plt.show()

def display_structures(data, Excape_gene, oe_gene, max_display = 5):
    
    ref_smiles_col = "SMILES_standard_Excape"
    target_smiles_col = "SMILES_standard_generated"

    # Count how many times a given molecule was generated
    generated_smiles = data.groupby(by="SMILES_standard_generated").count().index
    generated_smiles_counts = data.groupby(by="SMILES_standard_generated").count().values[:,0]
    data["generated_counts"]=data["SMILES_standard_generated"].map( dict(zip(generated_smiles, generated_smiles_counts)))
    data = data.drop_duplicates( subset = "SMILES_standard_generated")[0: min(max_display, len(data)) ].reset_index(drop=True)
    
    # Add structures to the dataframe
    ref_mol_col= ref_smiles_col.replace("SMILES_standard","Mol")
    target_mol_col= target_smiles_col.replace("SMILES_standard","Mol")
    display_cols = ["Gene_Symbol_Excape","Gene_Symbol_conditioned", ref_mol_col, target_mol_col, "KNNs_ecfp_dice_similarity","generated_counts"]
    PandasTools.AddMoleculeColumnToFrame(data, smilesCol=ref_smiles_col , molCol=ref_mol_col ,includeFingerprints=False)
    PandasTools.AddMoleculeColumnToFrame(data, smilesCol=target_smiles_col ,molCol=target_mol_col.replace("SMILES_standard","Mol"),includeFingerprints=False)
    display(HTML(data[display_cols].to_html(escape=False)))


def plot_projections( embedding, metadata_df, color_col, color_dict=None, title=None, 
                     x_lims=[None,None], y_lims=[None,None], markersize=3, marker='o',
                    figsize=(6,3.6)):
    color_groups = metadata_df[color_col].unique()
    fig = plt.figure(figsize=figsize)
    ax=plt.subplot(111)
    legends =[]              
    for i , group in enumerate(color_groups):
        idx = metadata_df[color_col] == group
        if color_dict:
            ax.plot( embedding[idx,0] , embedding[idx,1], markersize=markersize, marker=marker,color=color_dict[group], linewidth=0, label=group)
        else:
            ax.plot( embedding[idx,0] , embedding[idx,1], markersize=markersize, marker=marker, linewidth=0, label=group)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=False, shadow=False, ncol=1,fontsize=12)
    plt.tight_layout()
    plt.xlabel('UMAP 1',fontsize=14)
    plt.ylabel('UMAP 2',fontsize=14)
    plt.xlim(x_lims)
    plt.ylim(y_lims)
    if title: plt.title(title,fontsize=14)
    return fig, ax
"""

