import numpy as np
import pandas as pd
from multiprocessing import Pool
import logging 
from rdkit.Chem import MolFromSmiles, AllChem, MACCSkeys
from rdkit import DataStructs
import tqdm
from sklearn.metrics import pairwise_distances

paralleize_simMat_computation = True

def smiles_to_ecfps_parallel(smiles):
    pool = Pool(24) 
    mols = pool.map( smile_to_mol , smiles )
    ecfps = pool.map( mol_to_ecfp , mols )
    pool.close()
    pool.join()
    return ecfps

def smiles_to_maccs_parallel(smiles):
    pool = Pool(24) 
    mols = pool.map( smile_to_mol , smiles )
    maccs = pool.map( mol_to_maccs , mols )
    pool.close()
    pool.join()
    return maccs

def bulk_dice_similarities_parallel(fp_pairs):
    pool = Pool(16) 
    simMat = np.array( pool.map( bulk_dice_sim , fp_pairs ) )
    pool.close()
    pool.join()
    return simMat
    
def smile_to_mol(smile):
    "Warper to make it pickable"
    return MolFromSmiles(smile)

def mol_to_ecfp(mol):
    "Warper to make it pickable"
    return AllChem.GetMorganFingerprintAsBitVect(mol, 3,1024, useChirality=True)

def mol_to_maccs(mol):
    "Warper to make it pickable"
    return MACCSkeys.GenMACCSKeys(mol)

def bulk_dice_sim(fp_pair):
    "Warper to make it pickable"
    return DataStructs.BulkDiceSimilarity(fp_pair[0], fp_pair[1])

def set_to_set_knns_multiple_pairs( pairs , k, Nprocessors=12):
    with Pool(Nprocessors) as pool:
        knns = list( tqdm.tqdm( pool.imap(set_to_set_knns, pairs, k), total=len(pairs) ) )
    knns_set1 = pd.concat( [ knns[i][0] for i in range(len(pairs)) ], axis=0).reset_index(drop=True)  
    knns_set2 = pd.concat( [ knns[i][1] for i in range(len(pairs)) ], axis=0).reset_index(drop=True)  
    pool.close()
    pool.join()
    return knns_set1, knns_set2

def set_to_set_knns( pair , k ):
    """Compute bidrectional k nearests neighbors between two sets of compounds based on ecfp and maccs fingerprints
    :param pair: list of two dataframes, each having columns ecfp and maccs
    :param k: number fo nearest neigbohrs"""
    set1 = pair[0].reset_index(drop=True)
    set2 = pair[1].reset_index(drop=True)
    set1_knns, set2_knns = get_nearest_neighbors_from_fingerprints(set1, set2, k)
    return set1_knns, set2_knns 

def set_to_set_knns_mol_embeddings( pair , k , embedding_cols, metric='cosine'):
    """Compute bidrectional k nearests neighbors between two sets of compounds based on molecular embeddings 
    :param pair: list of two dataframes, each having columns ecfp and maccs
    :param k: number fo nearest neigbohrs
    :param embedding_cols: list os columns containing the embeddings"""
    set1 = pair[0].reset_index(drop=True)
    set2 = pair[1].reset_index(drop=True)
    set1_knns, set2_knns = get_nearest_neighbors_from_mol_embeddings(set1, set2, k, embedding_cols, metric=metric)
    return set1_knns, set2_knns 

def dice_similairty_matrix( set1 , set2 ):
    SimMat = bulk_dice_similarities_parallel([(fp, set2)  for fp in set1])
    return SimMat

def get_nearest_neighbors_from_fingerprints(set1_df, set2_df, k, drop_fingerprint_cols = True):

    switch_sets_for_speed = (len(set2_df) > len(set1_df)) & paralleize_simMat_computation
    if switch_sets_for_speed:
        set1_df, set2_df = set2_df, set1_df
        
    ecfps_set1 , maccs_set1 = set1_df["ecfp"].values, set1_df["maccs"].values
    ecfps_set2 , maccs_set2 = set2_df["ecfp"].values, set2_df["maccs"].values
    if drop_fingerprint_cols:
        set1_df = set1_df.drop(columns=["ecfp","maccs"])
        set2_df = set2_df.drop(columns=["ecfp","maccs"])

    # Compute simmilarity matrices
    if paralleize_simMat_computation:
        ecfps_SimMat = bulk_dice_similarities_parallel([(fp, ecfps_set2)  for fp in ecfps_set1])
        maccs_SimMat = bulk_dice_similarities_parallel([(fp, maccs_set2) for fp in maccs_set1])
    else:
        ecfps_SimMat = np.array([DataStructs.BulkDiceSimilarity(fp, ecfps_set2) for fp in ecfps_set1])
        maccs_SimMat = np.array([DataStructs.BulkDiceSimilarity(fp, maccs_set2) for fp in maccs_set1])

    # Fetch nearest neighbors 
    set1_knns_ecfp,  set2_knns_ecfp  = get_nearest_neighbors_from_sim_matrix( k, set1_df, set2_df, ecfps_SimMat, output_prefix= "ecfp_dice" )
    set1_knns_maccs, set2_knns_maccs = get_nearest_neighbors_from_sim_matrix( k, set1_df, set2_df, maccs_SimMat, output_prefix= "maccs_dice")
    concat_cols = list(set(set1_knns_maccs.columns).difference(set1_knns_ecfp))
    set1_knns = pd.concat( [ set1_knns_ecfp, set1_knns_maccs[concat_cols]], axis=1 )
    set2_knns = pd.concat( [ set2_knns_ecfp, set2_knns_maccs[concat_cols]], axis=1 )
    
    if switch_sets_for_speed:
        set1_knns, set2_knns = set2_knns, set1_knns
        
    return set1_knns, set2_knns

def get_nearest_neighbors_from_mol_embeddings(set1_df, set2_df, k, embedding_cols, metric='cosine', drop_embedding_cols=True):
    
    embd_set1 , embd_set2 = set1_df[embedding_cols].values, set2_df[embedding_cols].values
    if drop_embedding_cols:
        set1_df = set1_df.drop(columns=embedding_cols)
        set2_df = set2_df.drop(columns=embedding_cols)
    SimMat = pairwise_distances(embd_set1, embd_set2, metric=metric)
    if metric == 'cosine':
        SimMat = 1-SimMat
    else:
        SimMat = -SimMat
    set1_knns, set2_knns = get_nearest_neighbors_from_sim_matrix( k, set1_df, set2_df, SimMat, output_prefix= "MolEmb_"+metric )

    return set1_knns, set2_knns

def get_nearest_neighbors_from_sim_matrix( k, set1_df, set2_df, sim_matrix , output_prefix= "", bidirectional=True):
    
    # prefix for all added columns with nearest neighbor results
    pref= "KNNs_"+output_prefix+"_"
    
    # For each element in set 1, get Knns on set 2 and viceversa
    set1_knn_indices, set1_knn_sims = get_top_k( sim_matrix, k , mode="max", axis=0)
    if bidirectional:
        set2_knn_indices, set2_knn_sims = get_top_k( sim_matrix, k , mode="max", axis=1)
        set2_knn_indices, set2_knn_sims = set2_knn_indices.transpose() , set2_knn_sims.transpose()
    
    # Adjust number of neighbors in case len(set1_df)<k or len(set2_df)<k 
    set1_k, set2_k = set1_knn_indices.shape[1], set2_knn_indices.shape[1]
    
    # Add nearest neighbors and similarity values to the original data frames
    set1_out = pd.concat([set1_df]*set1_k ).sort_index().reset_index(drop=True)
    set1_out[pref+"SMILES_standard"] = set2_df.loc[ set1_knn_indices.ravel() , "SMILES_standard" ].values
    set1_out[pref+"similarity"] = set1_knn_sims.ravel()
    set1_out[pref+"k"] = np.tile(range(set1_k),len(set1_df))
    if bidirectional:
        set2_out = pd.concat([set2_df]*set2_k).sort_index().reset_index(drop=True)
        set2_out[pref+"SMILES_standard"] = set1_df.loc[ set2_knn_indices.ravel() , "SMILES_standard" ].values
        set2_out[pref+"similarity"] = set2_knn_sims.ravel()
        set2_out[pref+"k"] = np.tile(range(set2_k),len(set2_df))
        return set1_out, set2_out
    else:
        return set1_out


def get_top_k( arr:np.ndarray, k:int=5, axis:int=0, mode:str="max" ):
    
    """
    Gets the indices of the top K (smallest or largest) elements of a 2D array along a specified dimension sorted according to mode. 
    The code uses a partition which performs a local sort and is faster performing a full sort
    :param arr: 2-dimensional numpy array
    :param k: number of elements to retrieve
    :param mode: direction of sorting
        mode=max retrieves the top k largest elements
        mode=min retrieves the top k smallest elements
    :param axis: dimension to find the top elements. 
        axis=0 gets topK column indices/values for each row
        axis=1 gets topK row indices/values for each column
    :returns: array with indices , array with values
    """

    # Set k to at most the maximum allowed by the dimensions of arr
    if axis==0:
        k =  min( k, arr.shape[1] ) 
    elif axis==1:
        k =  min( k, arr.shape[0] ) 
        arr = np.transpose(arr)
        
    # Use partition to return UNORDERED K largest indices.  
    if mode =="max":
        col_indices = np.argpartition( arr, -k )[:, -k:]
    elif mode =="min":
        col_indices = np.argpartition( arr, k )[:, :k]
    else:
        logging.error('Unrecognize mode')
        raise ValueError('Unrecognize mode')
    
    # Sort column indices based on their array values
    rows = np.repeat( np.arange(arr.shape[0]), k)
    values = arr[ rows , col_indices.ravel() ].reshape( arr.shape[0], k)
    sorted_col_indices_idx = np.argsort(values, axis=1)
    sorted_col_indices = col_indices[ rows , sorted_col_indices_idx.ravel() ].reshape( arr.shape[0], k )
    if mode=="max":
        sorted_col_indices = sorted_col_indices[:,::-1]    
    sorted_values = arr[ rows,sorted_col_indices.ravel() ].reshape( arr.shape[0], k )
    
    if axis==1:
        sorted_col_indices = np.transpose(sorted_col_indices)    
        sorted_values = np.transpose(sorted_values)    
        
    return sorted_col_indices, sorted_values

def get_nearest_neighbors_from_profiles(set1_df, set2_df, k, embedding_cols, id_cols_1, id_cols_2, metric='cosine', 
                                        bidirectional=True, drop_embedding_cols=True, output_prefix=''):
    
    embd_set1 , embd_set2 = set1_df[embedding_cols].values, set2_df[embedding_cols].values
    if drop_embedding_cols:
        set1_df = set1_df.drop(columns=embedding_cols)
        set2_df = set2_df.drop(columns=embedding_cols)
        disMat = pairwise_distances(embd_set1, embd_set2, metric=metric)
    
    # For each element in set 1, get Knns on set 2 and viceversa
    set1_knn_indices, set1_knn_dists = get_top_k( disMat, k , mode="min", axis=0)
    set1_k = set1_knn_indices.shape[1] # in case len(set1_df)<k 
    if bidirectional:
        set2_knn_indices, set2_knn_dists = et_top_k( disMat, k , mode="min", axis=1)
        set2_knn_indices, set2_knn_dists = set2_knn_indices.transpose() , set2_knn_dists.transpose()
        set2_k = set2_knn_indices.shape[1] # in case len(set2_df)<k 
    
    # Add nearest neighbors and distances to the original data frames
    pref= "KNNs_"+output_prefix+"_"
    set1_out = pd.concat([set1_df]*set1_k ).sort_index().reset_index(drop=True)
    for id_col in id_cols_2: 
        set1_out[pref+id_col] = set2_df.loc[ set1_knn_indices.ravel() , id_col ].values
    set1_out[pref+'_'+metric+"_distance"] = set1_knn_dists.ravel()
    set1_out[pref+"k"] = np.tile(range(set1_k),len(set1_df))
    if bidirectional:
        set2_out = pd.concat([set2_df]*set2_k).sort_index().reset_index(drop=True)
        for id_col in id_cols_1: 
            set2_out[pref+id_col] = set1_df.loc[ set2_knn_indices.ravel() , id_col ].values
        set2_out[pref+'_'+metric+"_distance"] = set2_knn_dists.ravel()
        set2_out[pref+"k"] = np.tile(range(set2_k),len(set2_df))
        return set1_out, set2_out
    else:
        return set1_out

