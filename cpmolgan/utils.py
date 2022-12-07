import os
import numpy as np
import pandas as pd
from collections import defaultdict
from rdkit import Chem, DataStructs
from selfies import decoder
from rdkit.Chem import MolStandardize, Descriptors, MolFromSmiles
from rdkit.Chem.rdMolDescriptors import CalcTPSA
from rdkit.Chem import DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
from multiprocessing import Pool
import logging 

def efcp_to_int(x):
    arr = np.zeros((0,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(x, arr) 
    return arr

def num_atoms_from_smiles(smiles):
    mol  = Chem.MolFromSmiles(smiles)
    return mol.GetNumAtoms()


class PhysChemFilters:
    """
    Applies filters on several physicochemical properties and structural alerts 
    """
    def __init__(self, alerts_file):        
        smarts = pd.read_csv(alerts_file, header=None, sep='\t', encoding='utf-8')[1].tolist()
        self.alert_mols = [Chem.MolFromSmarts(smart) for smart in smarts if Chem.MolFromSmarts(smart) is not None]
        self.range_logP = [-2,7]
        self.range_mw = [250, 750]
        self.max_hba_hbd = 10
        self.max_tpsa = 150
        self.max_nrb = 10
    
    def compute_properties(self, smile_standard):
        mol = Chem.MolFromSmiles(smile_standard)
        try:
            return {
                "logP": np.round(Descriptors.MolLogP(mol), 4),
                "mw": np.round(Descriptors.MolWt(mol), 4),
                "hba": Descriptors.NumHAcceptors(mol),
                "hbd": Descriptors.NumHDonors(mol),
                "tpsa": np.round(CalcTPSA(mol), 4) ,
                "nrb": Descriptors.NumRotatableBonds(mol),
                "alert": np.any([mol.HasSubstructMatch(alert) for alert in self.alert_mols])
            }
        except Exception as e:
            logging.warning(e)
            return None

    def apply_filters(self, smile_standard):
        props = self.compute_properties(smile_standard)
        if props:
            f1 = self.range_logP[0] <props["logP"] < self.range_logP[1]
            f2 = self.range_mw[0] < props["mw"] < self.range_mw[1]
            f3 = props["hba"] + props["hbd"] < self.max_hba_hbd
            f4 = props["tpsa"] < self.max_tpsa
            f5 = props["nrb"] < self.max_nrb
            f6 = props["alert"] == False
            return f1 and f2 and f3 and f4 and f5 and f6
        else:
            return False

def explode_list_columns(df,explode_cols):
    """Own implementation of pandas explode we need an older version of pandas to run the code"""
    no_explode_cols = list(np.setdiff1d( df.columns , explode_cols))
    df = df.set_index(no_explode_cols).apply(lambda x: x.apply(pd.Series).stack()).reset_index()
    return df

def explode_str_columns(df,explode_cols, separator):
    for col in explode_cols:
        df[col] = df[col].apply(lambda x: str(x).split(separator))
    no_explode_cols = list(np.setdiff1d( df.columns , explode_cols))
    df = df.set_index(no_explode_cols).apply(lambda x: x.apply(pd.Series).stack()).reset_index()
    return df

def get_feature_cols(df):
    "Splits columns of input dataframe into columns contining metadata and columns containing morphological profile features"
    feature_cols = [ c for c in df.columns if ("Cells_" in c)|("Nuclei_" in c)|("Cytoplasm_" in c) ]
    info_cols =  list( set(df.columns).difference(set(feature_cols)) )
    return feature_cols, info_cols
    
def clean_smiles(smiles):
    'Standarize molecules and SMILES'
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        mol = MolStandardize.Standardizer().standardize(mol)
        largerFragment = MolStandardize.fragment.LargestFragmentChooser().choose(mol)
        neutral = MolStandardize.charge.Uncharger().uncharge(largerFragment)
        cleaned_smiles = Chem.MolToSmiles(neutral, isomericSmiles=False)
    except:
        cleaned_smiles = None
    return cleaned_smiles

def clean_smiles_parallel(smiles, Nprocessors=50):
    'Standarize molecules and SMILES using a parallel Pool'
    try:
        pool = Pool(Nprocessors) 
        smiles = pool.map( clean_smiles, smiles)
    finally: # To make sure processes are closed in the end, even if errors happen
        pool.close()
        pool.join()
    return smiles


def getSMILES(states_value, decoder_model, transform_model, charset, num_tokens=60):
    'Transforms state to SELFIES'
    states_value = transform_model.predict(states_value)

    target_seq = np.zeros((len(states_value), 1, num_tokens))

    target_seq[:, 0, 1] = 1.


    stop_condition = False
    decoded_sentence = np.ones((len(states_value),1))
   
    while not stop_condition:
        output_tokens, h = decoder_model.predict(
            [target_seq] + [states_value])
        
        # Update states
        states_value = h
        
        #Token to character
        sampled_token_index = [[output_tokens[mol,char].argmax()] for char in range(output_tokens.shape[1])
                               for mol in range(output_tokens.shape[0])]
        #sampled_token_index = [[np.argmax(output_tokens[mol,char] + np.random.gumbel(size=num_tokens))] 
        #                       for char in range(output_tokens.shape[1])
        #                       for mol in range(output_tokens.shape[0])]
        target_seq = np.zeros((len(states_value), 1, num_tokens))
        for i in range(len(target_seq)): target_seq[i, 0, sampled_token_index[i][0]] = 1
        
        # Exit condition: either hit max length
        # or find stop character.
        if (decoded_sentence.shape[1] > 135): stop_condition = True
        else: decoded_sentence = np.append(decoded_sentence, sampled_token_index, axis=1)

    decoded_selfies = [''.join([charset[int(mol[i])] for i in range(1,len(mol[1:])) if all(mol[:i+1] != 0)  ]) 
                       for mol in decoded_sentence]
    return [decoder(m) for m in decoded_selfies]

def generate_scaffold(smiles, generic=False):
    """return scaffold string of target molecule"""
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if generic:
        try:
            generic = MurckoScaffold.MakeScaffoldGeneric(scaffold)            
            scaffold = Chem.MolToSmiles(generic)
        except:
            scaffold = Chem.MolToSmiles(scaffold)
    else: 
        scaffold = Chem.MolToSmiles(scaffold)
    return scaffold

def scafolds_from_smiles_list(smiles_list, generic=False):
    scaffolds = defaultdict(list)
    for ind, smiles in enumerate(smiles_list):
        scaffold = generate_scaffold(smiles, generic=generic)
        scaffolds[scaffold].append(ind)
    scaffolds_df = pd.DataFrame()
    scaffolds_df['scafold'] = list(scaffolds.keys())
    scaffolds_df['indexes'] = list(scaffolds.values())
    scaffolds_df['counts'] = scaffolds_df['indexes'].apply(lambda x: len(x))
    return scaffolds_df

def quantify_common_scafolds_from_smiles(smiles_1, smiles_2, generic=False):
    scaf_df_1 = scafolds_from_smiles_list(smiles_1, generic=generic)
    scaf_df_2 = scafolds_from_smiles_list(smiles_2, generic=generic)
    common_scaf_idx_1 = scaf_df_1.scafold.isin( scaf_df_2.scafold)
    N_comm_scaf_cpds_1 = scaf_df_1.loc[ common_scaf_idx_1 , 'counts'].sum()
    common_scaf_idx_2 = scaf_df_2.scafold.isin( scaf_df_1.scafold)
    N_comm_scaf_cpds_2 = scaf_df_2.loc[ common_scaf_idx_2 , 'counts'].sum()
    Ncommon_scaf  = common_scaf_idx_1.sum()
    return len(scaf_df_1), len(scaf_df_2), Ncommon_scaf, N_comm_scaf_cpds_1, N_comm_scaf_cpds_2

def quantify_common_scafolds(smiles_1, smiles_2, generic=False):
    scaf_df_1 = scafolds_from_smiles_list(smiles_1, generic=generic)
    scaf_df_2 = scafolds_from_smiles_list(smiles_2, generic=generic)
    common_scaf_idx_1 = scaf_df_1.scafold.isin( scaf_df_2.scafold)
    N_comm_scaf_cpds_1 = scaf_df_1.loc[ common_scaf_idx_1 , 'counts'].sum()
    common_scaf_idx_2 = scaf_df_2.scafold.isin( scaf_df_1.scafold)
    N_comm_scaf_cpds_2 = scaf_df_2.loc[ common_scaf_idx_2 , 'counts'].sum()
    Ncommon_scaf  = common_scaf_idx_1.sum()
    return len(scaf_df_1), len(scaf_df_2), Ncommon_scaf, N_comm_scaf_cpds_1, N_comm_scaf_cpds_2

def smile_to_mol(smile):
    "Warper to make it pickable"
    return MolFromSmiles(smile)


