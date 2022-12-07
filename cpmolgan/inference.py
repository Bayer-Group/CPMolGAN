import numpy as np
import pandas as pd
from re import split
import selfies
import logging
from tqdm import tqdm
from cpmolgan import utils
from cpmolgan.models import build_encoder_decoder
from cpmolgan.models import GANwCondition


# Global variables for hardcoded model parameters
latent_dim = 256
noise_dim = 1000  
condition_random_seed = 42
max_smiles_length = 120 ## set because we don't want to have 'larger' compounds...
max_selfie_lenght = 131 ## note this is different from the max SMILES length and set as 131 for the VAE, which takes as input (n_batch, max_selfie_lenght, len(charset)). Largest SELFIES representation consists of 131 tokens, hence we set max_selfie_lenght=131

charset = ['[END]', '[START]', '[#C]', '[#N+]', '[#N]', '[-c]', '[-n+]', '[-n]', '[=C]', '[=N+]', 
          '[=N-]', '[=N]', '[=O]', '[=P]', '[=S+]', '[=S]', '[=c]', '[As]', '[B-]', '[B]', '[Br]', 
          '[Branch1_1]', '[Branch1_2]', '[Branch1_3]', '[Branch2_1]', '[Branch2_2]', '[Branch2_3]', 
          '[C-]', '[C]', '[Cl]', '[Expl-Ring1]', '[Expl-Ring2]', '[Expl=Ring1]', '[F]', '[H]', '[I]', 
          '[N+]', '[N-]', '[N]', '[O-]', '[O]', '[P+]', '[PH]', '[P]', '[Ring1]', '[Ring2]', '[S+]', 
          '[S-]', '[S]', '[Se]', '[Si]', '[c]', '[epsilon]', '[n+]', '[nH]', '[n]', '[o]', '[s+]', '[s]', '[se]']

def generate_compounds_multiple_conditions(model, data, feature_cols, info_cols, nsamples=1000, seed=condition_random_seed):
    """
    Wraper to generate and format molecules from multiple conditions from the input data
    :param model: InferenceModel instance with trained weights
    :param data: DataFrame containing the condition feature vectors plus arbitrary information colums
    :param feature_cols: list of columns corresponding to condition vectors
    :param info_cols: all other columns 
    :param nsamples: number of samples to be generated per condition
    :param seed: numpy random seed for the generation process
    :return: DataFrame with generated smiles and respective classification scores plus the set of info_cols
    """
    conditions = data[feature_cols].values
    generated = data[info_cols].copy()      
    generated["SMILES"], generated["classification_score"]= [ "", np.nan]
    generated = pd.concat([generated]*nsamples).sort_index() # duplicate each row nsamples times keepeing the original index

    for idx in tqdm(range(conditions.shape[0])):
        condition = conditions[idx,:].reshape(1, len(feature_cols))        
        smiles,  scores = model.generate_compounds_with_condition(condition=condition, nsamples=nsamples, seed=seed) 
        if nsamples==1: 
            smiles = smiles[0]
            scores = scores[0] # Avoids pandas error
        generated.loc[idx,"SMILES"] = smiles
        generated.loc[idx,"classification_score"] = scores
    generated = generated.reset_index(drop=True)  
    
    return generated


def filter_valid_and_unique(data: pd.DataFrame,
                            cond_ID_cols: list=[], 
                            select_unique: bool=True):
    """
    Removes unvalid molecules and selects a set of unique molecules PER CONDITION
    :param data: DataFrame with list of molecules on the column "SMILES"
    :cond_ID_cols: set of columns which uniquely identify the condition used to generate the molecules
        Default (empty list) selects unique among all data disregarding the condition
    """
    data["SMILES_standard"]= utils.clean_smiles_parallel( data.SMILES )
    valid_idx = data.SMILES_standard.isnull()==False
    data = data.loc[valid_idx].reset_index(drop=True)
    if select_unique:
        data = data.drop_duplicates( subset = cond_ID_cols + ["SMILES_standard"]).reset_index(drop=True)
    return data

def split_selfies_char_into_list(selfies: str) -> list:
    """
    Splits a Selfies string with its characters into a list of each character and adds the START and END token to the 
    beginning and the end of the list.
    """
    splits_elements = split('(\[[^\]]*\])', selfies)
    ## only select elements where string is not empty
    splits_elements = [s for s in splits_elements if s!=""]
    selfies_token_list = ['[START]'] + splits_elements + ['[END]'] 
    return selfies_token_list
    
    
class InferenceModel:
    """
    End-to-end inference model based on pre-trianed weights. The end-to-end model brings together a selfies-to-selfies variational autoencoder and a conditioned wgan. Smiles are automatically translated to selfie representations. Thus, all methods take smiles as inputs and produce smiles as outputs. 
    """
    
    def __init__(self, weight_paths:dict(), charset:list=charset, max_smiles_length:int=max_smiles_length, verbose:bool=False ):
        """
        :param weight_paths: dictionary with file paths to all pre-trained weights stored as .h5 files. The dictionary shoudl have the follwoing structure:
            {
            'autoencoder': "filepath",
            'wgan':{
                'C': "filepath",
                'D': "filepath",
                'G': "filepath",
                'condition_encoder':"filepath",
                'classifier':"filepath"
            }
        :param charset: set of SMILE characters to be considered
        :param: max_smiles_length: maximum length of input smiles. Smiles shorter than "max_smiles_length" will be padded to this length to achieve a fixed input size in the autoencoders's encoder. Smiles longer than "max_smiles_length" will be automatically discarded by self.encode_smiles_to_latent()
        """
        
        self.get_smiles = utils.getSMILES
        self.charset = charset
        self.num_tokens = len(self.charset)
        self.max_smiles_length = max_smiles_length
        self.max_selfie_lenght = max_selfie_lenght
        self.token_index = dict([(char, i) for i, char in enumerate(self.charset)])
     
        # Reference to selfies decoder function
        self.selfies_decoder = selfies.decoder
        self.selfies_encoder = selfies.encoder
        
        # Initialize the VAE model
        _, self.encoder_model, self.transform_model , self.decoder_model=build_encoder_decoder(num_tokens= self.num_tokens,
                                                            latent_dim=latent_dim, 
                                                            weights=weight_paths["autoencoder"], 
                                                            verbose=verbose)
    
        # Initialize wgan model
        self.wgan = GANwCondition( latent_dim=latent_dim, noise_dim=noise_dim, verbose=verbose)
        self.wgan.C.load_weights(weight_paths["wgan"]["C"])
        self.wgan.D.load_weights(weight_paths["wgan"]["D"])
        self.wgan.G.load_weights(weight_paths["wgan"]["G"])
        self.wgan.condition_encoder.load_weights(weight_paths["wgan"]["condition_encoder"])
        self.wgan.classifier.load_weights(weight_paths["wgan"]["classifier"])
        
    
    def encode_smiles_to_latent(self, arr: list or np.ndarray) -> np.array:
        """
        Encodes a list of SMILES into the latent space
        """
        # encode the smiles into selfies
        selfies_list = [self.selfies_encoder(m) for m in arr if len(m) < self.max_smiles_length]
        selfies_list = [mol for mol in selfies_list if (mol is not None and all(t in self.charset for t in split_selfies_char_into_list(mol)))]
                                                        
        if len(selfies_list)<len(arr):
            logging.warning('latent econding for %i input smiles was skipped because the smiles werer either larger than the allowed maximum (%i) or contained characters out of the standard character set.'%(len(arr)-len(selfies_list), max_smiles_length))
            
        # zero pad and tokenize the selfies into one-hot-encoding of shape (n_batch, max_smiles_length, num_tokens)
        encoder_input_tokens = np.asarray([split_selfies_char_into_list(mol) for mol in selfies_list])
        encoder_input_tokens = np.array([np.array([self.token_index[char] for char in mol], dtype=int)
                                 for mol in encoder_input_tokens])
        encoder_input_data = np.zeros((len(encoder_input_tokens), self.max_selfie_lenght, self.num_tokens), dtype='float32')

        for i in range(len(encoder_input_tokens)):
            num_chars = len(encoder_input_tokens[i])
            encoder_input_data[i][np.arange(num_chars), encoder_input_tokens[i]] = 1.
            
        latent_representation = self.encoder_model.predict(encoder_input_data)
        
        return latent_representation
    
    
    def decode_latent_to_smiles(self, arr: list or np.ndarray) -> list:
        """
        Decodes a (list) of latent vectors into a (list of) string SMILES representation(s).
        Note, no canonicalization or validity check of the SMILES is done.
        """
        decoded_smiles = utils.getSMILES(arr, decoder_model=self.decoder_model, transform_model=self.transform_model,
                                   charset=self.charset, num_tokens=self.num_tokens)
        
        return decoded_smiles
    
    
    def generate_compounds_with_condition(self, condition: np.ndarray, nsamples: int = 1000, seed: int = condition_random_seed ):
        """
        Generated compounds that satisfy the condition
        """
        np.random.seed(seed)
        sampled_latent_space = self.wgan.G.predict([np.random.normal(0, 1, (nsamples, noise_dim)),np.repeat(condition, nsamples, axis=0)])
        decoded_smiles = self.decode_latent_to_smiles(sampled_latent_space)
        
        # Get classification values for compounds and conditions
        try:
            classification_scores = self.wgan.classifier.predict([sampled_latent_space, np.repeat(condition, nsamples, axis=0)])
        except Exception as e:
            logging.warning("error in computation classification scores. returning None.\n"+format(e))
            classification_scores = [None] * len(decoded_smiles)
            
        return decoded_smiles, classification_scores


    def encode_smiles_to_selfies(self, arr: list or np.ndarray) -> np.array:
        """
        Encodes a list of SMILES into the latent space keeping track of which SMILES are removed by the filters
        """
        # encode the smiles into selfies
        valid_idx = []
        selfies = []
        for m in arr:
            valid_length = len(m) < self.max_smiles_length
            selfie = self.selfies_encoder(m)
            valid_mol = selfie is not None 
            valid_chars = all(t in self.charset for t in split_selfies_char_into_list(selfie))
            is_valid = valid_length & valid_mol & valid_chars
            selfies.append( selfie )
            valid_idx.append(is_valid)
            
        return np.array(selfies), np.array(valid_idx)
    
    def encode_selfies_to_latent(self, selfies_list: list or np.ndarray) -> np.array:
        """
        Encodes a list of Selfies into the latent space
        """
        # zero pad and tokenize the selfies into one-hot-encoding of shape (n_batch, max_smiles_length, num_tokens)
        encoder_input_tokens = np.asarray([split_selfies_char_into_list(mol) for mol in selfies_list])
        encoder_input_tokens = np.array([np.array([self.token_index[char] for char in mol], dtype=int)
                                 for mol in encoder_input_tokens])
        encoder_input_data = np.zeros((len(encoder_input_tokens), self.max_selfie_lenght, self.num_tokens), dtype='float32')

        for i in range(len(encoder_input_tokens)):
            num_chars = len(encoder_input_tokens[i])
            encoder_input_data[i][np.arange(num_chars), encoder_input_tokens[i]] = 1.
            
        latent_representation = self.encoder_model.predict(encoder_input_data)
        
        return latent_representation