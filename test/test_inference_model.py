"""  
Test molecular autoencoder: Compute molecular embeddings on a small set of standardized smiles and compare the resulting latents to a reference file
"""
# %%
import os
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
tf.compat.v1.logging.set_verbosity (tf.compat.v1.logging.ERROR)

import cpmolgan.inference as infr
import cpmolgan.utils as utils
import pkg_resources
WEIGHTS_PATH = pkg_resources.resource_filename('cpmolgan','model_weights')

# Test will be run in the CPU by default
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# %% Load model 
model_weigth_paths={
    'autoencoder': os.path.join(WEIGHTS_PATH,'autoencoder.h5'),
        'wgan':{
            'C': os.path.join(WEIGHTS_PATH,'gan_C.h5'),
            'D': os.path.join(WEIGHTS_PATH,'gan_D.h5'),
            'G': os.path.join(WEIGHTS_PATH,'gan_G.h5'),
            'condition_encoder':os.path.join(WEIGHTS_PATH,'gan_condition_encoder.h5'),
            'classifier':os.path.join(WEIGHTS_PATH,'gan_classifier.h5')
        }
}
model = infr.InferenceModel(model_weigth_paths) 

# %% Test 1: Molecular autoencoder
smiles =  pd.read_csv("./autoencoder_input_smiles.csv").SMILES_standard.values.astype(str)
selfies, _ = model.encode_smiles_to_selfies(smiles)
latents = model.encode_selfies_to_latent(selfies)

latents_ref = np.load('autoencoder_output_latents.npy')
ref_correlation = np.corrcoef(latents.flatten(),latents_ref.flatten())[0,0]
max_diff = np.abs((latents-latents_ref)).max()
latents_match = (ref_correlation>0.99) & (max_diff<1e-4)

# %% Test 2: GAN

profiles = pd.read_csv('../test/GAN_input_condition_profiles.csv')
quantile_transformer =  pickle.load( open( os.path.join(WEIGHTS_PATH,'quantile_transformer.pkl'), 'rb' ) ) 
feature_cols , meta_cols = utils.get_feature_cols(profiles)
profiles[feature_cols] = quantile_transformer.transform(profiles[feature_cols].values) 
generated = infr.generate_compounds_multiple_conditions( model, profiles, feature_cols, meta_cols, seed=10, nsamples=2)

generated_ref = pd.read_csv('GAN_output_smiles.csv')
smiles_match = (generated['SMILES'] != generated_ref['SMILES']).sum() ==0
ref_correlation = np.corrcoef(generated['classification_score'].values, generated_ref['classification_score'])[0,0]
max_diff = np.abs((generated['classification_score'].values-generated_ref['classification_score'].values)).max()
class_scores_match =  (ref_correlation>0.99) & (max_diff<1e-4)

# %% Report results 

if latents_match:
    print("\nAutoencoder test: SUCCESSFUL \n\tLatent representations match reference latents")
else:
    print("\nAutoencoder test: FAILED \n\tLatent representations don't match reference latents")

if smiles_match & class_scores_match:
    print('\nGAN test: SUCCESSFUL\n\tGenerated SMILES and classification scores match reference values')
else:
    print('\nGAN test: FAILED')
    if not smiles_match:
        print("\tGenerated SMILES  dont't match reference SMILES")
    if not class_scores_match:
        print("\tClassification scores dont't match reference scores\n")


# %%
