# Molecular generation from clustered training set profiles
Collection of scripts to reproduce all results from publication Figure 2. Including profiles clustering, molecular generation and figures. Notebooks should be run sequentially as indicated by the digit prefix. All output files will be stored in the results folder, including generated molecules, molecular embeddings, nearest neighbors and tmap projections.

**Note**: the tmap visualization script `07__tmap_projections_generated_mols.ipynb` requires a different environment (`environment_tmap.ym`) due to version incompatibility between pandas, rkdkit and faerun. Install the tmap visualization environment by running:   
`conda env create -f environment_tmap.yml`  
`conda activate cpmolgan_tmap`  
`cd ../..`  
 `pip install -e .`  