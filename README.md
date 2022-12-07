# CPMolGAN: 
Supporting code to run inference experiments from the publication "Cell morphology-guided de novo hit design by
conditioning GANs on phenotypic image features" by Paula A. Marin-Zapata and Oscar Méndez-Lucio et al. (2022) https://doi.org/10.1039/D2DD00081D

# Installation

The CPMolGAN model requires Python 3.6, TensorFlow 1.15, selfies 1.0.4, and rdkit 2018.09.1. We provide an environment file for connivent installation using conda (`environment.yml`). For reproducibility, we also provide a more detailed environment including all dependencies of our development setup (`environment_detailed.yml`). Follow the instructions below to install cpmolgan:

Clone repository:   
`git clone https://github.com/Bayer-Group/CPMolGAN.git`  
`cd CPMolGAN`

Download pre-trained weights.  **Model weights are available for non-commercial use only**. (The model weights are fitted on **publicly available data** only)   
Download them directly from [Google Drive](https://drive.google.com/file/d/1KgVTglQ_LbKzzEkhGzEdZxBA9RBTXIFl/view?usp=sharing) and place them inside the directory `cpmolgan/model_weights`, or by executing the bash script:   
`chmod +x download_model_weights.sh`  
`./download_model_weights.sh`

Create conda environment:  
`conda env create -f environment.yml`

Activate your environment:  
`conda activate cpmolgan`

Install tensorflow without GPU support:  
`conda install tensorflow=1.15.0`

or with GPU support:  
`conda install tensorflow-gpu=1.15.0`

Install this project as python package (in editable mode `-e` if desired):    
 `pip install -e .`

Finally, install the environment kernel to run jupyter notebooks:
`python -m ipykernel install --user --name cpmolgan`

# Testing
Test the inference model by running the Python script in the test folder with your activated environment. A successful test will print one confirmation message for the autoencoder and one for the GAN.  
`cd test`  
`python test_inference_model.py`  

The molecular autoencoder is tested by computing molecular embeddings for the set of smiles in `test/autoencoder_input_smiles.csv` and comparing against a reference embeddings file (`test/autoencoder_output_latents.npy`). The GAN is tested by generating molecules conditioning of a set of reference morphological profiles (`test/GAN_input_condition_profiles.csv`) and comparing against reference generated molecules and classification scores in `test/GAN_output_smiles.csv`

# Data
This project only utilized **publicly available** data (as listed below) to train the machine learning algorithm.  
Data can be downloaded by executing the bash script:  
`chmod +x download_data.sh`  
`./download_data.sh`  

Alternatively, data is available for download from [Google Drive](https://drive.google.com/drive/u/0/folders/1o9H5V1B7xDuJaX4mU1rtXfzoUKvTiGNE). Make sure to unzip and place all files inside the `data` directory in order to run the scripts in the `publication_notebooks` directory.

The following files are provided:
- `ChEMBL_standardized_smiles.csv`: Training set for the molecular autoencoder with standardized SMILES
- `train_set_30kcpds_normalized_profiles.csv.gz`: Training set for the GAN with normalized well profiles for ~30,000 compounds and standardized SMILES. Downloaded from [gigadb.org](http://gigadb.org/dataset/view/id/100351/File_page/1). 
- `test_set_overexpression_normalized_profiles.csv`: Well profiles from gene over-expression perturbations used as test set. Illumination corrected images were downloaded from [idr.openmicroscopy](https://idr.openmicroscopy.org/webclient/?show=screen-1751) using a web client. Profiles were extracted with the [analysis.cppipe](https://github.com/gigascience/paper-bray2017/tree/master/pipelines) CellProfiler pipeline provided in [1].
- `ExCAPE_ligands_agonist_noTrainSet.csv`: Agonists towards the genes in the over-expression test set which were found in the ExCAPE database and are not part of the training set.
- `sure_chembl_alerts.txt`: sure ChEMBL alerts used to filter generated molecules with desired physicochemical properties

# Getting Started
The examples directory provides notebooks exemplifying the usage of the pre-trained inference model:
- `examples/run_molecular_autoencoder_inference.ipynb`: shows how to use the autoencoder model to compute molecular representations from SMILES. It also summarizes preprocessing steps applied to SMILES strings before being feed to the model.
- `examples/run_GAN_inference.ipynb`: shows how to use the GAN model to generate SMILES conditioning on morphological profiles. It also summarizes preprocessing steps applied to morphological profiles before being feed to the model.

# LICENSE
Model parameters are available for non-commercial use only, under the terms of the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) [license](https://creativecommons.org/licenses/by-nc/4.0/legalcode).  
Code is available under BSD 3-Clause License.

For using the publicly available data as described in #Data, we refer to the original licenses of the data owners.  
- `ChEMBL_standardized_smiles.csv`: The ChEMBL data is made available on a [Creative Commons Attribution-Share Alike 3.0 Unported License](https://creativecommons.org/licenses/by-sa/3.0/).
- `train_set_30kcpds_normalized_profiles.csv.gz`: Datasets associated with GigaScience journal articles are specifically released under a [Creative Commons CC0 waiver](http://gigadb.org/site/term) unless stated otherwise.
- `test_set_overexpression_normalized_profiles.csv`: CC BY 4.0 https://creativecommons.org/licenses/by/4.0/.
- `ExCAPE_ligands_agonist_noTrainSet.csv`: Creative Commons Public Domain Dedication waiver (http://creativecommons.org/publicdomain/zero/1.0/)
- `sure_chembl_alerts.txt`: See https://www.surechembl.org/terms/

# Reproducing publication results
The directory `publication_notebooks` provides a comprehensive set of notebooks to reproduce most publication results and figures. 

# References
**Cell morphology-guided *de novo* hit design by conditioning GANs on phenotypic image features**  
P. A. Marin Zapata, O. Méndez-Lucio, T. Le, C. J. Beese, J. Wichard, D. Rouquié and D. Clevert,  
*Digital Discovery*, 2023, Advance Article , [**DOI: 10.1039/D2DD00081D**](https://doi.org/10.1039/D2DD00081D)
