from rdkit import Chem
from rdkit.Chem import Draw, MolFromSmiles
from mhfp.encoder import MHFPEncoder
import tmap as tm
from tqdm import tqdm
from faerun import Faerun
from PIL import Image
import base64
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pickle

config = tm.LayoutConfiguration()
config.node_size = 1 / 26
config.mmm_repeats = 2
config.sl_extra_scaling_steps = 5
config.k = 20
config.sl_scaling_type = tm.RelativeToAvgLength


top_10 = ['RAF1', 'JUN', 'ATF4', 'BRAF', 'CEBPA', 'RELB', 'MEK1', 'PIK3CD','AKT3', 'WWTR1']
Excape_genes = ["TP53","BRCA1","NFKB1","HSPA5", "CREBBP", "STAT1", "STAT3","HIF1A", "NFKBIA","JUN","PRKAA1","PDPK1"]
gene_colors_dict ={
    'DMSO': [0,0,0]
}
for gene in top_10+Excape_genes:
    gene_colors_dict[gene]=[0,1,1]

def save_coordinates(x, y, s, t, filename):
    x = list(x)
    y = list(y)
    s = list(s)
    t = list(t)
    pickle.dump( (x, y, s, t), open(filename, "wb+"), protocol=pickle.HIGHEST_PROTOCOL )
    
def load_coordinates(filename):
    x, y, s, t = pickle.load(open(filename, "rb"))
    return x, y, s, t


def lsh_forest_from_smiles( smiles, fp_length = 1024):
    enc = MHFPEncoder(fp_length)
    lf = tm.LSHForest(fp_length, 64)
    fps = []
    for smile in tqdm(smiles):
        mol = Chem.MolFromSmiles(smile)
        fps.append(tm.VectorUint(enc.encode_mol(mol)))
    lf.batch_add(fps)
    lf.index()
    return lf 

def load_lsh_forest( filename, fp_length = 1024):
    enc = MHFPEncoder(fp_length)
    lf = tm.LSHForest(fp_length, 64)
    lf.restore(filename)
    return lf


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

def plot_tree(df,x,y,s,t, html_output_file="coil", colors=None, smiles_pic_col=None, plot_name='tmap', max_point_size=5):
    
    # Create Pictures for each molecule
    smiles_pics = []
    if smiles_pic_col:
        print('Generating compound pictures')
        for smiles in df[smiles_pic_col]:
            img = Image.fromarray(np.array(Draw.MolToImage(Chem.MolFromSmiles(smiles))))
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue())
            smiles_pics.append(
                "data:image/bmp;base64," + str(img_str).replace("b'", "").replace("'", "")
            )
    
    labels = df[['label','label_name']].drop_duplicates()
    legend_labels = [ (l,n) for l,n in zip(labels.label, labels.label_name)]
    
    if colors ==None:
        colors = get_diverse_colors(len(legend_labels))
    colormap = ListedColormap(colors)
    print(colormap)
    
    #faerun = Faerun(clear_color="#111111", view="front", coords=False) 
    faerun = Faerun(clear_color='#ffffff', view="front", coords=False)

    faerun.add_scatter(
        plot_name,
        {"x": x, "y": y, "c": df['label'], "labels":smiles_pics},
        colormap=colormap,
        shader="smoothCircle",
        point_scale=100,
        max_point_size=max_point_size,
        has_legend=True,
        categorical=True,
        legend_labels=legend_labels
    )
    faerun.add_tree(
        "tree", {"from": s, "to": t}, point_helper=plot_name, color="#666666"
    )

    faerun.plot(html_output_file, template="url_image")

def smile_to_mol(smile):
    "Warper to make it pickable"
    return MolFromSmiles(smile)