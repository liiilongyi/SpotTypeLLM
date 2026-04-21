import scanpy as sc
import numpy as np
import scipy
from scipy.stats import gmean, rankdata
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from tqdm import trange

# 1. Load your AnnData object
adata = sc.read_h5ad('adata_RNA.h5ad')

# 2. Ensure X is CSR sparse matrix
if not scipy.sparse.issparse(adata.X):
    adata.X = scipy.sparse.csr_matrix(adata.X)
elif not isinstance(adata.X, scipy.sparse.csr_matrix):
    adata.X = adata.X.tocsr()

# 3. Extract spatial and latent embeddings
spatial_coords = adata.obsm['spatial']  # shape (n_spots, 2)

latent = np.load('adata_RNA_embedding.npy')
assert latent.shape[0] == adata.n_obs, "The number of rows in the embedding must match adata.n_obs."
adata.obsm['latent_embedding'] = latent.astype(np.float32)
latent_coords = adata.obsm['latent_embedding']

# 4. Build spatial neighbor dictionary
num_spatial_neighbors = 100  # adjust as desired (e.g., 101 for Stereo-seq)
nbrs = NearestNeighbors(n_neighbors=num_spatial_neighbors).fit(spatial_coords)
_, idx_spatial = nbrs.kneighbors(spatial_coords)
spatial_net = {i: idx_spatial[i] for i in range(adata.n_obs)}

print("# 5. Prepare rank matrix and global geometric mean")
n_cells, n_genes = adata.n_obs, adata.n_vars
# Compute per-spot ranks
ranks = np.zeros((n_cells, n_genes), dtype=np.float32)
for i in range(n_cells):
    values = adata.X[i].toarray().ravel()
    ranks[i] = rankdata(values, method='average')

# Global geometric mean of ranks
gM = gmean(ranks, axis=0)
# Standardize ranks
ranks /= gM

print("# 6. Compute global expression fractions")
X_bool = adata.X.astype(bool)
frac_whole = np.asarray(X_bool.sum(axis=0)).ravel() / n_cells
frac_whole += 1e-12  # avoid division by zero

print("# 7. Define function to get microdomain based on latent similarity")
def get_microdomain(cell_idx, d=50):
    neighbors = spatial_net[cell_idx]
    sims = cosine_similarity(latent_coords[cell_idx:cell_idx+1], latent_coords[neighbors]).flatten()
    top_k = neighbors[np.argsort(-sims)[:d]]
    return top_k

print("# 8. Compute GSS (marker scores)")
d = 50  # microdomain size (20 for Visium, 50 for Stereo-seq)
gss = np.zeros((n_cells, n_genes), dtype=np.float32)
for i in trange(n_cells, desc="Calculating GSS"):
    dom = get_microdomain(i, d=d)
    if dom.size == 0:
        continue
    # Geometric mean of standardized ranks in microdomain
    region_ranks = gmean(ranks[dom], axis=0)
    region_ranks[region_ranks <= 1] = 0
    # Expression fraction filter
    frac_dom = X_bool[dom].sum(axis=0).A1 / dom.size
    frac_ratio = frac_dom / frac_whole
    frac_ratio[frac_ratio <= 1] = 0
    frac_ratio[frac_ratio > 1] = 1
    # Combined specificity
    spec = region_ranks * frac_ratio
    # Exponential projection
    gss[i] = np.exp(spec) - 1
    #print(gss[i])

print("# 9. Store results in a DataFrame and save")
gss_df = pd.DataFrame(gss, index=adata.obs_names, columns=adata.var_names)
print(gss_df)

print("Adding GSS layer to AnnData and writing .h5ad…")
adata.layers['GSS'] = scipy.sparse.csr_matrix(gss)

print(adata.layers['GSS'])

adata.write_h5ad('adata_with_gss.h5ad')

