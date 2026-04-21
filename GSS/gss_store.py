import scanpy as sc
import numpy as np
import scipy.sparse as sp
from pathlib import Path

adata = sc.read_h5ad('adata_with_gss.h5ad')

print(adata.obsm['spatial'])

gss = adata.layers['GSS']
if not sp.isspmatrix_csr(gss):
    gss = sp.csr_matrix(gss)

top5_with_expr = {}

n_spots, n_genes = gss.shape

for i in range(n_spots):
    row = gss.getrow(i).toarray().ravel()

    idx5 = np.argpartition(-row, 5)[:5]

    idx5 = idx5[np.argsort(-row[idx5])]

    genes = adata.var_names[idx5]

    if sp.issparse(adata.X):
        exprs = adata.X.getrow(i).toarray().ravel()[idx5]
    else:
        exprs = adata.X[i, idx5]

    top5_with_expr[i] = {g: float(v) for g, v in zip(genes, exprs)}

out = Path('top5_gss_with_expression.txt')
with open(out, 'w') as f:
    for spot, gene_dict in top5_with_expr.items():
        f.write(f"{spot}: {gene_dict}\n")

print(f"Done! Saved to {out.resolve()}")
