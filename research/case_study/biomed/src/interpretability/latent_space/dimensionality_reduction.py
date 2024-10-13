import numpy as np
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def apply_pca(z, z_gen, n_components=2):
    pca = PCA(n_components=n_components)
    z_pca = pca.fit_transform(z)
    z_gen_pca = pca.transform(z_gen)
    return z_pca, z_gen_pca

def apply_tsne(z, z_gen, n_components=2, perplexity=30, learning_rate=200, n_iter=1000, random_state=42, **kwargs):
    combined_data = np.vstack((z, z_gen))
    tsne = TSNE(n_components=n_components, perplexity=perplexity, learning_rate=learning_rate, n_iter=n_iter, random_state=random_state, **kwargs)
    combined_tsne = tsne.fit_transform(combined_data)
    
    # Separate the results
    z_tsne = combined_tsne[:len(z)]
    z_gen_tsne = combined_tsne[len(z):]
    
    return z_tsne, z_gen_tsne

def apply_umap(z, z_gen, n_components=2, n_neighbors=15, min_dist=0.1, random_state=42, fit_on_combined=True, **kwargs):
    if fit_on_combined:
        # Combine the data and fit UMAP
        combined_data = np.vstack((z, z_gen))
        umap_reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state, **kwargs)
        combined_umap = umap_reducer.fit_transform(combined_data)
        
        # Separate the results
        z_umap = combined_umap[:len(z)]
        z_gen_umap = combined_umap[len(z):]
    else:
        # Fit on z and transform z_gen
        umap_reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state, **kwargs)
        z_umap = umap_reducer.fit_transform(z)
        z_gen_umap = umap_reducer.transform(z_gen)
    
    return z_umap, z_gen_umap

# # CHECK CHANGE!
# def apply_tsne(z, z_gen, n_components=2, perplexity=30, learning_rate=200, n_iter=1000, random_state=42, **kwargs):
#     tsne = TSNE(n_components=n_components, perplexity=perplexity, learning_rate=learning_rate, n_iter=n_iter, random_state=random_state, **kwargs)
#     z_tsne = tsne.fit_transform(z)
#     z_gen_tsne = tsne.fit_transform(z_gen)
#     return z_tsne, z_gen_tsne

# def apply_umap(z, z_gen, n_components=2, n_neighbors=15, min_dist=0.1, random_state=42, **kwargs):
#     umap_reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state, **kwargs)
#     z_umap = umap_reducer.fit_transform(z)
#     z_gen_umap = umap_reducer.transform(z_gen)
#     return z_umap, z_gen_umap