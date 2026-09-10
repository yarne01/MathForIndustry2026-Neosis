# MathForIndustry2026-Neosis


This is a small README.md for the PhD Week of 2026.

In it is the files provided by Neosis, and also the report which is required by KuLeuven.

## Arthur.py

`shape_challenge/Arthur.py` contains helpers for visualizing PCA latent vectors
with UMAP and for exploring their nearest-neighbour structure. The main entry
point is `plot_training_latent_umap_with_clustering`.

### Running the main function

Basically, you need the latent vectors and the original training meshes to call the entry function:

```python
		plot_training_latent_umap_with_clustering(
				training_latent,
				labels=None,
				clustering_method="mst",
				n_neighbors=5,
				umap_neighbors=10,
				min_dist=0.1,
				metric="euclidean",
				show=True,
		)
```

The function returns the Matplotlib `figure` and `axes`, the two-dimensional
UMAP `embedding`, the computed `mst_edges`, and the fitted UMAP `reducer` and
feature `scaler`.

### Input and clustering parameters

- `training_latent`: NumPy-compatible array with shape
	`(n_samples, n_features)` containing the PCA latent vectors.
- `labels`: Optional label for each sample. Labels are added to the UMAP plot.
- `clustering_method`: Clustering method, either `"mst"` or `"hdbscan"`.
- `n_neighbors`: Number of latent-space nearest neighbours used by the MST or
	as the HDBSCAN minimum cluster size. It must be smaller than the number of
	samples.

### UMAP parameters

- `umap_neighbors`: Number of neighbours used by UMAP. It must be between `2`
	and the number of samples.
- `min_dist`: Minimum distance between points in the UMAP embedding.
- `metric`: Distance metric used by UMAP and clustering, such as
	`"euclidean"`.
- `seed`: Random seed for reproducible UMAP results. The default is `42`.

### MST pruning parameters

These parameters affect the MST when `clustering_method="mst"`:

- `pruning_method`: `None`, `"threshold"`, `"shortest_percent"`, or `"iqr"`.
- `pruning_value`: Edge-length threshold for `"threshold"`, or percentage of
	shortest edges to keep for `"shortest_percent"`.
- `iqr_multiplier`: Multiplier used for the upper IQR fence when
	`pruning_method="iqr"`. The default is `1.5`.

### Cluster comparison parameters

- `compare_clusters`: Enables both MST and HDBSCAN cluster-center comparisons.
- `compare_mst`: Enables the MST center comparison independently.
- `compare_hdbscan`: Enables the HDBSCAN center comparison independently.
- `center_n_neighbors`: Number of original samples selected near each cluster
	center.
- `selected_center`: Optional zero-based center index to inspect exclusively.
- `source_meshes`: Original meshes aligned with `training_latent`.
- `show_meshes`: Displays the selected meshes in a coloured 3D grid when
	`True` and `source_meshes` is provided.

### Display and output parameters

- `title`: Plot title. The `{clustering_method}` placeholder is replaced with
	the selected clustering method.
- `case`: Optional case name used in saved filenames.
- `save_folder`: Optional directory where the UMAP PNG and comparison HTML
	files are saved. The directory is created when needed.
- `show`: Displays the figures when `True`. Figures are still saved when
	`save_folder` is provided.

When comparisons are enabled, the function can save the final UMAP image and
interactive MST/HDBSCAN Plotly HTML files in `save_folder`.



