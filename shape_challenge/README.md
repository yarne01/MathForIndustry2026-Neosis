# Repository Structure

The repository is organized as follows:

```text
.
├── test_files/
│   ├── Blended/
│   ├── Fan/
│   ├── Sokaris/
│   └── Tesla_valve/
│
├── results/
│   ├── images/
│   ├── jsons/
│   ├── pcas/
│   └── sdfs/
│
├── sdf_pca_for_KUL.py

├── generic_utils.py
├── geometry_utils.py
├── sdf_pca_utils.py
├── sdf_plot_utils.py
|
└── README.md
```

## Folder Description

### `test_files/`

This folder contains the datasets used to test and demonstrate the workflow. Each subfolder represents a complete use case, including the input geometry and any associated files required by the analysis.

The currently available examples are:

- **Blended**
  - *Description:* blended wing body, 100 geometries, 10 design variables. Used in external CFD. Source material available at: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/VJT9EP

- **Fan**
  - *Description:* 5-bladed fan, number of fans fixed. No known design variables for reference. Used in external CFD. 50 geometries.

- **Sokaris**
  - *Description:* hybrid architectural element. Truss-like structure, 9 design variables. 40 geometries.

- **Tesla_valve**
  - *Description:* Internal CFD case, 7 design variables. 24 geometries.

Additional test cases can be added by following the same directory structure. 


## Source Files

### Main application

| File | Purpose |
|------|---------|
| **sdf_pca_for_KUL.py** | Executes the complete workflow described above. It loads the dataset, computes SDFs, performs PCA, analyses the latent space and generates new candidate geometries. |

# Key parameters:
- Line 80: defines the current testcase
- Line 98: load_project allow to reload existings PCAs saved as JSON
- Line 104: generate_images if True, during the loading of the geometries will generate and save images for all the geometries from the three axis viewpoints.
- Line 105: reduced_exp_number, can be used to load a limited number of geometries from the dataset.
- Line 109: debug_sdf, generates slice by slice SDF plot
- Line 121: variance_threshold required representiveness used both in plots and to determine number of modes to be preserved and used
- Lines 128, 129: if evaluate_reconstruction is True, n_reconstructions random experiments from the training dataset (reduced form) are selected and projected back.
- Lines 136 to 155: if evaluate_new_samples is True, n_new_samples random experiments are removed from the training dataset and used for validation. 
If instead evaluate_specified_new_samples is True, depending on the case previously identified experiments are removed. Selected experiments were either the most dissimilar of the most universal.
- Lines 176 to 180: if the specific approach flag is True, generate_new_samples new geometries will be created using that approach.


## Workflow

The project follows the workflow below.

```
Dataset
   │
   ▼
Load geometries (STL)
   │
   ▼
Evaluate global bounding box
   │
   ▼
Generate common Cartesian grid
   │
   ▼
Compute Signed Distance Fields (SDF)
   │
   ▼
Apply PCA
   │
   ▼
Analyse latent space
   │
   ├── Variance analysis
   ├── Distance matrices
   ├── Cluster analysis
   ├── SOM visualization
   ├── UMAP visualization
   ├── Similarity search
   └── New geometry generation
```

The execution is divided into the following numbered stages.

| Step | Description |
|------|-------------|
| **0** | Load dataset configuration, experiment metadata and STL geometries. |
| **1** | Compute the global bounding box and generate a common Cartesian grid used for all Signed Distance Fields. |
| **2** | Compute the Signed Distance Field (SDF) representation for every geometry in the dataset. |
| **3** | Perform Principal Component Analysis (PCA) on the SDF dataset to obtain a compact latent representation. |
| **4** | Analyse explained variance and determine the number of principal components required to reach the target reconstruction accuracy. |
| **5** | *(Optional)* Visualize the contribution of individual PCA modes. |
| **6** | *(Optional)* Reconstruct geometries from the latent space to evaluate PCA reconstruction quality. |
| **7** | Compute pairwise distances in latent space, generate distance matrices and visualize the PCA embedding. |
| **8** | Compare new geometries against the training dataset, identify the closest existing designs and evaluate similarity metrics. |
| **9** | Identify the most isolated geometries in the latent space. |
| **10** | Identify the most representative (least isolated) geometries. |
| **11** | Visualize the distribution of the dataset in PCA space. |
| **12** | Train and visualize a Self-Organizing Map (SOM) representation of the latent space. |
| **13** | Build UMAP embeddings to visualize the latent manifold while preserving neighbourhood relationships. |
| **14** | Generate new candidate geometries by sampling or interpolating within the latent space and reconstruct them back into geometry. |


---

### Generic utilities

```
generic_utils.py
```

Reusable helper functions for:

- file handling
- JSON I/O
- folder management
- logging
- miscellaneous utilities

### Geometry

```
geometry_utils.py
```

Utilities for STL geometries:

- loading meshes
- bounding-box computation
- mesh preprocessing
- geometric helper functions

---

### Signed Distance Fields & PCA

```
sdf_pca_utils.py
```

Core algorithms of the project:

- grid generation
- SDF computation
- SDF weighting
- PCA fitting
- PCA reconstruction
- latent-space analysis
- clustering
- similarity search
- new geometry generation

---

### Plotting

```
plot_*.py
```

General visualization utilities.

```
sdf_plot_*.py
```

Plots specific to the SDF/PCA workflow:

- latent-space projections
- distance matrices
- PCA modes
- variance plots
- comparison figures

---

# Requirements
Developed and tested using Python3.10 on Windows OS. See requirements.txt for the full list of necessary packages.


# How to use
Quick guide to run the provided script (assuming you already have Python installed and accessible in the path).

1)Extract test_files.rar to this folder, expected result: 
- domain_exploration
	- test_files
		- Blended
		- Fan
		- Sokaris
		- Tesla_valve
		
2) Install requirements (strongly suggested to either use a IDE, like PyCharm, or at the very least a virtual environment):
```python -m pip install -r requirements.txt```

3) Run:
```python sdf_pca_for_KUL.py```