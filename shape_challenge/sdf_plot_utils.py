import os
import copy
import pickle

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

from skimage import measure

import generic_utils as gen_utils
import sdf_pca_utils as sdf_pca


# ============================================================================================================== Scatter
def plot_doe_distribution_pca(logger, original_space_, pca_, threshold=0.9, title_=None,
                              highlight_experiments=None,
                              case=None, save_folder=None):
    """
    Generate a multidimensional scatter plot of the experiments in PCA space.

    The number of plotted principal components is the minimum required to reach
    the specified cumulative explained-variance threshold. Highlighted experiments
    are displayed with a different marker.

    Parameters
    ----------
    logger : logging.Logger
        Logger instance.
    original_space_ : np.ndarray
        Array of shape (n_experiments, n_parameters) containing the original data.
    pca_ : sklearn.decomposition.PCA
        Fitted PCA object.
    threshold : float
        Cumulative explained-variance threshold in the range [0, 1].
    title_ : str
        Base title of the plot.
    highlight_experiments : array-like, optional
        Indices of the experiments to highlight using a different marker.
    case : str
        Name of the use case.
    save_folder : str or Path
        Directory where the figure will be saved.

    Returns
    -------
    None
    """

    if highlight_experiments is None:
        highlight_experiments = set()
    else:
        highlight_experiments = set(highlight_experiments)

    X_pca = pca_.fit_transform(original_space_)

    title_add = 'PCA1: X, PCA2: Y'
    plot_version = '2d'

    # components
    pc1 = X_pca[:, 0]
    pc2 = X_pca[:, 1]
    pc3 = None
    pc5 = None

    if sdf_pca.sum_pca(pca_, 2) > threshold:
        plot_version = '2d'
    elif (pca_.n_components_ >= 3) and (sdf_pca.sum_pca(pca_, 3) > threshold):
        plot_version = '3d'
    elif (pca_.n_components_ >= 4) and (sdf_pca.sum_pca(pca_, 4) > threshold):
        plot_version = '4d'
    elif (pca_.n_components_ >= 5) and (sdf_pca.sum_pca(pca_, 5) > threshold):
        plot_version = '5d'
    else:  # I never get above 90%, I use n_comp
        if pca_.n_components_ >= 5:
            plot_version = '5d'
        elif pca_.n_components_ >= 4:
            plot_version = '4d'
        elif pca_.n_components_ >= 3:
            plot_version = '3d'

    if plot_version == '3d' or plot_version == '4d' or plot_version == '5d':
        pc3 = X_pca[:, 2]
        title_add += ', PCA3: Z'

    if plot_version == '4d' or plot_version == '5d':
        title_add += ', PCA4: Marker size'

        pc4 = X_pca[:, 3]   # marker size

        # normalize marker size
        size_min, size_max = 40, 200
        pc4_norm = (pc4 - pc4.min()) / (pc4.max() - pc4.min() + 1e-12)
        sizes = size_min + pc4_norm * (size_max - size_min)

    if plot_version == '5d':

        title_add += ', PCA5: Marker color'

        pc5 = X_pca[:, 4]   # color
        if sdf_pca.sum_pca(pca_, 5) > threshold:
            plot_version = '5d'

    logger.info(f"\tplot_version: {plot_version}")
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')

    normal_idx = [i_ for i_ in range(len(original_space_)) if i_ not in highlight_experiments]
    highlight_idx = [i_ for i_ in range(len(original_space_)) if i_ in highlight_experiments]

    main_label = "Training"
    if plot_version == '2d':
        sc = ax.scatter(pc1[normal_idx], pc2[normal_idx], marker="o", label=main_label)

    elif plot_version == '3d':
        sc = ax.scatter(pc1[normal_idx], pc2[normal_idx], pc3[normal_idx], marker="o", label=main_label)

    elif plot_version == '4d':
        sc = ax.scatter(pc1[normal_idx], pc2[normal_idx], pc3[normal_idx], s=sizes[normal_idx], marker="o",
                        label=main_label)

    elif plot_version == '5d':
        sc = ax.scatter(pc1[normal_idx], pc2[normal_idx], pc3[normal_idx], c=pc5[normal_idx], cmap="viridis",
                        s=sizes[normal_idx], marker="o", label=main_label)
    else:
        logger.critical(f"\tUnknown plot_version: {plot_version}, ending plot")
        return

    other_label = 'LOO'
    # highlighted points
    if highlight_idx:
        if plot_version == '2d':
            ax.scatter(pc1[highlight_idx], pc2[highlight_idx], marker="^", edgecolor="black", label=other_label)

        elif plot_version == '3d':
            ax.scatter(pc1[highlight_idx], pc2[highlight_idx], pc3[highlight_idx], marker="^", edgecolor="black",
                       label=other_label)

        elif plot_version == '4d':
            ax.scatter(pc1[highlight_idx], pc2[highlight_idx], pc3[highlight_idx], s=sizes[highlight_idx],
                       marker="^", edgecolor="black", label=other_label)

        elif plot_version == '5d':
            ax.scatter(pc1[highlight_idx], pc2[highlight_idx], pc3[highlight_idx], c=pc5[highlight_idx],
                       cmap="viridis", s=sizes[highlight_idx], marker="^", edgecolor="black", label=other_label)

    # labels
    if plot_version == '2d':
        for i_, (x_, y_) in enumerate(zip(pc1, pc2)):
            ax.text(x_, y_, 0, str(i_+1), fontsize=8)
    else:
        for i_, (x_, y_, z_) in enumerate(zip(pc1, pc2, pc3)):
            ax.text(x_, y_, z_, str(i_+1), fontsize=8)

    ax.set_xlabel(f"PC1 ({pca_.explained_variance_ratio_[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pca_.explained_variance_ratio_[1]*100:.1f}% var)")
    if plot_version != '2d':
        ax.set_zlabel(f"PC3 ({pca_.explained_variance_ratio_[2]*100:.1f}% var)")

    title = f"{title_} DOE distribution in PCA space\n{title_add}"
    ax.set_title(title)

    # colorbar
    if plot_version == '5d':
        cbar = fig.colorbar(sc, ax=ax, shrink=0.6)
        cbar.set_label(f"PC5 ({pca_.explained_variance_ratio_[4]*100:.1f}% var)")

    ax.legend()
    plt.tight_layout()
    short_title = title.split("\n")[0]
    if save_folder:
        path_and_name = os.path.join(save_folder, f"{case}_{short_title}")

        plt.savefig(f"{path_and_name}.png", dpi=300, bbox_inches="tight")
        with open(f"{path_and_name}.pkl", "wb") as f:
            pickle.dump(fig, f)
    plt.show()


def plot_pca_projection(latent_training, latent_new=None, training_labels=None,
                        new_labels=None,
                        cluster_labels=None,
                        new_cluster_labels=None,
                        title="PCA 2D Projection", annotate=False,
                        case=None, save_folder=None):
    """
    Scatter plot of a 2D PCA projection, onlt first 2 components.

    Parameters
    ----------
    latent_training : (n_train, 2) ndarray
        Training points.
    latent_new : (n_new, 2) ndarray, optional
        New/evaluation points.
    training_labels : list[str], optional
        Labels for training points.
    new_labels : list[str], optional
        Labels for new points.
    cluster_labels : list[int], optional
        Cluster classification of each training point.
    new_cluster_labels : list[int], optional
        Cluster classification of each new point.
    title : str
        Plot title.
    annotate : bool
        If True, display labels next to points.
    case : str, optional
        Case name used when saving.
    save_folder : str, optional
        Folder where to save the figure.
    """

    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw coloured cluster background markers
    if cluster_labels is not None:

        unique_clusters = np.unique(cluster_labels)
        cmap = plt.get_cmap("tab20")

        for c in unique_clusters:

            idx = cluster_labels == c

            ax.scatter(latent_training[idx, 0], latent_training[idx, 1],
                       s=220,                    # larger than normal marker
                       color=cmap(c % 20), alpha=0.55, edgecolors='none', zorder=1)

            centroid = latent_training[idx].mean(axis=0)

            plt.scatter(centroid[0], centroid[1], marker='X', s=250,
                        c=[cmap(c % 20)], edgecolors='black', linewidths=1.5, zorder=6)

    # Training points
    ax.scatter(latent_training[:, 0], latent_training[:, 1], c='tab:blue',
               marker='o', s=50, label='Training')

    if annotate and training_labels is not None:
        for xy, label in zip(latent_training, training_labels):
            ax.text(xy[0], xy[1], label, fontsize=8, color='tab:blue')

    # Evaluation points
    if latent_new is not None and len(latent_new) > 0:

        if new_cluster_labels is not None:

            cmap = plt.get_cmap("tab20")

            for i in range(len(latent_new)):
                ax.scatter(latent_new[i, 0], latent_new[i, 1], s=260,
                           color=cmap(new_cluster_labels[i] % 20), alpha=0.55, edgecolors='none', zorder=1)

        ax.scatter(latent_new[:, 0], latent_new[:, 1], c='tab:red', marker='o', s=80, edgecolors='black',
                   linewidths=0.8, label='Evaluation')

        if annotate and new_labels is not None:
            for xy, label in zip(latent_new, new_labels):
                ax.text(xy[0], xy[1], label, fontsize=9, color='tab:red', fontweight='bold')

    ax.set_xlabel("Visualization PC1")
    ax.set_ylabel("Visualization PC2")
    ax.set_title(title)

    ax.grid(True)
    ax.legend()

    fig.tight_layout()

    if save_folder is not None:
        filename = gen_utils.clean_plot_title(title)
        if case is not None:
            filename = f"{case}_{filename}"

        path_and_name = os.path.join(save_folder, filename)

        fig.savefig(f"{path_and_name}.png", dpi=300)
        with open(f"{path_and_name}.pkl", "wb") as f:
            pickle.dump(fig, f)

    plt.show()


# ============================================================================================================== Heatmap
def plot_distance_matrix(D_, title_='Distance Matrix',
                         labels=None,
                         highlight_labels=None,
                         save_folder=None):
    """
    Display a distance matrix as a color-coded heatmap.

    Parameters
    ----------
    D_ : np.ndarray
        Square distance matrix of shape ``(n_geometries, n_geometries)``.
    title_ : str
        Title of the figure.
    labels : list[str]
        Labels for the x- and y-axis ticks. The order should correspond to the
        rows and columns of ``D_``.
    highlight_labels : list[str]
        Subset of ``labels`` to highlight (e.g. in red) to distinguish selected
        geometries.
    save_folder : str
        Directory where the figure will be saved.

    Returns
    -------
    None
    """

    fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.imshow(D_, interpolation="nearest")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title(title_)

    if labels is None:
        ax.set_xlabel("Geometry index")
        ax.set_ylabel("Geometry index")
    else:
        n = len(labels)

        ax.set_xticks(np.arange(n))
        ax.set_xticklabels(labels, rotation=90)

        ax.set_yticks(np.arange(n))
        ax.set_yticklabels(labels)

        # Highlight selected labels
        if highlight_labels is not None:
            highlight_set = set(highlight_labels)

            for tick in ax.get_xticklabels():
                if tick.get_text() in highlight_set:
                    tick.set_color("red")
                    tick.set_fontweight("bold")

            for tick in ax.get_yticklabels():
                if tick.get_text() in highlight_set:
                    tick.set_color("red")
                    tick.set_fontweight("bold")

    fig.tight_layout()

    if save_folder:
        filename = gen_utils.clean_plot_title(title_)
        path_and_name = os.path.join(save_folder, filename)

        fig.savefig(f"{path_and_name}.png",
                    dpi=300,
                    bbox_inches="tight")

        with open(f"{path_and_name}.pkl", "wb") as f:
            pickle.dump(fig, f)

    plt.show()


# ========================================================================================================= Mesh Objects
def plot_mesh(ax, verts_, faces_, color_, alpha_):
    """
    Plot a mesh using matplotlib. Not standalone, has to receive exising ax (appends on).
    Parameters
    ----------
    ax : mpl_toolkits.mplot3d.axes3d.Axes3D
        Matplotlib 3D axis on which the mesh will be plotted.
    verts_ : np.ndarray
        Array of vertex coordinates of shape ``(n_vertices, 3)``, where each
        row contains the ``(x, y, z)`` coordinates of a vertex.
    faces_ : np.ndarray
        Array of triangular face indices of shape ``(n_faces, 3)``. Each row
        contains the indices of the three vertices defining a triangle.
    color_ : str or tuple
        Face color of the mesh. Any Matplotlib-compatible color specification
        is accepted.
    alpha_ : float
        Transparency of the mesh, in the range ``[0, 1]``.

    Returns
    -------
    None
    """
    ax.plot_trisurf(verts_[:, 0], verts_[:, 1], verts_[:, 2], triangles=faces_, color=color_, alpha=alpha_,
                    linewidth=0)


def plot_pca_mode_true_scale(pca, grid_size_x, grid_size_y, grid_size_z, mode_index_=0, alpha_=1.0):
    """
    Visualize the effect of a principal component on the reconstructed geometry.

    The function reconstructs three signed distance fields (SDFs): the mean shape,
    the shape obtained by moving a distance ``+alpha`` along the selected principal
    component, and the shape obtained by moving ``-alpha`` along the same component.
    The corresponding zero-level isosurfaces are extracted using the marching cubes
    algorithm and displayed together in a 3D plot.

    Parameters
    ----------
    pca : sklearn.decomposition.PCA
        Fitted PCA model whose samples are flattened SDFs.
    grid_size_x : int
        Number of grid points along the x-axis.
    grid_size_y : int
        Number of grid points along the y-axis.
    grid_size_z : int
        Number of grid points along the z-axis.
    mode_index_ : int, optional
        Index of the principal component to visualize. Default is ``0``.
    alpha_ : float, optional
        Amplitude of the perturbation expressed in units of the standard deviation
        of the selected principal component. Default is ``1.0``.

    Returns
    -------
    None

    Notes
    -----
    The three reconstructed geometries are displayed with the following colors:

    - **Red:** ``-alpha`` perturbation
    - **Green:** mean geometry
    - **Blue:** ``+alpha`` perturbation

    The axis limits are adjusted so that all geometries are displayed with the
    same scale and aspect ratio, preserving their true geometric proportions.
    """

    # build SDFs
    mean_sdf = pca.mean_.reshape(grid_size_x, grid_size_y, grid_size_z)

    latent = np.zeros((1, pca.n_components_))
    latent[0, mode_index_] = alpha_ * np.sqrt(pca.explained_variance_[mode_index_])
    sdf_plus = pca.inverse_transform(latent).reshape(grid_size_x, grid_size_y, grid_size_z)

    latent[0, mode_index_] = -alpha_ * np.sqrt(pca.explained_variance_[mode_index_])
    sdf_minus = pca.inverse_transform(latent).reshape(grid_size_x, grid_size_y, grid_size_z)

    # extract meshes
    verts_mean, faces_mean, _, _ = measure.marching_cubes(mean_sdf, level=0.0)
    verts_plus, faces_plus, _, _ = measure.marching_cubes(sdf_plus, level=0.0)
    verts_minus, faces_minus, _, _ = measure.marching_cubes(sdf_minus, level=0.0)

    # plt
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    plot_mesh(ax, verts_minus, faces_minus, 'red', 0.5)
    plot_mesh(ax, verts_mean, faces_mean, 'green', 0.5)
    plot_mesh(ax, verts_plus, faces_plus, 'blue', 0.5)

    # true scale enforcer
    all_verts = np.vstack([verts_mean, verts_plus, verts_minus])

    x_min, x_max = all_verts[:, 0].min(), all_verts[:, 0].max()
    y_min, y_max = all_verts[:, 1].min(), all_verts[:, 1].max()
    z_min, z_max = all_verts[:, 2].min(), all_verts[:, 2].max()

    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min
    max_range = max(x_range, y_range, z_range)

    x_mid = (x_max + x_min) / 2
    y_mid = (y_max + y_min) / 2
    z_mid = (z_max + z_min) / 2

    ax.set_xlim(x_mid - max_range/2, x_mid + max_range/2)
    ax.set_ylim(y_mid - max_range/2, y_mid + max_range/2)
    ax.set_zlim(z_mid - max_range/2, z_mid + max_range/2)

    # set box aspect roperly
    ax.set_box_aspect([1, 1, 1])

    ax.set_title(f"PCA Mode {mode_index_}\n(R -alpha, G, B +alpha)")
    plt.show()


# =============================================================================================================== Others
# compare distances between new experiments and the existig ones
def plot_comparison_with_existing(D_eval, i_new_mesh=0,
                                  mean_training_distance=None,
                                  new_centroid_dist=None,
                                  existing_labels=None, new_labels=None, n_closest=5):
    """
    Plot the nearest training geometries to a selected evaluation geometry.

    The function displays the ``n_closest`` training geometries in terms of
    distance in the PCA latent space. It also shows the average pairwise training
    distance and the distance between the evaluation geometry and the training
    centroid for reference.

    Parameters
    ----------
    D_eval : np.ndarray
        Distance matrix of shape ``(n_evaluation, n_training)``, where each entry
        contains the latent-space distance between an evaluation geometry and a
        training geometry.
    i_new_mesh : int, optional
        Index of the evaluation geometry to visualize. Default is ``0``.
    mean_training_distance : float, optional
        Mean pairwise distance between the training geometries. Displayed as a
        vertical reference line.
    new_centroid_dist : np.ndarray or list[float], optional
        Distance of each evaluation geometry from the training centroid. The value
        corresponding to ``i_new_mesh`` is displayed as a vertical reference line.
    existing_labels : list[str], optional
        Labels of the training geometries.
    new_labels : list[str], optional
        Labels of the evaluation geometries.
    n_closest : int, optional
        Number of the nearest training geometries to display. Default is ``5``.

    Returns
    -------
    None
    """

    order = np.argsort(D_eval[0])
    closest_idx = order[:n_closest]
    closest_dist = D_eval[0, closest_idx]
    closest_labels = [existing_labels[i] for i in closest_idx]

    plt.figure(figsize=(8, 6))
    plt.barh(np.arange(len(closest_dist)), closest_dist)
    plt.axvline(mean_training_distance, linestyle='--', label='Average training distance')
    plt.axvline(new_centroid_dist[i_new_mesh], color='r', linestyle='--', label='Distance to centroid')
    plt.yticks(np.arange(len(closest_dist)), closest_labels)

    plt.xlabel("Distance in PCA latent space")
    plt.title(f"Closest geometries to {new_labels[i_new_mesh]}")

    plt.gca().invert_yaxis()
    plt.legend()
    plt.tight_layout()
    plt.show()


# ================================================================================================================== SDF
def plot_multiple_slices(sdf, axis='z', n_slices=None, title=''):
    """
    Display multiple slices of a signed distance field (SDF).

    The function extracts evenly spaced slices along the selected axis and displays
    them in a grid using a diverging colormap, allowing the internal structure of
    the SDF to be inspected visually.

    Parameters
    ----------
    sdf : np.ndarray
        Signed distance field of shape ``(Nx, Ny, Nz)``.
    axis : {"x", "y", "z"}, optional
        Axis along which the slices are extracted. Default is ``"z"``.
    n_slices : int or None, optional
        Number of evenly spaced slices to display. If ``None`` or evaluates to
        ``False``, all slices along the selected axis are plotted. Default is ``9``.
    title : str, optional
        Text prepended to the figure title. Default is an empty string.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If ``axis`` is not one of ``"x"``, ``"y"``, or ``"z"``.
    """

    Nx, Ny, Nz = sdf.shape

    if axis == 'z':
        if n_slices is None:
            n_slices = Nz
        indices = np.linspace(0, Nz - 1, n_slices, dtype=int)
    elif axis == 'y':
        if n_slices is None:
            n_slices = Ny
        indices = np.linspace(0, Ny - 1, n_slices, dtype=int)
    elif axis == 'x':
        if n_slices is None:
            n_slices = Nx
        indices = np.linspace(0, Nx - 1, n_slices, dtype=int)
    else:
        raise ValueError("axis must be 'x', 'y', or 'z'")

    ncols = int(np.ceil(np.sqrt(n_slices)))
    nrows = int(np.ceil(n_slices / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 12))
    axes = axes.flatten()

    for i, idx in enumerate(indices):
        if axis == 'z':
            slice_ = sdf[:, :, idx]
        elif axis == 'y':
            slice_ = sdf[:, idx, :]
        elif axis == 'x':
            slice_ = sdf[idx, :, :]
        else:
            raise ValueError("axis must be 'x', 'y', or 'z'")

        im = axes[i].imshow(slice_.T, origin='lower', cmap='seismic')
        axes[i].set_title(f"{axis}={idx}")
        axes[i].axis('off')

    # remove unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    fig.colorbar(im, ax=axes, shrink=0.6)
    fig.suptitle(f"{title} SDF slices along {axis}-axis")
    plt.tight_layout()
    plt.show()


def plot_cumvar_and_mse(ks_, vars_, mses_):
    """
    Plot cumulative explained variance and reconstruction error as a function of
    the number of PCA components.

    Parameters
    ----------
    ks_ : array-like
        Numbers of retained principal components.
    vars_ : array-like
        Cumulative explained variance corresponding to each value in ``ks_``.
    mses_ : array-like
        Reconstruction mean squared error (MSE) corresponding to each value in
        ``ks_``.

    Returns
    -------
    None
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("PCA model selection")

    # --- Cumulative variance ---
    ax1.plot(ks_, vars_, marker='o')
    ax1.set_xlabel("Components")
    ax1.set_ylabel("Cumulative Variance")
    ax1.grid(True)

    # --- MSE ---
    ax2.plot(ks_, mses_, marker='o')
    ax2.set_xlabel("Components")
    ax2.set_ylabel("Reconstruction MSE")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


def distances_to_colors(distances_, cmap='jet_r'):
    """
    Map a vector of scalar distances to RGB colors.

    The input distances are first normalized to the range [0, 1] and then
    converted to RGB values using the specified Matplotlib colormap. The
    resulting colors can be assigned to mesh vertices to visualize geometric
    errors or other scalar quantities.

    Parameters
    ----------
    distances_ : numpy.ndarray of shape (n_vertices,)
        Distance or error value associated with each mesh vertex.

    cmap : str, optional
        Name of the Matplotlib colormap to use. The default is ``'jet_r'``,
        which maps small values to blue and large values to red.

    Returns
    -------
    numpy.ndarray of shape (n_vertices, 3)
        RGB color values in the range [0, 1] corresponding to the normalized
        input distances.

    Notes
    -----
    If all distances are identical, the normalized values are set to zero,
    resulting in a uniform color for all vertices.
    """

    # normalize
    d_min = distances_.min()
    d_max = distances_.max()

    if d_max > d_min:
        norm = (distances_ - d_min) / (d_max - d_min)
    else:
        norm = np.zeros_like(distances_)

    # colormap
    colors = plt.get_cmap(cmap)(norm)[:, :3]  # drop alpha

    return colors


def color_mesh_by_error(mesh_a, mesh_b):
    """
    Color a mesh according to its geometric error with respect to a reference mesh.

    The error at each vertex of ``mesh_a`` is computed as the Euclidean
    distance to the nearest vertex of ``mesh_b``. These distances are then
    mapped to RGB colors and assigned to a copy of ``mesh_a`` for
    visualization.

    Parameters
    ----------
    mesh_a : open3d.geometry.TriangleMesh
        Mesh to be colored, typically the reconstructed or predicted geometry.

    mesh_b : open3d.geometry.TriangleMesh
        Reference mesh used to compute the vertex-wise errors.

    Returns
    -------
    mesh_colored : open3d.geometry.TriangleMesh
        Deep copy of ``mesh_a`` with vertex colors representing the local
        geometric error.

    distances_ : numpy.ndarray of shape (n_vertices,)
        Vertex-wise Euclidean distances from ``mesh_a`` to the nearest vertex
        in ``mesh_b``.

    Notes
    -----
    - The original mesh is left unmodified; a deep copy is created before
      assigning colors.
    - Errors are computed using nearest-neighbor vertex distances rather than
      true point-to-surface distances.
    - Colors are generated using :func:`distances_to_colors`.
    """

    distances_ = sdf_pca.compute_vertex_errors(mesh_a, mesh_b)
    colors = distances_to_colors(distances_)

    mesh_colored = copy.deepcopy(mesh_a)  # avoid modifying original
    mesh_colored.vertex_colors = o3d.utility.Vector3dVector(colors)

    return mesh_colored, distances_
