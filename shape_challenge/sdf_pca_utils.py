import open3d as o3d
import numpy as np
import logging
import os
import sys
import configparser
import trimesh
import umap  # used in final stage, evaluation and clustering

from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances  # used in distance matrix
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from scipy.spatial import procrustes  # used to re-organize the UMAPs
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
from scipy.stats import percentileofscore

from skimage import measure

import plot_utils as plot_utils
import sdf_plot_utils as sdf_plot

from minisom import MiniSom

standard_seed = 42
np.random.seed(seed=standard_seed)


# ------------------------------------------------------------------------------------------------------------------ SDF
def generate_sdf_grid(logger, geometric_bb, min_grid_number):
    """
    Generate a regular 3D grid of points inside a geometric bounding box.

    The grid resolution is computed such that the smallest bounding box
    dimension is divided into approximately `min_grid_number` intervals.
    The same grid spacing is then applied to all three directions.

    Parameters
    ----------
    logger : logging.Logger
        Logger object used to report grid generation information.

    geometric_bb : dict
        Dictionary containing the geometric bounding box limits. Expected keys:

        - ``x_min`` : float
            Minimum x-coordinate.
        - ``x_max`` : float
            Maximum x-coordinate.
        - ``y_min`` : float
            Minimum y-coordinate.
        - ``y_max`` : float
            Maximum y-coordinate.
        - ``z_min`` : float
            Minimum z-coordinate.
        - ``z_max`` : float
            Maximum z-coordinate.

    min_grid_number : int
        Number of grid divisions desired along the smallest bounding box axis.

    Returns
    -------
    grid_points : numpy.ndarray
        Coordinates of the generated grid points with shape:

        (n_grids_x, n_grids_y, n_grids_z, 3)

        where the last dimension contains the x, y, and z coordinates.

    minimum_axis : str or int
        Identifier of the smallest bounding box dimension, returned by
        ``find_minimum_axis``.

    n_grids_x : int
        Number of grid points along the x direction.

    n_grids_y : int
        Number of grid points along the y direction.

    n_grids_z : int
        Number of grid points along the z direction.

    grid_size : float
        Target grid spacing computed from the smallest bounding box dimension.

    Notes
    -----
    The returned grid includes both bounding box limits because ``numpy.linspace``
    includes the endpoint. Therefore, the actual spacing may differ slightly from
    ``grid_size`` due to integer rounding of the number of grid points.
    """

    size_x = geometric_bb['x_max'] - geometric_bb['x_min']
    size_y = geometric_bb['y_max'] - geometric_bb['y_min']
    size_z = geometric_bb['z_max'] - geometric_bb['z_min']

    minimum_axis = find_minimum_axis(size_x, size_y, size_z)
    grid_size = np.min([size_x, size_y, size_z]) / min_grid_number

    logger.info(f"grid_size: {grid_size}")
    n_grids_x = max(2, int(size_x / grid_size))
    n_grids_y = max(2, int(size_y / grid_size))
    n_grids_z = max(2, int(size_z / grid_size))

    logger.info(f"n_grids_x: {n_grids_x}, n_grids_y: {n_grids_y}, n_grids_z: {n_grids_z}")
    x = np.linspace(geometric_bb['x_min'], geometric_bb['x_max'], n_grids_x)
    y = np.linspace(geometric_bb['y_min'], geometric_bb['y_max'], n_grids_y)
    z = np.linspace(geometric_bb['z_min'], geometric_bb['z_max'], n_grids_z)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # grid for SDF
    grid_points = np.stack([X, Y, Z], axis=-1)  # (Nx, Ny, Nz, 3)
    logger.info(f"Shape of the grid_points:{grid_points.shape}")

    return grid_points, minimum_axis, n_grids_x, n_grids_y, n_grids_z, grid_size


def compute_sdf_from_mesh(logger, mesh_, grid_points_, clip_=0.1, threshold=0.01):
    """
    Compute the signed distance field (SDF) of a triangular mesh on a regular grid.

    The signed distance values are evaluated at the specified grid points using
    Open3D. The resulting distances are optionally clipped to the interval
    ``[-clip_, clip_]``. The function also checks that the mesh encloses a
    sufficient fraction of the grid points to detect invalid or poorly defined
    geometries.

    Parameters
    ----------
    logger : logging.Logger
        Logger instance used to report progress and warnings.
    mesh_ : o3d.geometry.TriangleMesh
        Input triangular mesh.
    grid_points_ : np.ndarray
        Grid of query points with shape ``(Nx, Ny, Nz, 3)`` at which the signed
        distance field is evaluated.
    clip_ : float, optional
        Maximum absolute signed distance retained in the output. Distances larger
        than ``clip_`` in magnitude are clipped. Default is ``0.1``.
    threshold : float, optional
        Minimum acceptable fraction of grid points lying inside the mesh. Meshes
        below this threshold are considered invalid. Default is ``0.01``.

    Returns
    -------
    np.ndarray
        Signed distance field with shape ``(Nx, Ny, Nz)``.

    Raises
    ------
    ValueError
        If the mesh does not satisfy the validity criterion based on
        ``threshold``.
    """
    tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh_)

    # Create scene for distance queries
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(tmesh)

    # Compute signed distance
    query_points = o3d.core.Tensor(grid_points_, dtype=o3d.core.Dtype.Float32)
    sdf_ = scene.compute_signed_distance(query_points).numpy()

    # Truncate SDF (recommended)
    sdf_ = np.clip(sdf_, -1 * clip_, clip_)
    neg_ratio = np.mean(sdf_ < 0)
    if neg_ratio < threshold:
        logger.critical(f"\tSDF Fraction of internal points: {neg_ratio:.4f} is below threshold of {threshold}")
        logger.critical("SDF will be discarded")
        sdf_ = None
    else:
        logger.info(f"\tSDF Fraction of internal points: {neg_ratio:.4f}")
    return sdf_


def to_o3d_mesh(verts_, faces_):
    """
    Create an Open3D triangular mesh from arrays of vertices and faces.

    Parameters
    ----------
    verts_ : np.ndarray
        Array of vertex coordinates with shape ``(n_vertices, 3)``.
    faces_ : np.ndarray
        Array of triangular face indices with shape ``(n_faces, 3)``. Each row
        contains the indices of the three vertices defining a triangle.

    Returns
    -------
    o3d.geometry.TriangleMesh
        Open3D triangular mesh constructed from the provided vertices and faces.
    """

    mesh_ = o3d.geometry.TriangleMesh()
    mesh_.vertices = o3d.utility.Vector3dVector(verts_)
    mesh_.triangles = o3d.utility.Vector3iVector(faces_)
    mesh_.compute_vertex_normals()
    return mesh_


def compute_sdfs(logger, meshes_, grid_points_, geometry_dict=None, type_='',
                 make_plot=False, axis='x', n_slices=None, clip_=0.1, save_path=None,
                 exp_ids=None, load_sdf=True):
    """
    Compute or load the signed distance fields (SDFs) for a collection of meshes.

    This function iterates over the supplied meshes, computes (or loads) their
    signed distance fields on the specified grid, optionally visualizes the SDFs,
    and returns the successfully processed experiments.

    Parameters
    ----------
    logger : logging.Logger
        Logger instance used to report progress and warnings.
    meshes_ : list[o3d.geometry.TriangleMesh]
        List of Open3D triangular meshes to process.
    grid_points_ : np.ndarray
        Grid of query points with shape ``(Nx, Ny, Nz, 3)`` on which the SDF is
        evaluated.
    geometry_dict : list[dict]
        List of dictionaries associating experiment folders with the corresponding
        source geometry, for example::

            {
                "geometry": "case_0000.vtk",
                "folder": "exp_1"
            }

        This list may contain more entries than ``meshes_`` and ``exp_ids``, as it
        typically includes all available geometries rather than only the current
        training or evaluation subset.
    type_ : str
        Label used in log messages (e.g. ``"Training"`` or ``"Evaluation"``).
    make_plot : bool
        If ``True``, generate slice-by-slice visualizations of the computed SDFs.
    axis : str
        Axis along which SDF slices are extracted for visualization.
    n_slices : int
        Number of slices to display.
    clip_ : float
        Maximum absolute signed distance retained in the SDF.
    save_path : str
        Path (without file extension) used to save or load SDF files.
    exp_ids : list[str]
        Experiment folder names corresponding to the meshes in ``meshes_``.
    load_sdf : bool, optional
        If ``True`` (default), existing SDF files are loaded when available instead
        of being recomputed.

    Returns
    -------
    list[np.ndarray]
        Signed distance fields corresponding to the successfully processed
        experiments.
    list[str]
        Experiment folder names for which a valid SDF was obtained.
    """

    # compute
    sdfs_ = []
    valid_experiments = []

    for index_exp_, exp_name_ in enumerate(exp_ids):

        geometry_name = find_geometry(geometry_dict, exp_name_)
        if not geometry_name:
            logger.critical(f"No geometry name found for experiment {exp_name_}")
            sys.exit()
        # try load existing sdf if requested and possible
        current_name = f"{geometry_name.split('.')[0]}_{clip_}_" \
                       f"{grid_points_.shape[0]}_{grid_points_.shape[1]}_{grid_points_.shape[2]}.npz"
        full_path = os.path.join(save_path, current_name)

        if save_path and load_sdf and os.path.exists(full_path):
            logger.info(f"Loading cached SDF: {current_name} for experiment: {exp_name_}")
            data = np.load(full_path)
            sdf_ = data["sdf"]
        else:
            if len(meshes_) > 0:
                logger.info(f'Working on {type_} {exp_name_}')
                sdf_ = compute_sdf_from_mesh(logger, meshes_[index_exp_], grid_points_, clip_=clip_)
                if sdf_ is not None:
                    np.savez_compressed(full_path, sdf=sdf_)
            else:
                logger.critical(f"Trying to use load on a project that was not run before!\n"
                                f"Cannot read SDF file: {full_path}")
                sys.exit(1)

        if sdf_ is not None:
            if make_plot:
                sdf_plot.plot_multiple_slices(sdf_, axis=axis, n_slices=n_slices, title=exp_name_)

            sdfs_.append(sdf_)
            valid_experiments.append(exp_name_)

    sdfs_ = np.array(sdfs_)

    logger.debug(f"{type_} SDFs, np.std(sdfs): {np.std(sdfs_)}")
    # flatten if needed
    if sdfs_.ndim > 2:
        sdfs_ = sdfs_.reshape(sdfs_.shape[0], -1)

    return sdfs_, valid_experiments


def update_bb_min_max(bb_, x_min_, x_max_, y_min_, y_max_, z_min_, z_max_):
    """
    Update a bounding box to include a new set of coordinate extrema.

    Parameters
    ----------
    bb_ : dict[str, float]
        Current bounding box with the keys ``"x_min"``, ``"x_max"``,
        ``"y_min"``, ``"y_max"``, ``"z_min"``, and ``"z_max"``.
    x_min_ : float
        Minimum x-coordinate of the new geometry.
    x_max_ : float
        Maximum x-coordinate of the new geometry.
    y_min_ : float
        Minimum y-coordinate of the new geometry.
    y_max_ : float
        Maximum y-coordinate of the new geometry.
    z_min_ : float
        Minimum z-coordinate of the new geometry.
    z_max_ : float
        Maximum z-coordinate of the new geometry.

    Returns
    -------
    dict[str, float]
        Updated bounding box enclosing both the previous bounding box and
        the new coordinate extrema.
    """

    bb_['x_min'] = check_value(bb_['x_min'], x_min_, 'min')
    bb_['y_min'] = check_value(bb_['y_min'], y_min_, 'min')
    bb_['z_min'] = check_value(bb_['z_min'], z_min_, 'min')
    bb_['x_max'] = check_value(bb_['x_max'], x_max_, 'max')
    bb_['y_max'] = check_value(bb_['y_max'], y_max_, 'max')
    bb_['z_max'] = check_value(bb_['z_max'], z_max_, 'max')

    return bb_


def find_minimum_axis(size_x_, size_y_, size_z_):
    """
    Return the axis with the smallest geometric extent.

    Parameters
    ----------
    size_x_ : float
        Extent of the geometry along the x-axis (``x_max - x_min``).
    size_y_ : float
        Extent of the geometry along the y-axis (``y_max - y_min``).
    size_z_ : float
        Extent of the geometry along the z-axis (``z_max - z_min``).

    Returns
    -------
    str
        Axis with the smallest extent: ``"x"``, ``"y"``, or ``"z"``.
    """

    sizes = {'x': size_x_, 'y': size_y_, 'z': size_z_}
    return min(sizes, key=sizes.get)


def get_bounds(X_):
    """
    Compute the minimum and maximum values of each feature.

    Parameters
    ----------
    X_ : ndarray of shape (n_samples, n_features)
        Input data.

    Returns
    -------
    list of tuple(float, float)
        A list where the *i*-th element contains the minimum and maximum values
        of the *i*-th feature as ``(min, max)``.
    """
    return list(zip(np.min(X_, axis=0), np.max(X_, axis=0)))


def apply_weight(sdfs_, alpha_):
    """
    Apply an exponential weighting to signed distance fields (SDFs).

    The weighting attenuates regions that are far from the geometry surface so
    that subsequent PCA focuses on geometrically relevant variations rather
    than the largely constant far-field region.

    Parameters
    ----------
    sdfs_ : ndarray
        Collection of signed distance fields. The first dimension is assumed to
        index the samples.

    alpha_ : float
        Decay parameter controlling how rapidly the weight decreases with the
        absolute distance from the surface. Larger values place more emphasis
        on voxels close to the zero level set.

    Returns
    -------
    ndarray of shape (n_samples, n_voxels)
        Weighted SDFs flattened into vectors, suitable for PCA.

    Notes
    -----
    The weighting function is

        ``w = exp(-alpha * |SDF|)``

    where

    - voxels close to the surface (|SDF| ≈ 0) have weights close to 1,
    - voxels far from the surface have weights approaching 0.

    Since most voxels in an SDF correspond to empty space far from the
    geometry, this weighting reduces their influence during PCA and encourages
    the decomposition to capture variations in the geometry itself rather than
    the surrounding domain.
    """

    weights_ = np.exp(-alpha_ * np.abs(sdfs_))

    weighted = sdfs_ * weights_
    return weighted.reshape(weighted.shape[0], -1)


# ------------------------------------------------------------------------------------------------------------------ PCA
def sum_pca(pca_, n_):
    """
    Provides cumulative variance ration summed up to component n_
    :param pca_: PCA object
    :param n_: number of components to be included in the sum
    :return: Cumulative variance up to component n
    """

    return pca_.explained_variance_ratio_[:n_].sum()


def evaluate_error_from_pca_components(logger, X_, max_components=50, make_plot=False):
    """
    Evaluate PCA reconstruction quality as a function of the number of retained components.

    A PCA model is fitted once to the input data. The dataset is then
    progressively reconstructed using the first ``k`` principal components,
    where ``k`` ranges from 1 to ``max_components``. For each reconstruction,
    the cumulative explained variance and a weighted mean squared error (MSE)
    are computed. Optionally, the cumulative variance and reconstruction error
    curves can be visualized.

    Parameters
    ----------
    logger : logging.Logger
        Logger used to report the reconstruction progress.

    X_ : ndarray of shape (n_samples, n_features)
        Input data to be decomposed by PCA.

    max_components : int, optional
        Maximum number of principal components to consider. Default is 50.

    make_plot : bool, optional
        If ``True``, generate a plot showing the cumulative explained variance
        and the weighted reconstruction error as functions of the number of
        retained PCA components. Default is ``False``.

    Returns
    -------
    list of tuple(int, float, float)
        A list containing one tuple for each number of retained components.
        Each tuple has the form

        ``(n_components, cumulative_variance, weighted_mse)``

        where

        - ``n_components`` is the number of retained principal components,
        - ``cumulative_variance`` is the cumulative, explained variance ratio,
        - ``weighted_mse`` is the weighted mean squared reconstruction error.

    Notes
    -----
    The reconstruction error is computed using the weight matrix

        ``weights = exp(-abs(X_))``

    which assigns larger weights to values close to zero and progressively
    smaller weights to values farther from zero. This is particularly useful
    when working with signed distance fields (SDFs), where voxels near the
    geometry surface (SDF ≈ 0) contain most of the geometric information,
    while large positive or negative values correspond to relatively
    uninformative far-field regions.

    A warning is logged if the first two principal components explain less
    than 50% of the total variance, as this may indicate that the dataset
    cannot be represented accurately using only a few modes.
    """

    # Fit PCA once
    pca = PCA(n_components=max_components, svd_solver='full')

    latent = pca.fit_transform(X_)

    components = pca.components_
    mean = pca.mean_

    explained = pca.explained_variance_ratio_

    # weights for weighted MSE
    weights = np.exp(-np.abs(X_))

    results = []

    # Progressive reconstruction
    for k in range(1, max_components + 1):
        logger.debug(f"\tPCA reconstruction with {k} components")

        # truncate latent
        latent_k = latent[:, :k]

        # truncate basis
        comp_k = components[:k]

        # manual inverse transform
        reconstructed = latent_k @ comp_k + mean

        # cumulative explained variance
        cumvar = np.sum(explained[:k])

        # weighted reconstruction error
        mse_weighted = np.mean(weights * (X_ - reconstructed) ** 2)

        results.append((k, cumvar, mse_weighted))

    ks, cvars, mses = zip(*results)

    if cvars[1] < 0.5:
        logger.warning("----------------------------------------------------------------------------------------------")
        logger.warning(f"\tPCA mode contributions, sum of first two: {cvars[1]:>4.2e}")
        logger.warning("----------------------------------------------------------------------------------------------")
    else:
        logger.info(f"\tPCA mode contributions, sum of first two: {cvars[1]:>4.2e}")

    if make_plot:
        sdf_plot.plot_cumvar_and_mse(ks, cvars, mses)

    return results


def n_components_for_accuracy(logger, pca, threshold=0.9):
    """
    Determine the minimum number of principal components required to
    explain a target fraction of the total variance.

    The function computes the cumulative explained variance ratio of a
    previously fitted PCA model and returns the smallest number of
    components whose cumulative variance exceeds the specified threshold.

    Parameters
    ----------
    logger : logging.Logger
        Logger object used to report the cumulative explained variance
        for each number of retained components.

    pca : sklearn.decomposition.PCA
        A fitted PCA object. The function uses the
        ``explained_variance_ratio_`` attribute and assumes that
        ``fit()`` or ``fit_transform()`` has already been called.

    threshold : float, optional
        Target cumulative explained variance ratio. Must be between 0 and
        1. The default is 0.9 (90% explained variance).

    Returns
    -------
    int
        Minimum number of principal components required to reach the
        requested explained variance threshold. If the threshold cannot
        be achieved, the total number of available components is returned.
    """

    cumvar = np.cumsum(pca.explained_variance_ratio_)

    for k, v in enumerate(cumvar, start=1):
        f"\tCumulative explained variance with {k} components: {v:>7.4e}"
        if v >= threshold:
            return k

    logger.warning(
        f"\tMaximum number of components ({len(cumvar)}) "
        f"explains only {cumvar[-1]:.4f} variance."
    )
    return len(cumvar)


# ---------------------------------------------------------------------------------------------------------- SDF and PCA
def project_new_mesh(logger, new_mesh, sdf_grid_points, sdf_alpha_weight, existing_pca):
    """
    Project a new mesh into an existing PCA latent space.

    The mesh is first converted into a signed distance field (SDF) evaluated on
    the provided grid. The SDF is then weighted to emphasize the geometry
    surface before being projected into the latent space defined by a
    previously fitted PCA model.

    Parameters
    ----------
    logger : logging.Logger
        Logger used to report progress during SDF computation.

    new_mesh : open3d.geometry.TriangleMesh
        Mesh to be projected into the PCA latent space.

    sdf_grid_points : numpy.ndarray
        Grid of points at which the signed distance field is evaluated. The
        array is typically generated by ``generate_sdf_grid`` and has shape
        ``(Nx, Ny, Nz, 3)``.

    sdf_alpha_weight : float
        Weighting parameter controlling the exponential attenuation applied to
        the SDF values. Larger values increase the emphasis on regions close to
        the geometry surface.

    existing_pca : sklearn.decomposition.PCA
        Previously fitted PCA model defining the latent space.

    Returns
    -------
    numpy.ndarray of shape (1, n_components)
        Latent-space coordinates of the input mesh.

    Notes
    -----
    The SDF weighting is performed using :func:`apply_weight`, which suppresses
    the contribution of far-field voxels and emphasizes regions close to the
    zero level set. The resulting weighted SDF is reshaped into a single-row
    feature vector before being projected using ``PCA.transform``.
    """

    # compute SDF of the new mesh
    new_sdf = compute_sdf_from_mesh(logger, new_mesh, sdf_grid_points)  # shape (grid_size^3,)

    # apply weight to focus on geometry
    new_sdf = apply_weight([new_sdf], sdf_alpha_weight)

    # flatten if needed
    new_sdf_flat = new_sdf.flatten().reshape(1, -1)

    # project into PCA space
    latent_new = existing_pca.transform(new_sdf_flat)  # shape (1, n_components)

    return latent_new


def project_new_experiment(logger, scaler, new_experiment, existing_pca):
    """
    Scale a new experiment using the training scaler and project it into an
    existing PCA latent space.

    Parameters
    ----------
    logger : logging.Logger
        Logger object.

    scaler : sklearn.preprocessing.MinMaxScaler or StandardScaler
        Scaler previously fitted on the training dataset.

    new_experiment : ndarray of shape (n_features,) or (1, n_features)
        New experiment to project.

    existing_pca : sklearn.decomposition.PCA
        PCA model fitted on the scaled training data.

    Returns
    -------
    ndarray of shape (1, n_components)
        Latent-space coordinates of the new experiment.
    """

    new_experiment = np.asarray(new_experiment, dtype=float)

    # Ensure shape (1, n_features)
    if new_experiment.ndim == 1:
        new_experiment = new_experiment.reshape(1, -1)

    # Use the already-fitted scaler
    experiment_scaled = scaler.transform(new_experiment)

    # Project into PCA space
    latent_new = existing_pca.transform(experiment_scaled)

    return latent_new


# --------------------------------------------------------------------------------------------------------- Dictionaries
def compute_nested_dict_stats(data):
    """
    Compute min, max, and range for each field in a nested dictionary
    Parameters
    ----------
    data : dict
        Example:
        {
            'geom_000': {'B1': 117.4, 'B2': 69.97},
            'geom_001': {'B1': 166.0, 'B2': 53.31}
        }

    Returns
    -------
    stats : dict
        Example:
        {
            'B1': {'min': 117.4, 'max': 166.0, 'range': 48.6},
            'B2': {'min': 53.31, 'max': 69.97, 'range': 16.66}
        }
    """

    if not data:
        raise ValueError("Input dictionary is empty")

    # Get all parameter names from first entry
    keys = next(iter(data.values())).keys()
    stats = {}

    for key in keys:
        values = [entry[key] for entry in data.values()]

        vmin = min(values)
        vmax = max(values)

        stats[key] = {"min": vmin, "max": vmax, "range": vmax - vmin}

    return stats


def init_output_data_dict(var_names):
    """
    Initialize a dictionary used to accumulate variable values across experiments.

    Parameters
    ----------
    var_names : list[str]
        Names of the variables to be stored.

    Returns
    -------
    dict[str, list[np.ndarray]]
        Dictionary mapping each variable name to an initially empty list. Each
        list is intended to store one NumPy array per experiment.
    """
    return {name: [] for name in var_names}


def stack_variable(data_dict, var_name, data):
    """
    Append data from a single experiment to the corresponding variable entry.

    Parameters
    ----------
    data_dict : dict[str, list[np.ndarray]]
        Dictionary storing the accumulated data for each variable.
    var_name : str
        Name of the variable to which the data will be appended.
    data : array-like
        Data vector associated with the current experiment.

    Returns
    -------
    None
    """
    if var_name not in data_dict:
        raise KeyError(f"Variable '{var_name}' not initialized")

    data_dict[var_name].append(np.asarray(data))


def check_value(value, new_value, metric='min'):
    """
    Return the minimum or maximum of two values according to the specified metric.

    Parameters
    ----------
    value : float
        Current reference value.
    new_value : float
        New value to compare against the current reference.
    metric : {"min", "max"}
        Comparison criterion. If ``"min"``, the smaller value is returned. If
        ``"max"``, the larger value is returned.

    Returns
    -------
    float
        The selected value according to the specified comparison criterion.

    Raises
    ------
    ValueError
        If ``metric`` is not ``"min"`` or ``"max"``.
    """

    if metric == "min":
        return min(value, new_value)
    elif metric == "max":
        return max(value, new_value)
    else:
        raise ValueError(f"Wrong metric: {metric!r}. Expected 'min' or 'max'.")


def find_geometry(dict_obj, exp_to_find):
    """
    Return the geometry filename associated with a given experiment folder.

    Parameters
    ----------
    dict_obj : list[dict]
        List of dictionaries describing the imported geometries. Each dictionary
        is expected to contain at least the keys ``"geometry"`` and ``"folder"``,
        for example::

            {
                "geometry": "case_0000.vtk",
                "folder": "exp_1",
                "bounding_box": [...]
            }

    exp_to_find : str
        Name of the experiment folder to search for (e.g. ``"exp_1"``).

    Returns
    -------
    str or None
        Geometry filename associated with the specified experiment folder, or
        ``None`` if no matching entry is found.
    """
    for entry in dict_obj:
        if entry["folder"] == exp_to_find:
            return entry["geometry"]
    return None


def read_ini_as_dict(filepath, max_entries=None):
    """
    Read an INI file into a nested dictionary.

    Each section of the INI file is converted into a dictionary whose keys are
    the parameter names and whose values are converted to ``float``.

    Parameters
    ----------
    filepath : str or pathlib.Path
        Path to the INI file.

    max_entries : int, optional
        Maximum number of sections to read from the file. If ``None`` (default),
        all sections are processed.

    Returns
    -------
    dict
        Nested dictionary of the form::

            {
                "geom_000": {
                    "B1": 117.4,
                    "B2": 69.97,
                    ...
                },
                "geom_001": {
                    "B1": 67.3,
                    "B2": 23.83,
                    ...
                },
                ...
            }

    Notes
    -----
    - Section names are preserved exactly as they appear in the INI file.
    - Option names are case-sensitive because ``config.optionxform`` is set to
      the identity function.
    - All parameter values are converted to ``float``. A ``ValueError`` will be
      raised if a value cannot be converted.
    """

    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(filepath)

    result = {}

    sections = config.sections()

    if max_entries is not None:
        sections = sections[:max_entries]

    for section in sections:
        result[section] = {}

        for key, value in config.items(section):
            result[section][key] = float(value)

    return result


def similarity_dictionary_entry(new_label, cluster_id, cluster_size, new_centroid_distance,
                                cluster_centroid_distance_percentile, radius_ratio, cluster_mahalanobis_distance,
                                mahal_percentile, cluster_distances,
                                novelty_ratio, novelty_ratio_z, nn_isolation_percentile, centroid_ratio):
    exp_result = {
        "experiment": str(new_label),

        "cluster": {
            "nearest_cluster": cluster_id,
            "population": cluster_size,
            "distance_to_centroid": new_centroid_distance,
            "centroid_distance_percentile": cluster_centroid_distance_percentile,
            "distance_to_radius_ratio": radius_ratio,
            "mahalanobis_distance": cluster_mahalanobis_distance,
            "mahalanobis_distance_percentile": mahal_percentile,
            "centroid_distances": {str(i): float(d) for i, d in enumerate(cluster_distances)}
        },

        "novelty": {
            "nearest_neighbor_ratio": float(novelty_ratio),
            "nearest_neighbor_zscore": float(novelty_ratio_z),
            "nearest_neighbor_percentile": float(nn_isolation_percentile),
            "centroid_ratio": float(np.asarray(centroid_ratio).item())
        }
    }
    return exp_result


# ------------------------------------------------------------------------------------------------------------ Distances
def evaluate_distance_metrics(logger, D_eval, D_training, new_experiment, training_experiments):
    """
    Evaluate experiment-to-experiment distance metrics for a new experiment.

    The function assigns the new experiment to its nearest k-means cluster and
    computes a set of geometric metrics describing its position relative to the
    training data in that cluster. These metrics include the Euclidean distance to
    the assigned cluster centroid, the corresponding percentile within the training
    cluster, the ratio between the centroid distance and the cluster radius, the
    Mahalanobis distance, and its percentile. Distances to all cluster centroids
    are also returned.

    Parameters
    ----------
    logger : logging.Logger
        Logger instance used to report the computed metrics.
    D_eval : np.ndarray
        Distance of the new experiment from all the existing ones
        ``(1, n_experiments)``.
    D_training : np.ndarray
        Distance across all the existing experiments
        ``(n_experiments, n_experiments)``.
    new_experiment : np.ndarray
        Latent representation of the experiment to evaluate, with shape
        ``(1, n_features)`` or ``(n_features,)``.
    training_experiments : np.ndarray
        Latent representations of the training experiments, with shape
        ``(n_training, n_features)``.

    Returns
    -------
    tuple
        Tuple containing:
        - **D_eval[0]** (*int*): (*np.ndarray*): Euclidean distances from the experiment to all other experiments.
        - **mean_training_distance** (*float*): Average eucidean distance between the experiments in the training dataset.
        - **novelty_ratio** (*float*): Ratio between distance new experiment from NN and average training distance.
        - **novelty_ratio_z** (*float*): Novelty ratio normalised over standard deviation of experiment-to-experiment distance.
        - **nn_isolation_percentile** (*float*): Percentage of experiments that have nn-distance closer than the new one.
        - **new_centroid_dist** (*float*): Distance of new experiment from centroid of training experiments.
        - **centroid_ratio** (*float*): ratio new_centroid_dist vs average centroid distance.
    """

    # distance from new experiment to existing ones
    new_nn = np.min(D_eval)

    # average distance between training experiments
    mean_training_distance = np.mean(D_training[np.triu_indices_from(D_training, k=1)])

    D_tmp = D_training.copy()
    np.fill_diagonal(D_tmp, np.inf)

    training_nn = np.min(D_tmp, axis=1)

    mean_training_nn = np.mean(training_nn)
    std_training_nn = np.std(training_nn)

    # Nearest-neighbor ratio = (distance form closest exp) / (average distance among training)
    novelty_ratio = new_nn / mean_training_nn

    # | Ratio     | Interpretation                      |
    # | -------   | ----------------------------------- |
    # | < 1.2     | familiar geometry                   |
    # | 1.2 - 1.5 | likely interpolation                |
    # | 1.5 - 2   | edge of explored domain             |
    # | > 2       | likely extrapolation                |
    # | > 3       | highly novel geometry               |
    logger.info(f"\tNovelty_ratio: {novelty_ratio:>4.2e}")

    # Z-score normalization, as above but normalized for std
    novelty_ratio_z = (new_nn - mean_training_nn) / std_training_nn

    # | Z-score | Interpretation             |
    # | ------- | -------------------------- |
    # | < 0     | familiar region            |
    # | 0-1     | normal                     |
    # | 1-2     | uncommon                   |
    # | 2-3     | rare                       |
    # | > 3     | potentially outside domain |
    logger.info(f"\tNovelty_ratio_z: {novelty_ratio_z:>4.2e}")

    # Percentile score
    # 10%  -> closer than almost all training geometries
    # 50%  -> typical
    # 75–90%: relatively isolated region.
    # 90%  -> more isolated than 90% of the training set
    # 99%  -> extreme novelty
    nn_isolation_percentile = 100 * np.mean(training_nn < new_nn)
    logger.info(f"\tNearest-neighbor isolation percentile: {nn_isolation_percentile:>4.2e}")

    # centroid of training space
    centroid = np.mean(training_experiments, axis=0)
    training_centroid_dist = np.linalg.norm(training_experiments - centroid, axis=1)
    new_centroid_dist = np.linalg.norm(new_experiment - centroid, axis=1)

    centroid_ratio = new_centroid_dist / np.mean(training_centroid_dist)
    # the greater, the more different from average geometry
    logger.info(f"\tCentroid_ratio: {centroid_ratio[0]:>4.2e}")

    # | NN ratio | Centroid ratio | Interpretation                          |
    # | -------- | -------------- | --------------------------------------- |
    # | low      | low            | central interpolation                   |
    # | low      | high           | interpolation near a peripheral cluster |
    # | high     | low            | sparse interior region                  |
    # | high     | high           | likely extrapolation                    |

    return D_eval[0], mean_training_distance, novelty_ratio, novelty_ratio_z, nn_isolation_percentile, \
           new_centroid_dist, centroid_ratio[0]


# --------------------------------------------------------------------------------------------------- Similarity metrics
def find_most_isolated(distance_matrix):
    """
    Identify the most isolated geometries based on a pairwise-distance matrix.

    Two complementary isolation metrics are evaluated:

    1. **Average distance**: the geometry with the largest mean distance to all
       other geometries.
    2. **Maximum minimum distance (max-min)**: the geometry whose closest
       neighbour is farther away than the closest neighbour of any other
       geometry.

    Parameters
    ----------
    distance_matrix : numpy.ndarray of shape (n_geometries, n_geometries)
        Symmetric pairwise distance matrix. The diagonal is assumed to contain
        the self-distance (typically zero).

    Returns
    -------
    idx_avg : int
        Index of the geometry with the largest average distance to all other
        geometries.

    avg_distances : numpy.ndarray of shape (n_geometries,)
        Average distance of each geometry to all the others, excluding the
        diagonal.

    idx_max_min : int
        Index of the geometry whose nearest neighbour is the most distant
        (i.e. the maximum of the minimum pairwise distances).

    min_distances : numpy.ndarray of shape (n_geometries,)
        Minimum distance from each geometry to any other geometry.

    Notes
    -----
    The average-distance criterion identifies globally isolated geometries,
    whereas the max-min criterion identifies geometries that are locally
    isolated from their nearest neighbour. The two criteria may select
    different geometries.
    """

    # Ignore self-distance (diagonal = 0)
    N = distance_matrix.shape[0]

    # Compute average distance per row (excluding diagonal)
    avg_distances = (np.sum(distance_matrix, axis=1) - np.diag(distance_matrix)) / (N - 1)

    # Find index of most isolated geometry on average
    idx_avg = np.argmax(avg_distances)

    D_tmp = distance_matrix.copy()
    np.fill_diagonal(D_tmp, np.inf)

    # Find min
    min_distances = np.min(D_tmp, axis=1)

    # Find max of min
    idx_max_min = np.argmax(min_distances)

    return idx_avg, avg_distances, idx_max_min, min_distances


def get_n_closest_indices(all_distances, n_closest=1, exclude_self_idx=None):
    """
    Return the indices of the nearest neighbours in a distance vector.

    Parameters
    ----------
    all_distances : numpy.ndarray of shape (n_samples,)
        One-dimensional array containing the distances from a reference sample
        to all samples in the dataset.

    n_closest : int, optional
        Number of nearest neighbours to return. Defaults to 1.

    exclude_self_idx : int or None, optional
        Index to exclude from the search, typically the index of the reference
        sample itself. When provided, the corresponding distance is temporarily
        set to infinity, so it cannot be selected. Defaults to None.

    Returns
    -------
    numpy.ndarray of shape (n_closest,)
        Indices of the `n_closest` smallest distances, sorted in ascending
        order of distance.

    Notes
    -----
    The function assumes that smaller values correspond to more similar
    samples. If multiple samples have the same distance, their relative order
    is determined by ``numpy.argsort``.
    """

    distances_ = all_distances.copy()

    # Optionally ignore self-distance
    if exclude_self_idx is not None:
        distances_[exclude_self_idx] = np.inf

    # Get sorted indices and take first N
    closest_indices = np.argsort(distances_)[:n_closest]

    return closest_indices


def find_least_isolated(distance_matrix):
    """
    Identify the most representative geometries in a dataset.

    Two complementary centrality criteria are evaluated:

    1. **Medoid**: the geometry with the smallest average distance to all
       other geometries.
    2. **Minimax centre**: the geometry whose maximum distance to any other
       geometry is minimal.

    Parameters
    ----------
    distance_matrix : numpy.ndarray of shape (n_geometries, n_geometries)
        Symmetric pairwise distance matrix. The diagonal is assumed to contain
        the self-distance (typically zero).

    Returns
    -------
    medoid_idx : int
        Index of the medoid, i.e. the geometry with the minimum average
        distance to all other geometries.

    medoid_mean_distance : float
        Average distance of the medoid to all other geometries.

    minimax_idx : int
        Index of the geometry that minimizes the maximum distance to any other
        geometry.

    minimax_max_distance : float
        Maximum distance from the minimax geometry to any other geometry.

    Notes
    -----
    The medoid minimizes the average distance to the dataset, while the
    minimax centre minimizes the worst-case distance to any other geometry.
    These criteria may identify different geometries.
    """

    mean_distances = np.mean(distance_matrix, axis=1)
    medoid_idx = np.argmin(mean_distances)

    max_distances = np.max(distance_matrix, axis=1)
    minimax_idx = np.argmin(max_distances)

    return medoid_idx, mean_distances[medoid_idx], minimax_idx, max_distances[minimax_idx]


def distance_matrix_sorting(sorting_type='first_PCA_component', latent_space=None, D=None, labels=None):
    """
    Reorder a distance matrix according to different sorting strategies.

    This function computes a permutation of the samples and applies it to both
    rows and columns of a square distance matrix, producing a reordered matrix
    that may reveal clusters or other structures more clearly when visualized.

    Parameters
    ----------
    sorting_type : str, optional
        Strategy used to determine the ordering of the samples. Supported
        values are:

        - ``'first_PCA_component'``:
          Sort samples according to the first coordinate of the latent-space
          representation. Requires ``latent_space``.
        - ``'hierarchical_clustering'``:
          Sort samples according to the leaf ordering produced by average-linkage
          hierarchical clustering. Requires ``D``.
        - ``'optimal_leaf'``:
          Perform average-linkage hierarchical clustering followed by optimal
          leaf ordering to minimize the distances between adjacent leaves.
          Requires ``D``.

    latent_space : ndarray of shape (n_samples, n_features), optional
        Latent-space representation of the samples. Only required when
        ``sorting_type='first_PCA_component'``.

    D : ndarray of shape (n_samples, n_samples), optional
        Symmetric distance matrix between samples. Required for hierarchical
        clustering methods.

    labels : sequence of length n_samples, optional
        Labels associated with the samples. These are reordered consistently
        with the distance matrix.

    Returns
    -------
    D_sorted : ndarray of shape (n_samples, n_samples)
        Distance matrix reordered according to the selected sorting strategy.

    labels_sorted : list
        Labels reordered according to the computed sample ordering.

    Notes
    -----
    - The function assumes that ``D`` is a symmetric distance matrix.
    - For ``'optimal_leaf'``, the distance matrix is converted to condensed
      form using ``scipy.spatial.distance.squareform`` before clustering.
    - No validation is performed to ensure that the required inputs are
      provided for the selected sorting strategy.
    """

    if sorting_type == 'first_PCA_component':
        if latent_space is None:
            raise ValueError("latent_space must be provided.")

        # sort by first PCA component
        order = np.argsort(latent_space[:, 0])

    elif sorting_type == 'hierarchical_clustering':
        if D is None:
            raise ValueError("D must be provided.")

        # Hierarchical clustering
        from scipy.cluster.hierarchy import linkage, leaves_list
        Z = linkage(D, method='average')
        order = leaves_list(Z)

    elif sorting_type == 'optimal_leaf':
        if D is None:
            raise ValueError("D must be provided.")

        # Optimal leaf ordering
        from scipy.cluster.hierarchy import linkage, optimal_leaf_ordering, leaves_list
        from scipy.spatial.distance import squareform

        condensed = squareform(D)
        Z = linkage(condensed, method='average')
        Z_opt = optimal_leaf_ordering(Z, condensed)
        order = leaves_list(Z_opt)

    D_sorted = D[order][:, order]
    labels_sorted = [labels[i] for i in order]

    return D_sorted, labels_sorted


def reconstruct_mesh(logger, pca_, latent_, index_, n_grids_x_, n_grids_y_,
                     n_grids_z_, sdfs_, source_mesh_, grid_size_,
                     make_plot_=True, title_=None):
    """
    Reconstruct a mesh from its PCA latent representation.

    The latent vector is projected back into the original signed distance field
    (SDF) space using the inverse PCA transformation. The reconstructed SDF is
    reshaped into its three-dimensional grid, optionally smoothed using a
    Gaussian filter, and converted into a surface mesh through the marching-cubes
    algorithm.

    When a reference mesh is provided, reconstruction quality is evaluated
    using both the SDF reconstruction error and the Hausdorff distance between
    the reconstructed and reference meshes. Optionally, the reconstructed mesh,
    reference mesh, and a 'vertex-wise' error heatmap can be displayed.

    Parameters
    ----------
    logger : logging.Logger
        Logger used to report reconstruction statistics and diagnostics.

    pca_ : sklearn.decomposition.PCA
        Fitted PCA model used to reconstruct the signed distance field.

    latent_ : numpy.ndarray of shape (1, n_components)
        Latent-space coordinates to reconstruct.

    index_ : int or None
        Index of the corresponding reference SDF in ``sdfs_``. If provided,
        the reconstruction mean squared error is computed.

    n_grids_x_ : int
        Number of grid points along the x-axis.

    n_grids_y_ : int
        Number of grid points along the y-axis.

    n_grids_z_ : int
        Number of grid points along the z-axis.

    sdfs_ : numpy.ndarray
        Collection of original SDFs used to compare the reconstructed field
        against the reference sample.

    source_mesh_ : open3d.geometry.TriangleMesh or None
        Reference mesh corresponding to the reconstructed sample. If provided,
        the Hausdorff distance and an optional error heatmap are computed.

    grid_size_ : float
        Uniform spacing between adjacent grid points used during SDF
        generation.

    make_plot_ : bool, optional
        If ``True``, display the reconstructed mesh, the reference mesh (when
        available), and a reconstruction error heatmap. Default is ``True``.

    title_ : str or None, optional
        Window title used for visualization. If ``None``, a default title is
        generated.

    Returns
    -------
    open3d.geometry.TriangleMesh
        Reconstructed mesh extracted from the reconstructed signed distance
        field.

    Notes
    -----
    The reconstruction workflow consists of:

    1. Applying the inverse PCA transform to recover the weighted SDF.
    2. Reshaping the SDF into its original three-dimensional grid.
    3. Applying a small Gaussian smoothing filter to reduce discretization
       artifacts.
    4. Extracting the zero level-set using the marching-cubes algorithm.
    5. Scaling and translating the resulting vertices to recover the original
       physical dimensions.

    If a reference mesh is available, the function additionally computes:

    - Mean squared reconstruction error in SDF space.
    - Hausdorff distance between reconstructed and reference meshes.
    - Optional vertex-wise error visualization using a color map.
    """

    reconstructed = pca_.inverse_transform(latent_)
    reconstructed_sdf = reconstructed.reshape(n_grids_x_, n_grids_y_, n_grids_z_)

    from scipy.ndimage import gaussian_filter
    reconstructed_sdf = gaussian_filter(reconstructed_sdf, sigma=0.1)

    # Extract surface
    verts, faces, _, _ = measure.marching_cubes(reconstructed_sdf, level=0.0)

    dx = grid_size_

    verts_rescaled = np.zeros_like(verts)
    verts_rescaled[:, 0] = verts[:, 0] * dx
    verts_rescaled[:, 1] = verts[:, 1] * dx
    verts_rescaled[:, 2] = verts[:, 2] * dx

    # Center mesh in the original coordinate system
    shift = np.array([dx * n_grids_x_ / 2.0, dx * n_grids_y_ / 2.0, dx * n_grids_z_ / 2.0])
    verts_rescaled -= shift

    mesh_reco = to_o3d_mesh(verts_rescaled, faces)
    logger.debug(f"\tReconstructed mesh. Vertices: {verts.shape}, faces: {faces.shape}")

    mesh_orig = None
    if source_mesh_:
        mesh_orig = to_o3d_mesh(source_mesh_.vertices, source_mesh_.triangles)

    if index_ is not None:
        original_sdf = sdfs_[index_].reshape(n_grids_x_, n_grids_y_, n_grids_z_)

        logger.debug(f"\toriginal_sdf min/max: {original_sdf.min()}, {original_sdf.max()}")
        logger.debug(f"\treconstructed_sdf min/max: {reconstructed_sdf.min()}, {reconstructed_sdf.max()}")

        error = np.mean((sdfs_[index_] - reconstructed.flatten()) ** 2)
        logger.info(f"Reconstruction MSE: {error}")

    if mesh_orig is not None:
        hd = mesh_hausdorff(mesh_orig, mesh_reco)
        logger.info(f"Hausdorff distance: {hd}")

    if make_plot_:

        mesh_reco.paint_uniform_color([1, 0, 0])  # red
        meshes = [mesh_reco]

        window_name = title_ if title_ else "Reconstructed (R)"

        if mesh_orig is not None:
            mesh_orig.paint_uniform_color([0, 0, 1])  # blue
            meshes.append(mesh_orig)
            window_name += ", Original (B)"

            mesh_error, distances = sdf_plot.color_mesh_by_error(mesh_reco, mesh_orig)

            o3d.visualization.draw_geometries([mesh_error], window_name="Error heatmap")

        o3d.visualization.draw_geometries(meshes, window_name=window_name, mesh_show_wireframe=True)

    return mesh_reco


def evaluate_new_experiment_similarity(logger, case, new_values, new_label, pca, latent, labels,
                                       n_clusters, kmeans, cluster_labels, cluster_centroids, cluster_ids,
                                       sorting_logic,
                                       scaler=None,
                                       make_plot_distance_matrix=False,
                                       plot_pca_foreach_new_experiment=False,
                                       images_folder=None):
    if scaler is not None:
        new_latent = project_new_experiment(logger, scaler, new_values, pca)
    else:
        new_latent = new_values

    # pca is fitted PCA object
    explained_variance = pca.explained_variance_ratio_  # fraction of variance per component
    # Cumulative variance
    cumulative_variance = np.cumsum(explained_variance)

    # analyse new experiment from cluster point of view
    cluster_id, cluster_size, new_centroid_distance, cluster_centroid_distance_percentile, radius_ratio, \
    cluster_mahalanobis_distance, mahal_percentile, cluster_distances = \
        evaluate_cluster_metrics(logger, n_clusters, kmeans,
                                 cluster_labels,
                                 cluster_centroids,
                                 new_latent,
                                 latent)

    # append new cluster to existing ones
    cluster_ids.append(cluster_id)

    D_training = pairwise_distances(latent, metric='euclidean')

    latent_all = np.vstack((latent, new_latent))
    labels_all = labels.copy()
    labels_all.append(new_label)

    # distance between experiments in training set, D_eval.shape = (n_train, n_train)
    D_all = pairwise_distances(latent_all, metric='euclidean')
    D_all = 0.5 * (D_all + D_all.T)
    D_all_sorted, labels_all_sorted = distance_matrix_sorting(sorting_type=sorting_logic,
                                                              latent_space=latent_all,
                                                              D=D_all,
                                                              labels=labels_all)

    # distance between training set and new, D_eval.shape = (n_eval, n_train)
    D_eval = pairwise_distances(new_latent, latent, metric='euclidean')

    if make_plot_distance_matrix:
        sdf_plot.plot_distance_matrix(D_all, title_=f"Evaluation: {new_label} in Distance Matrix (PCA space)",
                                      labels=labels_all, highlight_labels=[new_label],
                                      save_folder=images_folder)

        sdf_plot.plot_distance_matrix(D_all_sorted, title_=f"Evaluation: {new_label} in Sorted ({sorting_logic}) Distance Matrix (PCA space)",
                                      labels=labels_all_sorted, highlight_labels=[new_label],
                                      save_folder=images_folder)

    # collection of metrics as distances in latent space
    distances, mean_training_distance, novelty_ratio, novelty_ratio_z, nn_isolation_percentile, new_centroid_dist, centroid_ratio = \
        evaluate_distance_metrics(logger, D_eval, D_training, new_latent, latent)

    if plot_pca_foreach_new_experiment:
        sdf_plot.plot_pca_projection(latent, new_latent,
                                     training_labels=labels, new_labels=[new_label],
                                     cluster_labels=cluster_labels, new_cluster_labels=[cluster_id],
                                     annotate=True,
                                     title=f"Evaluation: {new_label} in training latent space\n(PCA1+PCA2 = {cumulative_variance[1]:>4.2e})",
                                     case=case, save_folder=images_folder)

    exp_result = similarity_dictionary_entry(new_label, cluster_id, cluster_size, new_centroid_distance,
                                             cluster_centroid_distance_percentile, radius_ratio,
                                             cluster_mahalanobis_distance,
                                             mahal_percentile, cluster_distances,
                                             novelty_ratio, novelty_ratio_z, nn_isolation_percentile, centroid_ratio)

    return distances, exp_result


# ------------------------------------------------------------------------------------------------------------- Sampling
def simplified_maximim_sampling(X_existing, bounds_, n_candidates=10000):
    """
    Generate a new sample using a simplified Maximin sampling strategy.

    A set of candidate points is generated uniformly at random within the
    specified bounds. The candidate whose minimum Euclidean distance to the
    existing samples is largest is selected as the next sample.

    Parameters
    ----------
    X_existing : ndarray of shape (n_samples, n_dimensions)
        Coordinates of the existing samples.

    bounds_ : sequence of tuple(float, float)
        Lower and upper bounds for each dimension as
        ``[(min_1, max_1), ..., (min_d, max_d)]``.

    n_candidates : int, optional
        Number of random candidate points to generate. Increasing this value
        generally improves the quality of the selected point at the expense of
        computational cost. Default is 10000.

    Returns
    -------
    best_point : ndarray of shape (n_dimensions,)
        Coordinates of the selected sample.

    best_distance : float
        Minimum Euclidean distance between the selected sample and the existing
        samples.

    Notes
    -----
    This is an approximate Maximin algorithm. Rather than solving the
    optimization problem exactly, it evaluates a finite set of randomly
    generated candidate points and selects the one maximizing the distance to
    its nearest existing neighbor.
    """

    d = X_existing.shape[1]

    # 1. Sample candidate points (uniform)
    X_candidates = np.random.rand(n_candidates, d)

    # scale to bounds
    for i in range(d):
        low, high = bounds_[i]
        X_candidates[:, i] = low + (high - low) * X_candidates[:, i]

    # 2. Distance from candidates to existing points
    distances_ = cdist(X_candidates, X_existing)

    # 3. Distance to nearest neighbor
    min_dist = np.min(distances_, axis=1)

    # 4. Pick the farthest
    best_idx = np.argmax(min_dist)
    best_point = X_candidates[best_idx]

    return best_point, min_dist[best_idx]


def pick_random_and_neighbors(D_sorted, labels_sorted):
    """
    Pick a random geometry from the sorted distance matrix and
    return its closest neighbors in the sorted ordering.

    If the selected index is:
        - in the middle: return previous and next geometries
        - at an edge: return only the available neighbor

    Parameters
    ----------
    D_sorted : np.ndarray
        Sorted distance matrix (NxN)

    labels_sorted : list
        Labels corresponding to sorted geometries

    Returns
    -------
    dict
        Information about selected geometry and neighbors
    """

    n = len(labels_sorted)

    if n < 2:
        raise ValueError("Need at least 2 geometries")

    # pick random index
    idx = np.random.randint(0, n)

    result = {"selected_index": idx, "selected_label": labels_sorted[idx], "neighbors": []}

    # left edge
    if idx == 0:
        result["neighbors"].append({"index": 1, "label": labels_sorted[1], "distance": D_sorted[idx, 1]})

    # right edge
    elif idx == n - 1:
        result["neighbors"].append({"index": n - 2, "label": labels_sorted[n - 2], "distance": D_sorted[idx, n - 2]})

    # middle
    else:
        result["neighbors"].append({"index": idx - 1, "label": labels_sorted[idx - 1],
                                    "distance": D_sorted[idx, idx - 1]})
        result["neighbors"].append({"index": idx + 1, "label": labels_sorted[idx + 1],
                                    "distance": D_sorted[idx, idx + 1]})

    return result


# ----------------------------------------------------------------------------------------------------------------- Mesh
def o3d_to_trimesh(mesh_o3d):
    """
    Convert an Open3D triangle mesh into a Trimesh mesh.

    Parameters
    ----------
    mesh_o3d : open3d.geometry.TriangleMesh
        Input Open3D triangle mesh.

    Returns
    -------
    trimesh.Trimesh
        Equivalent Trimesh object containing the same vertices and triangle
        connectivity. Automatic processing is disabled (``process=False``),
        so the mesh topology and vertex ordering are preserved.

    Notes
    -----
    The returned mesh shares the same geometry as the input mesh but is
    represented using the Trimesh data structure, allowing access to the
    functionality provided by the Trimesh library.
    """

    vertices = np.asarray(mesh_o3d.vertices)
    faces = np.asarray(mesh_o3d.triangles)
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def sample_points(mesh, n_points=10000):
    """
    Uniformly sample points on the surface of a triangular mesh.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input mesh to be sampled.

    n_points : int, optional
        Number of points to sample from the mesh surface. Default is 10000.

    Returns
    -------
    ndarray of shape (n_points, 3)
        Cartesian coordinates of the sampled surface points.

    Notes
    -----
    Sampling is performed with ``trimesh.sample.sample_surface()``, which
    selects triangles with probability proportional to their surface area and
    then samples uniformly within each selected triangle. Consequently, the
    returned points are approximately uniformly distributed over the mesh
    surface.
    """

    points, _ = trimesh.sample.sample_surface(mesh, n_points)
    return points


def hausdorff_distance(points_a, points_b):
    """
    Compute the symmetric Hausdorff distance between two point clouds.

    For each point cloud, the distance to its nearest neighbor in the other
    point cloud is computed. The returned value is the sum of the squared
    maximum nearest-neighbor distances in both directions.

    Parameters
    ----------
    points_a : ndarray of shape (n_points_a, 3)
        First point cloud.

    points_b : ndarray of shape (n_points_b, 3)
        Second point cloud.

    Returns
    -------
    float
        Symmetric Hausdorff distance computed as

            max(A → B)^2 + max(B → A)^2

        where ``A → B`` denotes the nearest-neighbor distances from points in
        ``A`` to points in ``B``.

    Notes
    -----
    Nearest-neighbor searches are accelerated using
    ``scipy.spatial.cKDTree``.
    """

    tree_a = cKDTree(points_a)
    tree_b = cKDTree(points_b)

    dist_a, _ = tree_b.query(points_a)  # A → B
    dist_b, _ = tree_a.query(points_b)  # B → A

    # use Hausdorff distance (max) rather than mean
    cd = np.max(dist_a ** 2) + np.max(dist_b ** 2)
    return cd


def mesh_hausdorff(mesh1, mesh2, n_points=10000):
    """ Compute the symmetric Hausdorff distance between two surface meshes.

    The meshes are first uniformly sampled to generate point clouds. The
    symmetric Hausdorff distance is then evaluated on the sampled points to
    quantify the maximum geometric deviation between the two surfaces.

    Parameters
    ----------
     mesh1 : open3d.geometry.TriangleMesh or trimesh.Trimesh
        First input mesh.

    mesh2 : open3d.geometry.TriangleMesh or trimesh.Trimesh
        Second input mesh.

    n_points : int, optional
        Number of points to sample from each mesh surface. Increasing this
        value generally improves the accuracy of the approximation at the
        expense of computational cost. Default is 10000.

    Returns
    -------
    float
        Approximate symmetric Hausdorff distance between the two meshes,
        computed from the sampled point clouds.

    Notes
    -----
    - If the input meshes are Open3D meshes, they are first converted to
    ``trimesh.Trimesh`` objects.
    - Surface sampling is performed uniformly with respect to triangle area.
    - The returned distance is an approximation whose accuracy depends on the
    number of sampled points.
    """

    # Convert if needed
    if not isinstance(mesh1, trimesh.Trimesh):
        mesh1 = o3d_to_trimesh(mesh1)
    if not isinstance(mesh2, trimesh.Trimesh):
        mesh2 = o3d_to_trimesh(mesh2)

    pts1 = sample_points(mesh1, n_points)
    pts2 = sample_points(mesh2, n_points)

    return hausdorff_distance(pts1, pts2)


def compute_vertex_errors(mesh_a, mesh_b):
    """
    Compute nearest-neighbor distances between the vertices of two meshes.

    For each vertex of ``mesh_a``, the Euclidean distance to the closest
    vertex of ``mesh_b`` is evaluated using a k-d tree.

    Parameters
    ----------
    mesh_a : open3d.geometry.TriangleMesh
        Source mesh whose vertices are evaluated.

    mesh_b : open3d.geometry.TriangleMesh
        Reference mesh whose vertices are used for the nearest-neighbor search.

    Returns
    -------
    numpy.ndarray of shape (n_vertices,)
        Euclidean distance from each vertex of ``mesh_a`` to its nearest
        vertex in ``mesh_b``.

    Notes
    -----
    This function computes **vertex-to-vertex** distances, not the true
    point-to-surface distance. Consequently, the returned distances depend on
    the mesh discretization and may overestimate the actual geometric error
    for coarse or non-uniform meshes.

    Nearest-neighbor queries are accelerated using
    ``scipy.spatial.cKDTree``.
    """

    verts_a = np.asarray(mesh_a.vertices)
    verts_b = np.asarray(mesh_b.vertices)

    tree = cKDTree(verts_b)
    distances_, _ = tree.query(verts_a)

    return distances_


# ----------------------------------------------------------------------------------------------------------------- Maps
def evaluate_SOM(latent, som_size=5, seed_=standard_seed):
    """
    Train a Self-Organizing Map (SOM) from latent vectors and assign each
    sample to its best matching unit (BMU).

    The latent vectors are first standardized to zero mean and unit variance.
    A square SOM is then initialized using PCA-based weight initialization and
    trained using random sampling. The function returns the coordinates of the
    winning neuron for each input sample.

    Parameters
    ----------
    latent : numpy.ndarray of shape (n_samples, n_features)
        Latent-space representation of the data (e.g., PCA coefficients).

    som_size : int, optional
        Number of neurons along each dimension of the square SOM grid. The
        resulting map contains ``som_size × som_size`` neurons. Default is 5.

    seed_ : int, optional
        Random seed used for reproducible initialization and training.
        Defaults to ``standard_seed``.

    Returns
    -------
    numpy.ndarray of shape (n_samples, 2)
        Coordinates ``(row, column)`` of the best matching unit (BMU) for each
        input sample.

    Notes
    -----
    - Input data are standardized using ``sklearn.preprocessing.StandardScaler``
      before training.
    - The SOM weights are initialized using PCA
      (``MiniSom.pca_weights_init``), which generally accelerates convergence.
    - Training is performed for 10,000 iterations using
      ``MiniSom.train_random``.
    """

    scaler = StandardScaler()
    latent_scaled = scaler.fit_transform(latent)

    np.random.seed(seed_)

    som = MiniSom(x=som_size, y=som_size, input_len=latent_scaled.shape[1],
                  sigma=1.0, learning_rate=0.2, random_seed=seed_)

    som.pca_weights_init(latent_scaled)
    som.train_random(latent_scaled, 10000)

    winners = np.array([som.winner(x) for x in latent_scaled])
    return winners


def evaluate_UMAP(latent, n_neighbors=10, min_dist=0.1, n_components=2, metric='euclidean', seed=standard_seed):
    """
    Compute a UMAP embedding of a latent-space representation.

    The input latent vectors are first standardized to zero mean and unit
    variance before applying Uniform Manifold Approximation and Projection
    (UMAP). Both the fitted UMAP reducer and the fitted scaler are returned
    alongside the embedding so that new samples can later be projected into
    the same embedding space.

    Parameters
    ----------
    latent : numpy.ndarray of shape (n_samples, n_features)
        Latent-space representation of the data (e.g., PCA coefficients).

    n_neighbors : int, optional
        Number of neighboring samples used by UMAP to construct the local
        manifold approximation. Smaller values emphasize local structure,
        while larger values preserve more global structure. Default is 10.

    min_dist : float, optional
        Minimum distance between embedded points. Smaller values produce
        tighter clusters, whereas larger values yield a more uniform
        distribution. Default is 0.1.

    n_components : int, optional
        Dimensionality of the embedding. Default is 2.

    metric : str, optional
        Distance metric used by UMAP in the input space. Default is
        ``'euclidean'``.

    seed : int, optional
        Random seed used for reproducibility. Defaults to
        ``standard_seed``.

    Returns
    -------
    embedding : numpy.ndarray of shape (n_samples, n_components)
        Low-dimensional embedding produced by UMAP.

    reducer : umap.UMAP
        Fitted UMAP model, which can be used to transform new samples.

    scaler : sklearn.preprocessing.StandardScaler
        Fitted scaler used to standardize the input latent vectors prior to
        UMAP embedding.

    Notes
    -----
    Standardizing the latent vectors before UMAP generally improves the
    embedding by ensuring that all latent dimensions contribute on a
    comparable scale.
    """

    scaler = StandardScaler()
    latent_scaled = scaler.fit_transform(latent)

    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components,
                        metric=metric, random_state=seed)

    embedding = reducer.fit_transform(latent_scaled)
    return embedding, reducer, scaler


def build_multiple_UMAPs(data_array_scaled):
    """
    Compute multiple UMAP embeddings using different neighborhood sizes.

    Three UMAP embeddings are generated using slightly different values of
    ``n_neighbors`` around a reference value determined from the number of
    samples. Consecutive embeddings are aligned using Procrustes analysis to
    remove arbitrary rotations, reflections, and scaling, making them directly
    comparable.

    Parameters
    ----------
    data_array_scaled : ndarray of shape (n_samples, n_features)
        Scaled input data used to compute the UMAP embeddings.

    Returns
    -------
    all_embedding_parameters : list of ndarray
        List containing the aligned 2D UMAP embeddings. Each array has shape
        ``(n_samples, 2)``.

    all_embedding_labels : list of str
        Labels corresponding to each embedding. Each label is the value of
        ``n_neighbors`` used to generate the embedding.

    Notes
    -----
    The reference number of neighbors is computed as

        ``max(5, n_samples // 10)``

    and the three embeddings are generated using

        - ``n_neighbors - 2``
        - ``n_neighbors``
        - ``n_neighbors + 2``

    Consecutive embeddings are aligned using Procrustes analysis to facilitate
    visual comparison across different neighborhood sizes.
    """

    n_neighbors_params = max(5, data_array_scaled.shape[0] // 10)
    logging.info(f"Number of neighbors: {n_neighbors_params}")

    all_embedding_parameters = []
    all_embedding_labels = []
    neighbor_values = [n_neighbors_params - 2, n_neighbors_params, n_neighbors_params + 2]
    for i_n, n_n in enumerate(neighbor_values):

        embedding_params, reducer_params, scaler_params = evaluate_UMAP(data_array_scaled,
                                                                        n_neighbors=n_n,
                                                                        min_dist=0.1,
                                                                        n_components=2,
                                                                        seed=standard_seed)
        if i_n == 0:
            all_embedding_parameters.append(embedding_params)
        elif i_n == 1:
            emb1_aligned, emb2_aligned, _ = procrustes(all_embedding_parameters[-1], embedding_params)
            all_embedding_parameters[-1] = emb1_aligned
            all_embedding_parameters.append(emb2_aligned)
        elif i_n > 1:
            _, emb2_aligned, _ = procrustes(all_embedding_parameters[-1], embedding_params)
            all_embedding_parameters.append(emb2_aligned)

        all_embedding_labels.append(str(n_n))

    return all_embedding_parameters, all_embedding_labels


# ---------------------------------------------------------------------------------------------------- Complete analysis
def analylis_wrapper(logger, scaler,
                     pca=None, data_array=None,
                     variance_threshold=0.99,
                     columns_=None,
                     labels_=None,
                     type_=None,
                     case=None,
                     images_folder=None,
                     pcas_folder=None,
                     full_variance_report=False,
                     sorting_type=None,
                     save_latent_space=False,
                     make_plot_parallel_coordinates=False,
                     make_plot_distance_matrix=False,
                     make_plot_UMAP=False):
    """
    Perform the complete dimensionality-reduction analysis workflow for a
    dataset.

    Depending on the analysis type, this function optionally standardizes the
    input data, generates exploratory visualizations, computes a PCA model,
    determines the minimum number of principal components required to satisfy
    a prescribed explained-variance threshold, projects the data into the
    latent space, and performs a compact PCA analysis.

    Three types of analyses are supported:

    - ``"Parameters"``: physical or simulation parameters.
    - ``"Design variables"``: design variables describing the geometry.
    - ``"Geometry"``: latent geometric representations already obtained from a
      previously fitted PCA (no additional PCA fitting is performed).

    Parameters
    ----------
    logger : logging.Logger
        Logger used to report progress and diagnostics.

    scaler : sklearn.preprocessing.StandardScaler
        Scaler used to normalize the input data before PCA. It is ignored when
        ``type_ == "Geometry"``.

    pca : sklearn.decomposition.PCA, optional
        Previously fitted PCA model. Required only when ``type_`` is
        ``"Geometry"``.

    data_array : numpy.ndarray, optional
        Input dataset.

        - For ``"Parameters"`` and ``"Design variables"``, the array has shape
          ``(n_samples, n_features)``.
        - For ``"Geometry"``, the array is assumed to already contain the PCA
          latent coordinates.

    variance_threshold : float, optional
        Minimum cumulative explained variance required to determine the number
        of retained PCA components. Default is ``0.99``.

    columns_ : list of str, optional
        Names of the original variables, used for visualization.

    labels_ : list of str, optional
        Labels identifying each sample (e.g., experiment or mesh names). If
        omitted, labels of the form ``exp_i`` are generated automatically.

    type_ : {"Parameters", "Design variables", "Geometry"}
        Type of analysis to perform.

    case : str, optional
        Case identifier used when generating figures and output files.

    images_folder : str or pathlib.Path, optional
        Directory where generated figures are saved.

    pcas_folder : str or pathlib.Path, optional
        Directory where latent-space coordinates are stored when
        ``save_latent_space`` is enabled.

    full_variance_report : bool, optional
        If ``True``, report the explained variance contribution of every PCA
        mode. Otherwise, only the total explained variance is reported.

    sorting_type : str, optional
        Ordering strategy passed to the compact PCA analysis (for example,
        ordering of the distance matrix).

    save_latent_space : bool, optional
        If ``True``, save the computed latent coordinates to disk.

    make_plot_parallel_coordinates : bool, optional
        If ``True``, generate parallel-coordinate plots of the normalized
        dataset together with summary statistics.

    make_plot_distance_matrix : bool, optional
        If ``True``, generate the latent-space distance matrix during the
        compact PCA analysis.

    make_plot_UMAP : bool, optional
        If ``True``, compute several UMAP embeddings using different values of
        the neighborhood parameter and display the aligned embeddings.

    Returns
    -------
    pca_:
    latent_:

    Notes
    -----
    For ``"Parameters"`` and ``"Design variables"``, the workflow consists of:

    1. Standardizing the dataset.
    2. Identifying the sample closest to the centroid of the dataset.
    3. Optionally generating parallel-coordinate and UMAP visualizations.
    4. Fitting a PCA model.
    5. Automatically selecting the number of retained components based on the
       requested explained-variance threshold.
    6. Projecting the data into the PCA latent space.
    7. Performing a compact PCA analysis.

    For ``"Geometry"``, the PCA model and latent coordinates are assumed to
    have been computed previously, so only the compact PCA analysis is
    performed.
    """

    allowed_types = ['Parameters', 'Design variables', 'Geometry']
    if type_ not in allowed_types:
        raise ValueError(f"{type_} should be in {allowed_types}")

    if labels_ is not None:
        training_meshes_exp = labels_
    else:
        training_meshes_exp = [f'exp_{i}' for i in range(1, data_array.shape[0] + 1)]

    # 'Parameters', 'Design variables' are tables of data, 'Geometry' is already the n_experiment x n_features PCA
    logger.info(f"Analysis wrapper for {type_}")
    logger.info(f"{type_} as array of shape {data_array.shape}")

    if type_ != 'Geometry':
        # compute averaged
        avg_params = np.mean(data_array.astype(float), axis=0)

        distances_param_space = np.linalg.norm(data_array.astype(float) - avg_params, axis=1)
        closest_idx_param_space = np.argmin(distances_param_space)

        logger.info(f"Experiment closer to center of {type_} space (index): {closest_idx_param_space}")

        data_array_scaled = scaler.fit_transform(data_array.astype(float))
        logger.info(f"\tOriginal dimension: {data_array_scaled.shape[0]} x {data_array_scaled.shape[1]}")

        if make_plot_parallel_coordinates:
            plot_utils.plot_parallel_coordinates(data_array_scaled,
                                                 plot_title=f'Parallel coordinates, normalized {type_}',
                                                 labels_=training_meshes_exp,
                                                 highlighted_experiments_bold=[closest_idx_param_space],
                                                 component_names_=columns_, case=case, save_folder=images_folder)
            plot_utils.plot_parallel_coordinates_statistics(data_array,
                                                            plot_title=type_,
                                                            component_names_=columns_, case=case,
                                                            save_folder=images_folder)

        # UMAP
        if make_plot_UMAP:
            all_embedding_parameters, all_embedding_labels = build_multiple_UMAPs(data_array_scaled)

            plot_utils.plot_overlapped_umap(all_embedding_parameters, labels=training_meshes_exp,
                                            legend_labels=all_embedding_labels,
                                            show_quivers=True, title=f"{type_} UMAP",
                                            case=case, save_folder=images_folder)

        logger.info(f"\tEvaluating PCA on {type_}...")
        training_pca_tmp = PCA(svd_solver="full")
        training_pca_tmp.fit_transform(data_array_scaled)

        # Decide how many modes to keep
        n_components_parameters_pca = n_components_for_accuracy(logger, training_pca_tmp, threshold=variance_threshold)
        pca_ = PCA(n_components=n_components_parameters_pca, svd_solver='full')

        logger.info(f"\tComponents to achieve threshold of {variance_threshold}: {n_components_parameters_pca}")
        latent_ = pca_.fit_transform(data_array_scaled)
        logger.info(f"\t{type_} variance ratios: {pca_.explained_variance_ratio_}")
        logger.info(f"\t{type_} variance ratios (cumulative): {np.cumsum(pca_.explained_variance_ratio_)}")

    else:
        # using geometries we already have the latent space at this point
        if pca is not None:
            pca_ = pca
        else:
            raise ValueError("PCA is None")

        if data_array is not None:
            latent_ = data_array
        else:
            raise ValueError("Latent space is None")

    logger.info(f"\tLatent dimension: {latent_.shape[0]} x {latent_.shape[1]}")

    if save_latent_space:
        np.savetxt(os.path.join(pcas_folder, f'{type_}.out'), np.array(latent_))

    n_clusters, kmeans, cluster_labels, cluster_centroids = evaluate_clusters(logger, latent_)

    compact_pca_analysis(logger, pca_, latent_, case, type_, sorting_type,
                         cluster_labels,
                         variance_threshold=variance_threshold,
                         full_variance_report=full_variance_report,
                         training_meshes_exp=training_meshes_exp, images_folder=images_folder,
                         make_plot_distance_matrix=make_plot_distance_matrix)

    return pca_, latent_, (n_clusters, kmeans, cluster_labels, cluster_centroids)


def compact_pca_analysis(logger, training_pca, training_space,
                         case, type_, sorting_logic,
                         cluster_labels,
                         variance_threshold=0.9,
                         full_variance_report=False,
                         training_meshes_exp=None, images_folder=None, make_plot_distance_matrix=False):
    """
    Perform a compact analysis of a PCA latent space.

    This function summarizes the geometric structure of a PCA latent space by
    evaluating the explained variance, identifying representative and isolated
    samples, computing pairwise distances, and generating optional visualizations.

    Parameters
    ----------
    logger : logging.Logger
        Logger used to report progress and analysis results.

    training_pca : sklearn.decomposition.PCA
        Fitted PCA model corresponding to the latent space.

    training_space : numpy.ndarray of shape (n_samples, n_components)
        Coordinates of the training samples expressed in the PCA latent space.

    case : str, optional
        Case identifier used when saving generated figures.

    type_ : str
        Name of the analyzed latent space (e.g. ``"Geometry"``,
        ``"Design variables"``, or ``"Parameters"``). Used for reporting
        and figure titles.

    sorting_logic : str
        Strategy used to reorder the latent-space distance matrix for
        visualization. Passed directly to ``distance_matrix_sorting()``.

    cluster_labels : array-like of shape (n_samples,)
        Cluster assignment of each training sample. These labels are used only
        for visualization of the latent-space projection.

    variance_threshold : float, optional
        Target cumulative explained variance used to determine the minimum
        number of principal components required to reach the requested
        variance. Default is ``0.9``.

    full_variance_report : bool, optional
        If ``True``, report the explained variance contribution of every PCA
        mode. Otherwise, only the total explained variance is reported.

    training_meshes_exp : list of str, optional
        Labels associated with the training samples (typically experiment or
        geometry names). Used for reporting and plot annotations.

    images_folder : str or pathlib.Path, optional
        Directory where generated figures are saved.

    make_plot_distance_matrix : bool, optional
        If ``True``, generate and save both the original and reordered
        pairwise-distance matrices.

    Returns
    -------
    None

    Notes
    -----
    The analysis performs the following operations:

    1. Compute the explained variance and cumulative explained variance of the
       PCA model.
    2. Determine the minimum number of principal components required to reach
       the requested cumulative variance threshold.
    3. Compute the Euclidean distance matrix between all samples in the latent
       space.
    4. Symmetrize the distance matrix to eliminate numerical round-off
       asymmetries.
    5. Reorder the distance matrix according to the selected sorting strategy.
    6. Identify the most isolated samples using:
       - maximum average distance to all other samples;
       - maximin criterion (largest minimum distance).
    7. Identify the most representative samples using:
       - the medoid criterion (minimum average distance);
       - the minimax criterion (minimum maximum distance).
    8. Optionally generate the original and reordered distance-matrix plots.
    9. Generate a two-dimensional projection of the latent space using the
       first two principal components, with optional cluster colouring and
       sample annotations.

    This function is intended as a compact diagnostic tool to assess the
    distribution and coverage of a PCA latent space before further analysis or
    model training.
    """

    # pca is fitted PCA object
    explained_variance = training_pca.explained_variance_ratio_  # fraction of variance per component
    # Cumulative variance
    cumulative_variance = np.cumsum(explained_variance)

    # number of modes to achive threshold or more
    # n_representative_modes = len(cumulative_variance)
    for i_variance, variance in enumerate(cumulative_variance):
        if variance >= variance_threshold:
            logger.info(f"\tVariance threshold of {variance_threshold} reached with {i_variance + 1} modes")
            # n_representative_modes = i_variance + 1
            break

    if full_variance_report:
        logger.info("\tPCA mode contributions and Cumulative variance:")
        for i, var in enumerate(explained_variance):
            logging.info(
                f"\t\tMode {i + 1}: {var * 100:.2f}% of total variance, (Cumulative: {cumulative_variance[i] * 100:.2f})")
    else:
        logger.info(f"\tAll modes ({len(explained_variance)}); Total variance: {cumulative_variance[-1] * 100:.2f})")

    logger.info("")

    # distance matrix (n_mesh x n_mesh)
    D_training = pairwise_distances(training_space, metric='euclidean')
    # ensure perfect simmetry
    D_training = 0.5 * (D_training + D_training.T)

    D_training_sorted, labels_training_sorted = distance_matrix_sorting(sorting_type=sorting_logic,
                                                                        latent_space=training_space,
                                                                        D=D_training,
                                                                        labels=training_meshes_exp)

    logger.info(" Most isolated geometry")
    idx_largest_average, avg_dist, idx_maxmin, maxmin_dist = find_most_isolated(D_training)

    logger.info(f"\tMax average index: {idx_largest_average}, Experiment: {training_meshes_exp[idx_largest_average]}")
    logger.info(f"\tMaximin index: {idx_maxmin}, Experiment: {training_meshes_exp[idx_maxmin]}")

    logger.info(" Least isolated geometry")

    # Most representative (minimum average) and most central in the minimax sense
    medoid_idx, metoid_distance, minimax_idx, minimax_distance = find_least_isolated(D_training)

    logger.info(f"\tMedoid index: {medoid_idx}, Experiment: {training_meshes_exp[medoid_idx]}")
    logger.info(f"\tMinimax index: {minimax_idx}, Experiment: {training_meshes_exp[minimax_idx]}")

    # plot distance matrix, as is (experiment by experiment) and sorted according to selected logic
    if make_plot_distance_matrix:
        sdf_plot.plot_distance_matrix(D_training,
                                      title_=f"Distance Matrix (Latent {type_} space)",
                                      labels=training_meshes_exp, save_folder=images_folder)
        sdf_plot.plot_distance_matrix(D_training_sorted,
                                      title_=f"Sorted ({sorting_logic}) Distance Matrix (Latent {type_} space)",
                                      labels=labels_training_sorted, save_folder=images_folder)

    sdf_plot.plot_pca_projection(training_space, None,
                                 training_labels=training_meshes_exp, new_labels=None,
                                 cluster_labels=cluster_labels, new_cluster_labels=None,
                                 annotate=True,
                                 title=f"All experiments in latent {type_} space\n(PCA1+PCA2 = {cumulative_variance[1]:>4.2e})",
                                 case=case, save_folder=images_folder)


# =========================================================================================================== Clustering
def cluster_silhouette_score(minimum_number_clusters, maximum_number_clusters,
                             data, random_state=standard_seed):
    """
    Evaluate clustering quality using the silhouette score for different
    numbers of clusters.

    The function performs K-means clustering for every number of clusters in
    the interval ``[minimum_number_clusters, maximum_number_clusters)`` and
    computes the corresponding silhouette score. The resulting scores can be
    used to identify a suitable number of clusters.

    The silhouette score ranges from -1 to 1:
        - > 0.70 : very well separated clusters
        - 0.50–0.70 : good clustering
        - 0.25–0.50 : acceptable clustering
        - < 0.25 : weak or poorly separated clusters

    Parameters
    ----------
    minimum_number_clusters : int
        Minimum number of clusters to evaluate (inclusive).

    maximum_number_clusters : int
        Maximum number of clusters to evaluate (exclusive).

    data : numpy.ndarray of shape (n_samples, n_features)
        Dataset to be clustered.

    random_state : int, optional
        Random seed used to initialize the K-means algorithm.
        Defaults to ``standard_seed``.

    Returns
    -------
    numpy.ndarray
        One-dimensional array containing the silhouette score for each
        evaluated number of clusters. The first element corresponds to
        ``minimum_number_clusters``, the second to
        ``minimum_number_clusters + 1``, and so on.

    Notes
    -----
    The silhouette score measures how similar each sample is to its own
    cluster compared to the nearest neighboring cluster. Larger values
    indicate more compact and better-separated clusters.
    """

    scores = []
    for k in range(minimum_number_clusters, maximum_number_clusters):
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        labels = kmeans.fit_predict(data)

        score = silhouette_score(data, labels)
        scores.append(score)
    return np.asarray(scores)


def evaluate_clusters(logger, latent_space, minimum_number_clusters=2, random_state=standard_seed):
    """
    Determine the optimal number of clusters in the latent space and perform
    K-Means clustering.

    The function first estimates the appropriate number of clusters using
    ``evaluate_number_of_clusters`` and then fits a K-Means model to the
    latent representations.

    Parameters
    ----------
    logger : logging.Logger
        Logger used to report clustering information and diagnostics.

    latent_space : numpy.ndarray
        Two-dimensional array of latent representations with shape
        ``(n_samples, n_features)``. Each row corresponds to one geometry
        or experiment encoded in the latent space.

    minimum_number_clusters : int, optional
        Minimum number of clusters considered during the cluster-number
        evaluation. Default is 2.

    random_state : int, optional
        Random seed used to ensure reproducible cluster evaluation and
        K-Means initialization. Default is ``standard_seed``.

    Returns
    -------
    n_clusters : int
        Estimated optimal number of clusters.

    kmeans : sklearn.cluster.KMeans
        Fitted K-Means model.

    cluster_labels : numpy.ndarray
        Cluster assignment for each sample in ``latent_space``.
        Shape ``(n_samples,)``.

    cluster_centroids : numpy.ndarray
        Coordinates of the cluster centroids in the latent space.
        Shape ``(n_clusters, n_features)``.
    """

    # clustering of training data
    n_clusters = evaluate_number_of_clusters(logger, latent_space,
                                             minimum_number_clusters=minimum_number_clusters,
                                             random_state=random_state)
    # cluster the training data
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    cluster_labels = kmeans.fit_predict(latent_space)
    cluster_centroids = kmeans.cluster_centers_

    return n_clusters, kmeans, cluster_labels, cluster_centroids


def evaluate_number_of_clusters(logger, training_data, minimum_number_clusters=2, random_state=standard_seed):
    """
    Uses silhouette score to determine 'optimal' number of clusters.

    Parameters
    ----------
    logger : logger object
    training_data : (n_experiments, n_features) ndarray
        2d array, with geometries projection in latent space of normalize parameters.
    minimum_number_clusters : int, minimum number of clusters, default 2, will loop till min(15, half number of experiments)
        Labels for training points.
    random_state : int, optional
        standard seed to be used for consistent random number generation.

    Returns
    -------
        n_clusters, index associated to the highest silhouette score + minimum_number_clusters
    """

    maximum_number_clusters = min(15, training_data.shape[0])
    silhouette_scores = cluster_silhouette_score(minimum_number_clusters, maximum_number_clusters,
                                                 training_data, random_state=random_state)
    silhouette_scores_print = [round(float(f), 4) for f in silhouette_scores]

    logger.info(f"Clustering, silhouette scores: {silhouette_scores_print}")
    n_clusters = np.argmax(silhouette_scores) + minimum_number_clusters
    silhouette_max_print = round(float(np.max(silhouette_scores)), 4)
    logger.info(" > 0.7 : very well separated")
    logger.info(" 0.5–0.7 : good")
    logger.info(" 0.25–0.5 : acceptable")
    logger.info(" < 0.25 : clustering is weak")
    logger.info(f"Selected number of clusters: {n_clusters} (index {np.argmax(silhouette_scores)}) "
                f"value: {silhouette_max_print}")
    if np.max(silhouette_scores) < 0.25:
        logger.warning("----------------------------------------------------------------------------------------------")
        logger.warning(f"Clustering at {silhouette_max_print} is weak")
        logger.warning("----------------------------------------------------------------------------------------------")

    return n_clusters


def evaluate_cluster_metrics(logger, n_clusters, kmeans, cluster_labels, cluster_centroids,
                             new_experiment, training_experiments):
    """
    Evaluate cluster-based metrics for a new experiment.

    The function assigns the new experiment to its nearest k-means cluster and
    computes a set of geometric metrics describing its position relative to the
    training data in that cluster. These metrics include the Euclidean distance to
    the assigned cluster centroid, the corresponding percentile within the training
    cluster, the ratio between the centroid distance and the cluster radius, the
    Mahalanobis distance, and its percentile. Distances to all cluster centroids
    are also returned.

    Parameters
    ----------
    logger : logging.Logger
        Logger instance used to report the computed metrics.
    n_clusters : int
        Number of clusters in the fitted k-means model.
    kmeans : sklearn.cluster.KMeans
        Fitted k-means model.
    cluster_labels : np.ndarray
        Cluster assignment of each training experiment.
    cluster_centroids : np.ndarray
        Coordinates of the cluster centroids in latent space, with shape
        ``(n_clusters, n_features)``.
    new_experiment : np.ndarray
        Latent representation of the experiment to evaluate, with shape
        ``(1, n_features)`` or ``(n_features,)``.
    training_experiments : np.ndarray
        Latent representations of the training experiments, with shape
        ``(n_training, n_features)``.

    Returns
    -------
    tuple
        Tuple containing:

        - **cluster_id** (*int*): Assigned cluster.
        - **cluster_size** (*int*): Number of training experiments in the assigned cluster.
        - **new_centroid_distance** (*float*): Euclidean distance to the assigned cluster centroid.
        - **cluster_centroid_distance_percentile** (*float*): Percentile of the centroid distance with respect to the training members of the assigned cluster.
        - **radius_ratio** (*float*): Ratio between the centroid distance and the cluster radius.
        - **cluster_mahalanobis_distance** (*float*): Mahalanobis distance from the assigned cluster.
        - **mahal_percentile** (*float*): Percentile of the Mahalanobis distance within the assigned cluster.
        - **cluster_distances** (*np.ndarray*): Euclidean distances from the experiment to all cluster centroids.
    """

    cluster_id = kmeans.predict(new_experiment).item()
    logger.info(f"\tNearest cluster: {cluster_id}")

    cluster_size = np.sum(cluster_labels == cluster_id)
    logger.info(f"\tCluster population: {cluster_size}")

    cluster_distances = np.linalg.norm(cluster_centroids - new_experiment, axis=1)

    # distance from the cluster centroid
    cluster_centroid = cluster_centroids[cluster_id]
    new_centroid_distance = np.linalg.norm(new_experiment - cluster_centroid)
    logger.info(f"\tDistance to cluster centroid: {new_centroid_distance:>4.2e}")

    # loop on all other clusters
    for i_cluster_id in range(n_clusters):
        if i_cluster_id != cluster_id:
            logger.info(f"\t\tDistance to cluster {i_cluster_id} centroid: {cluster_distances[i_cluster_id]:>4.2e}")

    # centroid distance percentile
    cluster_members = np.where(cluster_labels == cluster_id)[0]
    training_centroid_distances = np.linalg.norm(training_experiments[cluster_members] - cluster_centroid, axis=1)

    cluster_centroid_distance_percentile = percentileofscore(training_centroid_distances,
                                                             new_centroid_distance, kind="weak")
    # 5%   -> very central
    # 50%  -> typical
    # 95%  -> among the most peripheral points
    # 100% -> further than every training experiment in this cluster
    logger.info(f"\tCentroid distance percentile: {cluster_centroid_distance_percentile}")

    # < 0.5 -> inside the cluster
    # about 1 -> close to the boundary
    # > 1 -> outside the envelope of the training members in that cluster
    cluster_radius = np.max(training_centroid_distances)
    radius_ratio = new_centroid_distance / cluster_radius
    logger.info(f"\tRatio distance to cluster radius: {radius_ratio:>4.2e}")

    # Mahalanobis distance
    cluster_points = training_experiments[cluster_members]
    cov = np.cov(cluster_points.T)
    inv_cov = np.linalg.pinv(cov)

    delta = new_experiment.ravel() - cluster_centroid.ravel()
    cluster_mahalanobis_distance = np.sqrt(delta @ inv_cov @ delta)
    logger.info(f"\tMahalanobis distance: {cluster_mahalanobis_distance:>4.2e}")

    # Mahalanobis percentile
    training_mahal = []
    for x in cluster_members:
        d = x - cluster_centroid
        training_mahal.append(np.sqrt(d @ inv_cov @ d))
    training_mahal = np.asarray(training_mahal)
    # delta = cluster_points - cluster_centroid
    # training_mahal = np.sqrt(np.sum((delta @ inv_cov) * delta, axis=1))

    mahal_percentile = percentileofscore(training_mahal, cluster_mahalanobis_distance, kind="weak")
    logger.info(f"\tMahalanobis distance percentile: {mahal_percentile:>4.2e}")

    return cluster_id, int(cluster_size), float(new_centroid_distance), float(cluster_centroid_distance_percentile), \
           float(radius_ratio), float(cluster_mahalanobis_distance), float(mahal_percentile), cluster_distances
