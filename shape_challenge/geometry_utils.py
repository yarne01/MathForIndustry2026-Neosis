import logging
import os
import copy
import open3d as o3d
import numpy as np
import igl
import trimesh
from pathlib import Path

import generic_utils as gen_utils


# ---------------------------------------------------------------------------------------------------- Files and Folders
def load_mesh(path):
    """
    Uses Open3D to load a triangular mesh.

    Parameters
    ----------
    path: str
        Path and name and extension of the file to be loaded.

    Returns
    -------
    vertexes: np.array(nVertex, 3)
        Vertexes in the mesh.
    triangles: np.array(nTriangle, 3)
        Elements in the mesh.
    """

    mesh = o3d.io.read_triangle_mesh(path)
    mesh.compute_vertex_normals()
    return np.asarray(mesh.vertices), np.asarray(mesh.triangles, dtype=np.int32)


def all_folders_contain_stl(folders):
    """
    Check whether every folder in a collection contains at least one STL file.

    Parameters
    ----------
    folders : iterable of str or pathlib.Path
        Collection of folder paths to inspect.

    Returns
    -------
    ok : bool
        True if every folder contains at least one file with the ``.stl``
        extension, False otherwise.

    missing_folder : pathlib.Path or None
        Path of the first folder that does not contain any STL file.
        Returns None if all folders contain at least one STL file.

    Notes
    -----
    - Only files located directly inside each folder are considered.
      To search recursively through subdirectories, replace
      ``Path(folder).glob("*.stl")`` with ``Path(folder).rglob("*.stl")``.
    - The search stops as soon as a folder without STL files is found.
    """

    for folder in folders:
        if not any(Path(folder).glob("*.stl")):
            return False, folder
    return True, None


def load_geometry(logger_, dataset_json_, full_path_, folder_, type_="Source",
                  suppress_all=False):
    """
    Load a given gemetry mesh from file. If possible, will read the json and get the information,
    otherwise will go for a more direct and hardcoded approach.

    Parameters
    ----------
    logger_: logging.Logger
        Logger used for diagnostic output.
    dataset_json_: str
        Path and name till database json (nvision dataset one).
    full_path_: str
        Full path, including directory subfolders.
    folder_: str
        Full path, including directory subfolders.
    name of the folder: str
        Name of the folder where mesh is locates, e.g. exp_6.
    type_: str
        Source or Target to be printed in logs for clarity.
    suppress_all: bool
        Suppress all log messages.

    Returns
    -------
    V_: np.array(nVertex,3)
        Coordinates of the vertexes.
    F_: np.array(nElem,3)
        ID of the vertexes in every element.
    list_V_surface_: np.array(nVertex_surface,3)
        Coordinates of the vertexes on the surface of the mesh.
    list_F_surface_: np.array(nElem_surface,3)
        ID of the vertexes in every element on the surface of the mesh.
    """

    if os.path.exists(dataset_json_):

        jsonobj_database_ = gen_utils.read_json(dataset_json_)
        geometry_name_ = jsonobj_database_["variables"]["inputs"]["ShellMesh"]["basename"]
        geometry_ext_ = jsonobj_database_["variables"]["inputs"]["ShellMesh"]["extension"]

        path_geometry_ = os.path.join(full_path_, folder_)

        geometry_ = f"{geometry_name_}{geometry_ext_}"

        logger_.debug(f"{type_} geometry: {geometry_}")

        # mesh as loaded
        V_, F_ = load_mesh(os.path.join(path_geometry_, geometry_))

        F_, F_canonical_ = unify_duplicate_vertices(logger_, f"F_{type_}", V_, F_, suppress_all=suppress_all)

        V_original_ = V_.copy()
        F_original_ = F_.copy()
        V_, F_, _, IM_src_ = igl.remove_unreferenced(V_original_, F_original_)

        list_V_surface_ = V_
        list_F_surface_ = F_

    else:
        V_ = []
        F_ = []
        list_V_surface_ = []
        list_F_surface_ = []
        logger_.error(f'Error while loading: {full_path_}')

    logger_.debug(f"\tF_{type_}.shape: {F_.shape}")
    logger_.debug(f"\tlist_F_surface_{type_}.shape: {list_F_surface_.shape}")
    logger_.debug(f"\tV_{type_}.shape: {V_.shape}")
    logger_.debug(f"\tlist_V_surface_{type_}.shape: {list_V_surface_.shape}")

    count_degenerate(F_, logger_, suppress_all, "Volume")
    orphan_vertexes(F_, V_, logger_, suppress_all, "Volume")
    count_degenerate(list_F_surface_, logger_, suppress_all, "Surface")
    orphan_vertexes(list_F_surface_, list_V_surface_, logger_, suppress_all, "Surface")

    return V_, F_, list_V_surface_, list_F_surface_


def unify_duplicate_vertices(logger, varname, V, F, suppress_all=False, eps=1e-12):
    """
    Detect vertices with identical coordinates (within eps),
    and remap the face indices so that all duplicates refer to the
    first occurrence. DOES NOT remove vertices from V.

    V : (N,3) float array
    F : (M,3) int array
    suppress_all: bool, removes all log messages

    Returns:
        F_new : faces remapped to canonical vertex indices
        map_to_canonical : length-N array with the canonical ID for each vertex
    """

    # quantize for floating point safety
    Vq = np.round(V / eps).astype(np.int64)

    # canonical index lookup: first time a coordinate appears
    unique_map = {}
    canonical = np.arange(len(V), dtype=np.int32)

    for i, key in enumerate(map(tuple, Vq)):
        if key not in unique_map:
            unique_map[key] = i  # first occurrence = canonical
        canonical[i] = unique_map[key]

    # remap faces
    F_new = canonical[F]

    if len(unique_map) != len(V) and not suppress_all:
        logger.warning(f"\t{varname}: {len(unique_map)} unique vertices among {len(V)}, {len(V) - len(unique_map)} duplicated nodes")

    return F_new, canonical


def count_degenerate(F, logger, suppress_all, msg=""):
    same01 = (F[:, 0] == F[:, 1])
    same02 = (F[:, 0] == F[:, 2])
    same12 = (F[:, 1] == F[:, 2])
    deg = same01 | same02 | same12
    if deg.sum() > 0 and not suppress_all:
        logger.warning(f"{msg}, degenerate triangles: {deg.sum()} / {len(F)}")


def orphan_vertexes(F, V, logger, suppress_all, msg=""):
    """
    :param F: shape (nTr, 3) → indices of vertices forming each triangle
    :param V: shape (nNodes, 3) → vertex coordinates
    :param logger: logger entity
    :param suppress_all: bool, if true all logs are suppressed
    :param msg: message, volume of surface
    :return:
    """
    used_vertices = np.unique(F)
    all_vertices = np.arange(V.shape[0])
    unused_vertices = np.setdiff1d(all_vertices, used_vertices)

    if unused_vertices.size > 0 and not suppress_all:
        logger.warning(f"{msg}, N. unused vertices: {len(unused_vertices)}")
        # logger.warning(f"{msg}, Unused vertices: {unused_vertices}")
    return unused_vertices


def prepare_o3d_mesh(logger, vertices_, faces_, colour=[1.0, 1.0, 1.0], name="Source"):

    mesh_ = o3d.geometry.TriangleMesh()
    mesh_.vertices = o3d.utility.Vector3dVector(vertices_)
    mesh_.triangles = o3d.utility.Vector3iVector(faces_)
    col_ = copy.deepcopy(mesh_).paint_uniform_color(colour)
    mesh_aabb_ = get_mesh_bounding_box(mesh_, logger, text=name)

    return mesh_, col_, mesh_aabb_


def center_mesh_on_origin(mesh_):
    """
    Translate a mesh so that the center of its axis-aligned bounding box
    is located at (0, 0, 0).

    Parameters
    ----------
    mesh_ : o3d.geometry.TriangleMesh

    Returns
    -------
    mesh_centered : o3d.geometry.TriangleMesh
        Centered copy of the input mesh.

    translation : np.ndarray
        Translation vector applied.
    """

    mesh_centered = copy.deepcopy(mesh_)

    aabb = mesh_centered.get_axis_aligned_bounding_box()

    center = 0.5 * (aabb.min_bound + aabb.max_bound)

    mesh_centered.translate(-center)

    return mesh_centered, -center


# converts quad elements into tri elements
def quads_to_tris(quads):
    idx_f = [[0, 1, 2],
             [2, 3, 0]]
    tris = np.empty(shape=(len(quads) * 2, 3), dtype=int)
    tris = quads[:, idx_f]
    tris = np.reshape(tris, (len(quads) * 2, 3))

    return tris


def get_mesh_bounding_box(mesh_, logger_, text=""):

    aabb = mesh_.get_axis_aligned_bounding_box()
    logger_.debug(f"\t{text} AABB min corner: {aabb.get_min_bound()}")
    logger_.debug(f"\t{text} AABB max corner: {aabb.get_max_bound()}")
    logger_.debug(f"\t{text} AABB center: {aabb.get_center()}")


    return aabb


def load_mesh_wrapper(logger, dataset_json_file, full_path, folder_exp, type_="Source",
                      suppress_all=False):
    """
    Wrapper to cover load mesh functionality and conversion to o3d object
    :param logger: logger entity
    :param dataset_json_file: dict loaded with information of mesh name and extension
    :param full_path: path till root of all exp_ folder
    :param folder_exp: current exp_ to be loaded
    :param type_: str, legacy, only used in log messages
    :param suppress_all: bool, suppress all log messages
    :return:
    """
    V_, F_, list_V_surface_, list_F_surface_ = load_geometry(logger, dataset_json_file,
                                                             full_path, folder_exp,
                                                             type_=type_,
                                                             suppress_all=suppress_all)

    # reduce meshes to external surfaces
    F_work = list_F_surface_
    mesh_, col_, mesh_aabb_ = prepare_o3d_mesh(logger, V_, F_work, colour=[1.0, 0.7, 0.1], name=type_)

    if not suppress_all:
        tm = trimesh.Trimesh(vertices=np.asarray(mesh_.vertices), faces=np.asarray(mesh_.triangles))
        logger.info(f"\tWatertight: {tm.is_watertight}")
        logger.info(f"\tWinding consistent: {tm.is_winding_consistent}")
        logger.info(f"\tVolume: {tm.volume}")
        bbox = tm.bounds

        size = bbox[1] - bbox[0]

        logger.info(f"\tsize: {size}")
        logger.info(f"\tbbox volume: {np.prod(size)}, mesh volume: {tm.volume}")

    mesh_, shift = center_mesh_on_origin(mesh_)

    mesh_aabb_ = mesh_.get_axis_aligned_bounding_box()

    return mesh_, col_, mesh_aabb_
