from fractions import Fraction
import json
import os

import numpy as np
import scipy.sparse
import scipy.sparse.csgraph
from scipy.spatial import cKDTree
import igl
import open3d as o3d

from repair_utils import self_intersecting_faces, repair_mesh


# ---------------------------------------------------------------------------------------------------- Helpers
def _validate_mesh(vertices, faces):
    vertices, faces = np.asarray(vertices), np.asarray(faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices) or vertices.dtype.kind not in 'fiu':
        raise ValueError('vertices must be a nonempty real array of shape (n, 3)')
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces) or faces.dtype.kind not in 'iu':
        raise ValueError('faces must be a nonempty integer array of shape (m, 3)')
    if not np.isfinite(vertices).all() or np.any(faces < 0) or np.any(faces >= len(vertices)):
        raise ValueError('vertices must be finite and face indices must be in range')
    return np.ascontiguousarray(vertices, dtype=float), np.ascontiguousarray(faces, dtype=np.int64)


def _face_geometry(vertices, faces):
    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    twice_area = np.linalg.norm(cross, axis=1)
    normals = np.zeros_like(cross)
    np.divide(cross, twice_area[:, None], out=normals, where=twice_area[:, None] > 0)
    return twice_area / 2, normals


def _count_collinear_faces(vertices, faces):
    triangles = vertices[faces]
    # The filter only certifies nonzero minors; uncertain arithmetic is checked exactly.
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        u = triangles[:, 1] - triangles[:, 0]
        v = triangles[:, 2] - triangles[:, 0]
        left = u[:, [1, 2, 0]] * v[:, [2, 0, 1]]
        right = u[:, [2, 0, 1]] * v[:, [1, 2, 0]]
        cross = left - right
        error = 16 * np.finfo(float).eps * (np.abs(left) + np.abs(right)) + 16 * np.finfo(float).tiny
    uncertain = (np.abs(cross) <= error) | ~np.isfinite(cross) | ~np.isfinite(error)
    count = 0
    for triangle in triangles[np.all(uncertain, axis=1)]:
        a, b, c = [[Fraction(float(value)) for value in corner] for corner in triangle]
        # Subtract original dyadic coordinates, not already-rounded float edge vectors.
        u = [b[i] - a[i] for i in range(3)]
        v = [c[i] - a[i] for i in range(3)]
        count += all(u[i] * v[j] == u[j] * v[i] for i, j in ((0, 1), (1, 2), (2, 0)))
    return count


# ---------------------------------------------------------------------------------------------------- Topology
def _signed_volume(vertices, faces):
    triangles = vertices[faces] - vertices[faces[0, 0]]
    return float(np.einsum('ij,ij->i', triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])).sum() / 6)


def _edge_data(faces):
    directed = np.stack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=1).reshape(-1, 2)
    edges, first, inverse, counts = np.unique(
        np.sort(directed, axis=1), axis=0, return_index=True, return_inverse=True, return_counts=True)
    return directed, edges, first, inverse, counts


def _connected_components(pairs, size):
    graph = scipy.sparse.coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(size, size))
    return scipy.sparse.csgraph.connected_components(graph.tocsr(), directed=False)


def _non_manifold_vertices(faces, n_vertices):
    # libigl checks for a single incident face fan; unused vertices are reported separately.
    used_vertices = np.unique(faces)
    non_manifold = np.zeros(n_vertices, dtype=bool)
    non_manifold[used_vertices] = ~igl.is_vertex_manifold(faces)[used_vertices]
    # A connected fan can still branch at an edge or contain collapsed/duplicate triangles.
    _, edges, _, _, counts = _edge_data(faces)
    non_manifold[np.unique(edges[(counts > 2) | (edges[:, 0] == edges[:, 1])])] = True
    unique_faces, counts = np.unique(np.sort(faces, axis=1), axis=0, return_counts=True)
    non_manifold[np.unique(unique_faces[counts > 1])] = True
    return non_manifold


def _is_orientable(faces):
    # A BFS assigns face flips; a contradictory cycle leaves a shared edge with equal winding.
    oriented_faces, _ = igl.bfs_orient(faces)
    directed, _, _, edge_ids, counts = _edge_data(oriented_faces)
    winding = np.bincount(edge_ids, weights=np.where(directed[:, 0] < directed[:, 1], 1, -1))
    return bool(np.all(winding[counts == 2] == 0))


def mesh_quality(vertices, faces, check_intersections=True):
    """Measure indexed topology, area and signed closed volume.

    Parameters
    ----------
    vertices, faces : (n, 3), (m, 3) ndarray
    check_intersections : bool

    Returns
    -------
    quality : dict
        Euler excludes unused vertices; genus sums closed orientable components.
        Boundary loops are simple cycles. zero_area_faces counts float area == 0;
        collinear_faces uses exact input-coordinate collinearity. degenerate_faces
        is an alias of collinear_faces, not a small-area tolerance check.
    """
    vertices, faces = _validate_mesh(vertices, faces)
    area, _ = _face_geometry(vertices, faces)
    directed, edges, first, inverse, counts = _edge_data(faces)
    owners = np.repeat(np.arange(len(faces)), 3)
    components, _ = _connected_components(np.column_stack((owners, owners[first[inverse]])), len(faces))
    winding = np.bincount(inverse, weights=np.where(directed[:, 0] < directed[:, 1], 1, -1))
    consistent = bool(np.all(counts <= 2) and np.all(winding[counts == 2] == 0)
                      and np.all(edges[:, 0] != edges[:, 1]))
    boundary = edges[counts == 1]
    loops = 0
    if len(boundary):
        used, remap = np.unique(boundary, return_inverse=True)
        number, labels = _connected_components(remap.reshape(-1, 2), len(used))
        degrees = np.bincount(remap.reshape(-1), minlength=len(used))
        invalid = np.bincount(labels[degrees != 2], minlength=number)
        sizes = np.bincount(labels, minlength=number)
        loops = int(np.sum((invalid == 0) & (sizes >= 3)))
    non_manifold = _non_manifold_vertices(faces, len(vertices))
    used = len(np.unique(faces))
    euler_characteristic = int(used - len(edges) + len(faces))
    zero_area = int(np.sum(area == 0))
    collinear = _count_collinear_faces(vertices, faces)
    duplicates = len(faces) - len(np.unique(np.sort(faces, axis=1), axis=0))
    watertight = bool(np.all(counts == 2))
    closed_manifold = watertight and not non_manifold.any() and (consistent or _is_orientable(faces))
    genus = (2 * components - euler_characteristic) // 2 if closed_manifold else None
    intersections = int(len(self_intersecting_faces(vertices, faces))) if check_intersections else None
    return dict(n_vertices=len(vertices), n_faces=len(faces), unused_vertices=len(vertices) - used,
                zero_area_faces=zero_area, collinear_faces=collinear, degenerate_faces=collinear,
                duplicate_faces=int(duplicates), components=int(components),
                boundary_edges=int(np.sum(counts == 1)), boundary_loops=loops,
                non_manifold_edges=int(np.sum(counts > 2)), non_manifold_vertices=int(non_manifold.sum()),
                consistently_oriented=consistent, watertight=watertight, euler_characteristic=euler_characteristic,
                genus=genus, area=float(area.sum()), volume=_signed_volume(vertices, faces) if watertight else None,
                self_intersecting_faces=intersections)


# ---------------------------------------------------------------------------------------------------- Surface distances
def _point_surface_distances(points, vertices, faces):
    squared, ids, _ = igl.point_mesh_squared_distance(points, vertices, faces)
    return np.sqrt(squared), ids


def _distance_vector(values, name):
    array = np.asarray(values)
    if array.ndim != 1 or not len(array) or array.dtype.kind not in 'fiu':
        raise ValueError(name + ' must be a nonempty real vector')
    if not np.isfinite(array).all() or np.any(array < 0):
        raise ValueError(name + ' must be finite and nonnegative')
    return array.astype(float)


def sample_surface(vertices, faces, n_points, seed=42):
    """Sample uniformly by area using a local random generator.

    Parameters
    ----------
    vertices, faces : (n, 3), (m, 3) ndarray
    n_points, seed : int

    Returns
    -------
    points, normals : (n_points, 3) ndarray
    """
    vertices, faces = _validate_mesh(vertices, faces)
    if n_points <= 0:
        raise ValueError('n_points must be positive')
    area, normals = _face_geometry(vertices, faces)
    if area.sum() <= 0:
        raise ValueError('surface sampling requires positive total area')
    rng = np.random.default_rng(seed)
    ids = rng.choice(len(faces), size=n_points, p=area / area.sum())
    uv = rng.random((n_points, 2))
    root = np.sqrt(uv[:, 0])
    barycentric = np.column_stack((1 - root, root * (1 - uv[:, 1]), root * uv[:, 1]))
    points = np.einsum('ni,nij->nj', barycentric, vertices[faces[ids]])
    return points, normals[ids]


def compare_surfaces(reference_vertices, reference_faces, vertices, faces, n_points=20000, thresholds=None, seed=42):
    """Measure unsquared bidirectional point-to-triangle distances without alignment.

    Parameters
    ----------
    reference_vertices, reference_faces, vertices, faces : ndarray
    n_points, seed : int
    thresholds : array_like or None
        Defaults to reference bbox fractions [.001, .005, .01, .02].

    Returns
    -------
    comparison : dict
        Distances, sampled Hausdorff estimate, orientation-insensitive normal angles and F-scores.
    """
    reference_vertices, reference_faces = _validate_mesh(reference_vertices, reference_faces)
    vertices, faces = _validate_mesh(vertices, faces)
    reference_points, reference_normals = sample_surface(reference_vertices, reference_faces, n_points, int(seed) + 1)
    points, normals = sample_surface(vertices, faces, n_points, seed)
    reference_distances, candidate_ids = _point_surface_distances(reference_points, vertices, faces)
    candidate_distances, reference_ids = _point_surface_distances(points, reference_vertices, reference_faces)
    reference_area, reference_face_normals = _face_geometry(reference_vertices, reference_faces)
    area, face_normals = _face_geometry(vertices, faces)
    if np.any(reference_area[reference_ids] == 0) or np.any(area[candidate_ids] == 0):
        raise ValueError('closest-face normals are undefined on degenerate triangles')
    dots = np.concatenate((np.einsum('ij,ij->i', reference_normals, face_normals[candidate_ids]),
                           np.einsum('ij,ij->i', normals, reference_face_normals[reference_ids])))
    angles = np.degrees(np.arccos(np.clip(np.abs(dots), 0, 1)))
    if thresholds is None:
        thresholds = np.array([.001, .005, .01, .02]) * np.linalg.norm(np.ptp(reference_vertices, axis=0))
    return dict(chamfer_mean=float((reference_distances.mean() + candidate_distances.mean()) / 2),
                reference_to_mesh_mean=float(reference_distances.mean()),
                mesh_to_reference_mean=float(candidate_distances.mean()),
                hausdorff=float(max(reference_distances.max(), candidate_distances.max())),
                reference_to_mesh_p95=float(np.percentile(reference_distances, 95)),
                mesh_to_reference_p95=float(np.percentile(candidate_distances, 95)),
                normal_angle_mean_deg=float(angles.mean()),
                fscore=fscore_curve(candidate_distances, reference_distances, thresholds))


def fscore_curve(distances_a, distances_b, thresholds):
    """Compute precision, recall and F-score using strict distance < threshold.

    Parameters
    ----------
    distances_a, distances_b : array_like
        Candidate-to-reference and reference-to-candidate nonnegative distances.
    thresholds : array_like

    Returns
    -------
    curve : dict
    """
    candidate_distances = np.sort(_distance_vector(distances_a, 'distances_a'))
    reference_distances = np.sort(_distance_vector(distances_b, 'distances_b'))
    levels = _distance_vector(thresholds, 'thresholds')
    precision = np.searchsorted(candidate_distances, levels, side='left') / len(candidate_distances)
    recall = np.searchsorted(reference_distances, levels, side='left') / len(reference_distances)
    score = np.zeros(len(levels))
    np.divide(2 * precision * recall, precision + recall, out=score, where=precision + recall > 0)
    return dict(thresholds=levels, precision=precision, recall=recall, fscore=score)


# ---------------------------------------------------------------------------------------------------- Shape descriptors
def mesh_descriptors(vertices, faces):
    """Measure size, surface centroid/inertia and sharp edges (> 30 degrees).

    Parameters
    ----------
    vertices, faces : (n, 3), (m, 3) ndarray

    Returns
    -------
    descriptors : dict
        Unit-area-density inertia eigenvalues ascend; principal axes are rows.
    """
    vertices, faces = _validate_mesh(vertices, faces)
    area, normals = _face_geometry(vertices, faces)
    total_area = float(area.sum())
    if total_area <= 0:
        raise ValueError('descriptors require positive total area')
    triangles = vertices[faces] - vertices[faces[0, 0]]
    local_centroid = np.einsum('i,ij->j', area, triangles.mean(axis=1)) / total_area
    centered = triangles - local_centroid
    sums = centered.sum(axis=1)
    # Exact triangle second moments avoid vertex-density bias in the principal axes.
    moment = (np.einsum('i,ij,ik->jk', area, sums, sums)
              + np.einsum('i,ilj,ilk->jk', area, centered, centered)) / 12
    inertia = np.eye(3) * np.trace(moment) - moment
    eigenvalues, eigenvectors = np.linalg.eigh(inertia)
    _, _, _, inverse, counts = _edge_data(faces)
    order = np.argsort(inverse)
    starts = np.cumsum(counts) - counts
    paired = starts[counts == 2]
    left, right = order[paired] // 3, order[paired + 1] // 3
    valid = (area[left] > 0) & (area[right] > 0)
    dots = np.einsum('ij,ij->i', normals[left], normals[right])
    sharp = np.sum(valid & (dots < np.cos(np.deg2rad(30))))
    volume = _signed_volume(vertices, faces) if np.all(counts == 2) else None
    compactness = None if volume is None else float(36 * np.pi * volume ** 2 / total_area ** 3)
    return dict(area=total_area, volume=volume, bbox_min=vertices.min(axis=0), bbox_max=vertices.max(axis=0),
                bbox_extents=np.ptp(vertices, axis=0), centroid=local_centroid + vertices[faces[0, 0]],
                principal_axes=eigenvectors.T.copy(), inertia_eigenvalues=eigenvalues,
                compactness=compactness, sharp_edge_count=int(sharp))


def curvature(vertices, faces):
    """Measure absolute H/K per vertex area, excluding boundary/non-manifold/unused vertices.

    Parameters
    ----------
    vertices, faces : (n, 3), (m, 3) ndarray

    Returns
    -------
    result : dict
        Area-weighted means, percentiles (50, 90, 95, 99) and evaluated vertex count.
    """
    vertices, faces = _validate_mesh(vertices, faces)
    area, _ = _face_geometry(vertices, faces)
    if np.any(area <= 0):
        raise ValueError('curvature requires nondegenerate triangles')
    _, edges, _, _, counts = _edge_data(faces)
    valid = ~_non_manifold_vertices(faces, len(vertices))
    valid[np.unique(edges[counts != 2])] = False
    mass = igl.massmatrix(vertices, faces, igl.MASSMATRIX_TYPE_BARYCENTRIC).diagonal()
    valid &= mass > 0
    if not valid.any():
        raise ValueError('curvature requires at least one interior manifold vertex')
    laplacian = igl.cotmatrix(vertices, faces) @ (vertices - vertices[faces[0, 0]])
    mean_curvature = np.linalg.norm(laplacian[valid], axis=1) / (2 * mass[valid])
    gaussian_curvature = np.abs(igl.gaussian_curvature(vertices, faces)[valid] / mass[valid])
    levels = np.array([50, 90, 95, 99])
    return dict(mean_abs_H=float(np.average(mean_curvature, weights=mass[valid])),
                mean_abs_K=float(np.average(gaussian_curvature, weights=mass[valid])),
                percentiles=levels, abs_H_percentiles=np.percentile(mean_curvature, levels),
                abs_K_percentiles=np.percentile(gaussian_curvature, levels), n_vertices_evaluated=int(valid.sum()))


def evaluate_symmetry(vertices, faces, plane_normal=(0., 1., 0.), plane_point=(0., 0., 0.), n_points=10000, seed=42):
    """Measure reflected sample distances, also normalized by bbox diagonal.

    Parameters
    ----------
    vertices, faces : (n, 3), (m, 3) ndarray
    plane_normal, plane_point : (3,) array_like
    n_points, seed : int

    Returns
    -------
    symmetry : dict
    """
    vertices, faces = _validate_mesh(vertices, faces)
    normal, point = np.asarray(plane_normal, dtype=float), np.asarray(plane_point, dtype=float)
    if normal.shape != (3,) or point.shape != (3,) or not np.isfinite([normal, point]).all():
        raise ValueError('plane_normal and plane_point must be finite vectors of shape (3,)')
    length = np.linalg.norm(normal)
    if length == 0:
        raise ValueError('plane_normal must have nonzero length')
    normal = normal / length
    points, _ = sample_surface(vertices, faces, n_points, seed)
    reflected = points - 2 * ((points - point) @ normal)[:, None] * normal
    distances, _ = _point_surface_distances(reflected, vertices, faces)
    diagonal = np.linalg.norm(np.ptp(vertices, axis=0))
    mean, maximum, p95 = float(distances.mean()), float(distances.max()), float(np.percentile(distances, 95))
    return dict(mean=mean, max=maximum, p95=p95, mean_normalized=float(mean / diagonal),
                max_normalized=float(maximum / diagonal), p95_normalized=float(p95 / diagonal))


# ---------------------------------------------------------------------------------------------------- Diversity
def mesh_distance_matrix(meshes, n_points=1024, seed=42):
    """Compute symmetric unsquared sampled Chamfer means, using the same seed per mesh.

    Parameters
    ----------
    meshes : list of (vertices, faces) pairs
    n_points, seed : int

    Returns
    -------
    distances : (n, n) ndarray
    """
    clouds = [sample_surface(vertices, faces, n_points, seed)[0] for vertices, faces in meshes]
    trees = [cKDTree(points) for points in clouds]
    distances = np.zeros((len(clouds), len(clouds)))
    for i in range(len(clouds)):
        for j in range(i + 1, len(clouds)):
            forward_distances = trees[j].query(clouds[i])[0]
            reverse_distances = trees[i].query(clouds[j])[0]
            distances[i, j] = distances[j, i] = (forward_distances.mean() + reverse_distances.mean()) / 2
    return distances


def _distance_summary(values):
    if not len(values):
        return dict(mean=None, median=None, min=None, max=None)
    return dict(mean=float(np.mean(values)), median=float(np.median(values)),
                min=float(np.min(values)), max=float(np.max(values)))


def summarize_diversity(distances, labels=None):
    """Summarize pair distances, nearest neighbours and within/between cluster means.

    Parameters
    ----------
    distances : (n, n) ndarray
        Finite nonnegative symmetric matrix with zero diagonal.
    labels : array_like or None

    Returns
    -------
    diversity : dict
        Empty pair statistics are None; between means include all meshes outside a cluster.
    """
    matrix = np.asarray(distances)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.dtype.kind not in 'fiu':
        raise ValueError('distances must be a square real matrix')
    matrix = matrix.astype(float)
    if not np.isfinite(matrix).all() or np.any(matrix < 0):
        raise ValueError('distances must be finite and nonnegative')
    if not np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-12) or np.any(np.diag(matrix) != 0):
        raise ValueError('distances must be symmetric with zero diagonal')
    n_meshes = len(matrix)
    result = _distance_summary(matrix[np.triu_indices(n_meshes, 1)])
    other = matrix.copy()
    np.fill_diagonal(other, np.inf)
    nearest = other.min(axis=1) if n_meshes > 1 else np.array([])
    result['nearest_neighbor'] = _distance_summary(nearest)
    result['delta_mean'] = float(np.mean(np.sqrt(matrix.sum(axis=1) / (n_meshes - 1))) ** 2) if n_meshes > 1 else None
    result['clusters'] = None
    if labels is not None:
        labels = np.asarray(labels)
        if labels.shape != (n_meshes,):
            raise ValueError('labels must contain one cluster label per mesh')
        clusters = {}
        for label in np.unique(labels).tolist():
            inside = np.flatnonzero(labels == label)
            outside = np.flatnonzero(labels != label)
            block = matrix[np.ix_(inside, inside)]
            within = block[np.triu_indices(len(inside), 1)]
            between = matrix[np.ix_(inside, outside)]
            within_mean = float(within.mean()) if within.size else None
            between_mean = float(between.mean()) if between.size else None
            clusters[label] = dict(count=len(inside), within_mean=within_mean, between_mean=between_mean)
        result['clusters'] = clusters
    return result


# ---------------------------------------------------------------------------------------------------- Evaluation
def evaluate_mesh(vertices, faces, reference=None):
    """Combine quality, descriptors and comparison (None without a reference).

    Parameters
    ----------
    vertices, faces : (n, 3), (m, 3) ndarray
    reference : (vertices, faces) pair or None

    Returns
    -------
    evaluation : dict
    """
    if reference is not None and (not isinstance(reference, (tuple, list)) or len(reference) != 2):
        raise ValueError('reference must be a (vertices, faces) pair')
    return dict(quality=mesh_quality(vertices, faces), descriptors=mesh_descriptors(vertices, faces),
                comparison=None if reference is None else compare_surfaces(*reference, vertices, faces))


# ---------------------------------------------------------------------------------------------------- Pipeline hook
def repair_and_report(logger, mesh_, name_, folder_):
    """
    Repair a reconstructed mesh and save its quality before and after repair as json.

    Parameters
    ----------
    logger: logging.Logger
        Logger used for the repair stages.
    mesh_: open3d.geometry.TriangleMesh
        Mesh returned by sdf_pca_utils.reconstruct_mesh.
    name_: str
        Name of the json file written in folder_ (extension not required).
    folder_: str
        Output folder, e.g. results/jsons/<case>.

    Returns
    -------
    repaired_mesh: open3d.geometry.TriangleMesh
        Repaired copy; the input mesh is not modified.
    """
    vertices, faces = np.asarray(mesh_.vertices), np.asarray(mesh_.triangles)
    before = mesh_quality(vertices, faces)
    new_vertices, new_faces, info = repair_mesh(vertices, faces, logger=logger)
    after = mesh_quality(new_vertices, new_faces)
    logger.info(f"\t{name_}: repair success {info['success']}, faces {len(faces)} -> {len(new_faces)}, "
                f"self-intersecting faces {before['self_intersecting_faces']} -> {after['self_intersecting_faces']}")

    # numpy scalars (np.int64, np.bool_) are not json serializable, .item() converts them
    with open(os.path.join(folder_, name_ + '_repair.json'), 'w') as outfile:
        json.dump(dict(before=before, repair=info, after=after), outfile, indent=4, default=lambda x: x.item())
    return o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(new_vertices), o3d.utility.Vector3iVector(new_faces))
