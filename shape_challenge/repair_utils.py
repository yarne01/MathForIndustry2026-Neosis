"""MeshFix-style repair using NumPy/SciPy; triangulations need not match native MeshFix.

Predicates use float64 tolerances. Difficult holes may remain open in the best returned mesh.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph
import scipy.sparse.linalg as spla
import scipy.spatial as spatial


# ---------------------------------------------------------------------------------------------------- Topology
def _mesh(vertices, faces):
    """Validate mesh arrays and return independent float64/int64 copies."""
    vertices, faces = np.asarray(vertices), np.asarray(faces)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.dtype.kind not in 'fiu'
            or not np.isfinite(vertices).all()):
        raise ValueError('vertices must be a finite numeric (n, 3) array')
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype.kind not in 'iu':
        raise ValueError('faces must be an integer (m, 3) array')
    if faces.size and (faces.min() < 0 or faces.max() >= len(vertices)):
        raise ValueError('faces contain out-of-range vertex indices')
    return vertices.astype(float, copy=True), faces.astype(np.int64, copy=True)


def _edges(faces):
    """Group directed triangle edges by their undirected vertex pair."""
    directed_edges = faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
    key_base = int(faces.max()) + 1 if faces.size else 1
    sorted_edges = np.sort(directed_edges, axis=1)
    # int64 keys avoid structured sorting and overflow in SciPy's int32 component labels.
    if key_base * key_base <= np.iinfo(np.int64).max:
        encoded_edges = sorted_edges.astype(np.int64, copy=False)
        keys = encoded_edges[:, 0] * key_base + encoded_edges[:, 1]
        keys, edge_ids, edge_counts = np.unique(keys, return_inverse=True, return_counts=True)
        edges = np.column_stack((keys // key_base, keys % key_base))
    else:
        edges, edge_ids, edge_counts = np.unique(sorted_edges, axis=0, return_inverse=True, return_counts=True)
    edge_order = np.argsort(edge_ids, kind='stable')
    return directed_edges, edges, edge_counts, edge_order, np.r_[0, np.cumsum(edge_counts)]


def _compact(vertices, faces):
    """Drop unused vertices and remap triangle indices in linear time."""
    used = np.zeros(len(vertices), dtype=bool)
    used[faces] = True
    remap = np.cumsum(used) - 1
    return vertices[used].copy(), remap[faces]


def _adjacency(faces):
    """Return directed edges and pairs of edge slots shared by two triangles."""
    directed_edges, _, edge_counts, edge_order, offsets = _edges(faces)
    paired = offsets[:-1][edge_counts == 2]
    edge_pairs = np.column_stack((edge_order[paired], edge_order[paired + 1]))
    return directed_edges, edge_pairs


def _components(faces):
    """Label edge-connected face components, including non-manifold edges."""
    _, _, edge_counts, edge_order, offsets = _edges(faces)
    first_faces = np.repeat(edge_order[offsets[:-1]] // 3, edge_counts - 1)
    group_first = np.zeros(len(edge_order), dtype=bool)
    group_first[offsets[:-1]] = True
    adjacent_faces = edge_order[~group_first] // 3
    rows, columns = np.r_[first_faces, adjacent_faces], np.r_[adjacent_faces, first_faces]
    graph = sp.coo_matrix((np.ones(len(rows)), (rows, columns)), shape=(len(faces), len(faces)))
    return csgraph.connected_components(graph, directed=False)


def merge_duplicate_vertices(vertices, faces):
    """Weld exactly coincident coordinates, as in MeshFix's loader.

    Parameters
    ----------
    vertices, faces : ndarray, shapes (n, 3) and (m, 3)

    Returns
    -------
    vertices, faces : ndarray
        New coordinates and remapped triangles.
    """
    vertices, faces = _mesh(vertices, faces)
    vertices, inverse = np.unique(vertices, axis=0, return_inverse=True)
    return vertices, inverse[faces]


def remove_degenerate_faces(vertices, faces):
    """Drop repeated-index and duplicate triangles, as in MeshFix's loader.

    Parameters
    ----------
    vertices, faces : ndarray, shapes (n, 3) and (m, 3)

    Returns
    -------
    vertices, faces : ndarray
        Copies with repeated-index and duplicate triangles removed; collinear faces remain.
    """
    vertices, faces = _mesh(vertices, faces)
    faces = faces[(faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 2] != faces[:, 0])]
    _, first = np.unique(np.sort(faces, axis=1), axis=0, return_index=True)
    return vertices, faces[np.sort(first)]


def orient_faces(vertices, faces):
    """BFS-orient each sheet and turn closed sheets outward, like MeshFix.

    Parameters
    ----------
    vertices, faces : ndarray
        Vertex coordinates and edge-manifold triangles.

    Returns
    -------
    faces : ndarray
        Consistently wound triangles; non-orientable sheets raise ValueError.
    """
    vertices, faces = _mesh(vertices, faces)
    faces, conflicts = _orient_sheets(vertices, faces)
    if len(conflicts):
        raise ValueError('faces contain a non-orientable sheet; run fix_connectivity first')
    return faces


def _orient_sheets(vertices, faces):
    """Orient a spanning forest, turn closed sheets outward and identify conflicting adjacencies."""
    directed_edges, _, edge_counts, edge_order, offsets = _edges(faces)
    if np.any(edge_counts > 2):
        raise ValueError('orient_faces requires edge-manifold faces; run fix_connectivity first')
    paired = offsets[:-1][edge_counts == 2]
    edge_pairs = np.column_stack((edge_order[paired], edge_order[paired + 1]))
    first_faces, second_faces = (edge_pairs // 3).T
    signs = np.where(directed_edges[edge_pairs[:, 0], 0] == directed_edges[edge_pairs[:, 1], 0], -1, 1)
    graph = sp.csr_matrix((np.r_[signs, signs], (np.r_[first_faces, second_faces],
                                               np.r_[second_faces, first_faces])), shape=(len(faces), len(faces)))
    component_count, labels = csgraph.connected_components(graph, directed=False)
    _, roots = np.unique(labels, return_index=True)
    # A common root lets SciPy traverse all sheets once, even for disconnected triangle soup.
    root_column = sp.csr_matrix((np.ones(component_count), (roots, np.zeros(component_count, dtype=int))),
                                shape=(len(faces), 1))
    forest = sp.bmat([[graph, root_column], [root_column.T, None]], format='csr')
    _, parents = csgraph.breadth_first_order(forest, len(faces), directed=False)
    parents[-1] = len(faces)
    flips = np.asarray(forest[parents, np.arange(len(parents))]).ravel().astype(int)
    flips[-1] = 1
    # Pointer doubling propagates winding without a Python loop over faces.
    while np.any(parents != parents[parents]):
        flips *= flips[parents]
        parents = parents[parents]
    conflicts = edge_pairs[flips[first_faces] * signs != flips[second_faces]] // 3
    faces = faces.copy()
    faces[flips[:-1] < 0] = faces[flips[:-1] < 0, ::-1]
    closed = np.ones(component_count, dtype=bool)
    closed[labels[edge_order[offsets[:-1][edge_counts != 2]] // 3]] = False
    triangles = vertices[faces]
    triangles = triangles - vertices[faces[roots[labels], 0]][:, None]
    volumes = np.bincount(labels, weights=np.einsum('ij,ij->i', triangles[:, 0],
                                                  np.cross(triangles[:, 1], triangles[:, 2])))
    inward = (closed & (volumes < 0))[labels]
    faces[inward] = faces[inward, ::-1]
    return faces, conflicts


def _split_fans(vertices, faces):
    """Give disconnected triangle fans at each vertex separate coordinate copies."""
    directed_edges, edge_pairs = _adjacency(faces)
    ascending = directed_edges[edge_pairs, 0] < directed_edges[edge_pairs, 1]
    low_corners = np.where(ascending, edge_pairs % 3, (edge_pairs % 3 + 1) % 3)
    high_corners = np.where(ascending, (edge_pairs % 3 + 1) % 3, edge_pairs % 3)
    corner_pairs = np.concatenate((edge_pairs // 3 * 3 + low_corners, edge_pairs // 3 * 3 + high_corners))
    first_corners, second_corners = corner_pairs.T
    rows, columns = np.r_[first_corners, second_corners], np.r_[second_corners, first_corners]
    graph = sp.coo_matrix((np.ones(len(rows)), (rows, columns)), shape=(faces.size, faces.size))
    _, labels = csgraph.connected_components(graph, directed=False)
    _, representatives = np.unique(labels, return_index=True)
    return vertices[faces.ravel()[representatives]].copy(), labels.reshape(-1, 3)


def fix_connectivity(vertices, faces):
    """Weld, deduplicate, cut excess edge sheets and vertex fans, then orient as MeshFix's loader does.

    Parameters
    ----------
    vertices, faces : ndarray, shapes (n, 3) and (m, 3)

    Returns
    -------
    vertices, faces : ndarray
        Compact, consistently oriented manifold sheets with unchanged triangle coordinates.
    """
    vertices, faces = merge_duplicate_vertices(vertices, faces)
    vertices, faces = remove_degenerate_faces(vertices, faces)
    while len(faces):
        directed_edges, _, edge_counts, edge_order, offsets = _edges(faces)
        if not np.any(edge_counts > 2):
            break
        # Detach excess triangles at both endpoints so alternative fan paths cannot rejoin the edge.
        excess = np.arange(len(edge_order)) - np.repeat(offsets[:-1], edge_counts) >= 2
        excess_edges = edge_order[excess]
        face_ids = np.repeat(excess_edges // 3, 2)
        corner_slots = np.column_stack((excess_edges % 3, (excess_edges % 3 + 1) % 3)).ravel()
        detached_corners = np.unique(face_ids * 3 + corner_slots)
        vertices = np.vstack((vertices, vertices[faces.ravel()[detached_corners]]))
        faces.ravel()[detached_corners] = np.arange(len(vertices) - len(detached_corners), len(vertices))
    vertices, faces = _split_fans(vertices, faces)
    faces, conflicts = _orient_sheets(vertices, faces)
    if len(conflicts):
        detached_faces = np.unique(conflicts[:, 1])
        vertices = np.vstack((vertices, vertices[faces[detached_faces].ravel()]))
        faces[detached_faces] = np.arange(len(vertices) - 3 * len(detached_faces), len(vertices)).reshape(-1, 3)
        vertices, faces = _split_fans(vertices, faces)
        faces, _ = _orient_sheets(vertices, faces)
    return _compact(vertices, faces)


def keep_largest_component(vertices, faces):
    """Keep the edge-connected shell with most triangles, like removeSmallestComponents.

    Parameters
    ----------
    vertices, faces : ndarray, shapes (n, 3) and (m, 3)

    Returns
    -------
    vertices, faces : ndarray
        Compact largest shell; ties keep the first shell.
    """
    vertices, faces = _mesh(vertices, faces)
    if len(faces):
        _, labels = _components(faces)
        faces = faces[labels == np.bincount(labels).argmax()]
    return _compact(vertices, faces)


def boundary_loops(faces):
    """Trace ordered boundary cycles for MeshFix-style hole triangulation.

    Parameters
    ----------
    faces : ndarray
        Nonnegative integer triangle indices (m, 3), consistently oriented and manifold.

    Returns
    -------
    loops : list of ndarray
        Simple cycles in face direction; touching cycles are split and open trails are skipped.
    """
    faces = np.asarray(faces)
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype.kind not in 'iu' or np.any(faces < 0):
        raise ValueError('faces must be a nonnegative integer (m, 3) array')
    directed_edges, _, edge_counts, edge_order, offsets = _edges(faces)
    boundary = directed_edges[edge_order[offsets[:-1][edge_counts == 1]]]
    following = {}
    for first, second in boundary:
        following.setdefault(int(first), []).append(int(second))
    loops = []
    while following:
        vertex = next(iter(following))
        walk, positions = [], {}
        while vertex in following:
            positions[vertex] = len(walk)
            walk.append(vertex)
            neighbours = following[vertex]
            vertex = neighbours.pop()
            if not neighbours:
                del following[walk[-1]]
            if vertex in positions:
                beginning = positions[vertex]
                cycle = walk[beginning:]
                if len(cycle) >= 3:
                    loops.append(np.array(cycle, dtype=int))
                for visited in cycle:
                    del positions[visited]
                del walk[beginning:]
    return loops


# ---------------------------------------------------------------------------------------------------- Intersections
def _segment_hits(segment_start, segment_end, triangles):
    """Test paired segments against triangles, including coplanar overlaps."""
    corner, second_corner, third_corner = triangles.transpose(1, 0, 2)
    first_edge, second_edge, direction = second_corner - corner, third_corner - corner, segment_end - segment_start
    normal = np.cross(first_edge, second_edge)
    direction_cross = np.cross(direction, second_edge)
    determinant = np.einsum('ij,ij->i', first_edge, direction_cross)
    tolerance = 64 * np.finfo(float).eps
    normal_length = np.linalg.norm(normal, axis=1)
    valid = np.abs(determinant) > tolerance * normal_length * np.linalg.norm(direction, axis=1)
    inverse = np.divide(1., determinant, out=np.zeros_like(determinant), where=valid)
    offset = segment_start - corner
    first_weight = np.einsum('ij,ij->i', offset, direction_cross) * inverse
    offset_cross = np.cross(offset, first_edge)
    second_weight = np.einsum('ij,ij->i', direction, offset_cross) * inverse
    fraction = np.einsum('ij,ij->i', second_edge, offset_cross) * inverse
    hits = (valid & (first_weight >= -tolerance) & (second_weight >= -tolerance)
            & (first_weight + second_weight <= 1 + tolerance) & (fraction >= -tolerance) & (fraction <= 1 + tolerance))
    coplanar = (~valid & (normal_length > 0)
                & (np.abs(np.einsum('ij,ij->i', offset, normal)) <= tolerance * normal_length))
    indices = np.flatnonzero(coplanar)
    if len(indices):
        normal = normal[indices]
        start, end, triangles = segment_start[indices], segment_end[indices], triangles[indices]
        edges = np.roll(triangles, -1, axis=1) - triangles
        start_sides = np.einsum('nij,nj->ni', np.cross(edges, start[:, None] - triangles), normal)
        end_sides = np.einsum('nij,nj->ni', np.cross(edges, end[:, None] - triangles), normal)
        hits[indices] |= (start_sides >= 0).all(axis=1) | (end_sides >= 0).all(axis=1)
        for edge in range(3):
            edge_start, edge_end = triangles[:, edge], triangles[:, (edge + 1) % 3]
            determinant = np.einsum('ij,ij->i', np.cross(end - start, edge_end - edge_start), normal)
            valid = determinant != 0
            inverse = np.divide(1., determinant, out=np.zeros_like(determinant), where=valid)
            along_segment = np.einsum('ij,ij->i', np.cross(edge_start - start, edge_end - edge_start), normal) * inverse
            along_edge = np.einsum('ij,ij->i', np.cross(edge_start - start, end - start), normal) * inverse
            hits[indices] |= (valid & (along_segment >= -tolerance) & (along_segment <= 1 + tolerance)
                              & (along_edge >= -tolerance) & (along_edge <= 1 + tolerance))
    return hits


def _triangle_hits(vertices, faces, face_pairs):
    """Filter candidate face pairs to crossings away from shared vertices and edges."""
    first_faces, second_faces = faces[face_pairs[:, 0]], faces[face_pairs[:, 1]]
    shared_vertices = (first_faces[:, :, None] == second_faces[:, None, :])
    shared_counts = shared_vertices.any(axis=2).sum(axis=1)
    candidate_ids = np.flatnonzero(shared_counts < 2)
    if not len(candidate_ids):
        return np.empty((0, 2), dtype=int)
    first_faces, second_faces = first_faces[candidate_ids], second_faces[candidate_ids]
    shared_vertices, shared_counts = shared_vertices[candidate_ids], shared_counts[candidate_ids]
    first_triangles, second_triangles = vertices[first_faces], vertices[second_faces]
    origins = first_triangles[:, :1].copy()
    scale = np.maximum(np.ptp(first_triangles, axis=1).max(axis=1), np.ptp(second_triangles, axis=1).max(axis=1))
    scale[scale == 0] = 1
    first_triangles = (first_triangles - origins) / scale[:, None, None]
    second_triangles = (second_triangles - origins) / scale[:, None, None]
    hits = np.zeros(len(candidate_ids), dtype=bool)
    for edge in range(3):
        following = (edge + 1) % 3
        first_allowed = ~(shared_vertices[:, edge].any(axis=1) | shared_vertices[:, following].any(axis=1))
        second_allowed = ~(shared_vertices[:, :, edge].any(axis=1) | shared_vertices[:, :, following].any(axis=1))
        hits |= first_allowed & _segment_hits(first_triangles[:, edge], first_triangles[:, following], second_triangles)
        hits |= second_allowed & _segment_hits(second_triangles[:, edge], second_triangles[:, following],
                                               first_triangles)
    return face_pairs[candidate_ids[hits]]


def self_intersecting_faces(vertices, faces):
    """Find float64 triangle crossings using bounded centroid trees and vectorized predicates.

    Parameters
    ----------
    vertices, faces : ndarray, shapes (n, 3) and (m, 3)
        Shared edges and isolated shared vertices are ignored.

    Returns
    -------
    face_ids : ndarray
        Sorted indices of intersecting faces, including coplanar overlaps.
    """
    vertices, faces = _mesh(vertices, faces)
    return _intersecting_faces(vertices, faces)


def _intersecting_faces(vertices, faces, selected=None):
    """Check all crossings, or only pairs involving selected faces."""
    if not len(faces):
        return np.empty(0, dtype=int)
    selected_mask = np.zeros(len(faces), dtype=bool)
    if selected is not None:
        selected_mask[selected] = True
        if not selected_mask.any():
            return np.empty(0, dtype=int)
    triangles = vertices[faces]
    centers = triangles.mean(axis=1)
    radii = np.linalg.norm(triangles - centers[:, None], axis=2).max(axis=1)
    low, high = triangles.min(axis=1), triangles.max(axis=1)
    minimum_radius = max(radii.max() * 1e-12, np.finfo(float).tiny)
    bins = np.floor(np.log2(np.maximum(radii, minimum_radius) / minimum_radius)).astype(int)
    found = np.zeros(len(faces), dtype=bool)
    for size_bin in np.unique(bins):
        group = np.flatnonzero(bins == size_bin)
        tree = spatial.cKDTree(centers[group])
        radius = radii[group].max()
        queries = np.flatnonzero(bins <= size_bin) if selected is None else np.flatnonzero(selected_mask)
        for start in range(0, len(queries), 4096):
            query_ids = queries[start:start + 4096]
            neighbours = tree.query_ball_point(centers[query_ids], np.nextafter(radii[query_ids] + radius, np.inf),
                                               return_sorted=False)
            lengths = np.fromiter(map(len, neighbours), dtype=int, count=len(query_ids))
            first_faces = np.repeat(query_ids, lengths)
            second_faces = group[np.concatenate(neighbours).astype(int)]
            if selected is None:
                candidates = (bins[first_faces] < size_bin) | (first_faces < second_faces)
            else:
                candidates = (first_faces < second_faces) | ~selected_mask[second_faces]
            candidates &= (low[first_faces] <= high[second_faces]).all(axis=1)
            candidates &= (low[second_faces] <= high[first_faces]).all(axis=1)
            face_pairs = np.column_stack((first_faces[candidates], second_faces[candidates]))
            for offset in range(0, len(face_pairs), 16384):
                hits = _triangle_hits(vertices, faces, face_pairs[offset:offset + 16384])
                found[hits.ravel()] = True
    return np.flatnonzero(found)


# ---------------------------------------------------------------------------------------------------- Holes
def _cross2(first, second):
    """Compute signed cross products of two-dimensional vectors."""
    return first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0]


def _ear_angle(xy, polygon, previous, following, active, index, occupied, tolerance):
    """Score a convex, empty projected ear whose diagonal is not already occupied."""
    before, after = previous[index], following[index]
    incoming, outgoing = xy[before] - xy[index], xy[after] - xy[index]
    if _cross2(outgoing, incoming) <= tolerance or tuple(sorted((polygon[before], polygon[after]))) in occupied:
        return np.inf
    points = xy[active & ~np.isin(np.arange(len(polygon)), [before, index, after])]
    triangle = xy[[before, index, after]]
    sides = _cross2(np.roll(triangle, -1, axis=0) - triangle, points[:, None] - triangle)
    if np.any((sides >= -tolerance).all(axis=1)):
        return np.inf
    return np.arctan2(abs(_cross2(incoming, outgoing)), np.dot(incoming, outgoing))


def _triangulate(vertices, loop, occupied):
    """Ear-clip a reversed boundary, using minimum-area ears for difficult projections."""
    polygon = loop[::-1]
    xyz = vertices[polygon] - vertices[polygon].mean(axis=0)
    xy = xyz @ np.linalg.svd(xyz, full_matrices=False)[2][:2].T
    if np.sum(_cross2(xy, np.roll(xy, -1, axis=0))) < 0:
        xy[:, 1] *= -1
    count = len(polygon)
    previous, following = (np.arange(count) - 1) % count, (np.arange(count) + 1) % count
    active, triangles = np.ones(count, dtype=bool), []
    tolerance = np.finfo(float).eps * max(np.ptp(xy, axis=0).max() ** 2, np.finfo(float).tiny) * 64

    angles = np.array([_ear_angle(xy, polygon, previous, following, active, index, occupied, tolerance)
                       for index in range(count)])
    while active.sum() > 3:
        index = angles.argmin()
        if not np.isfinite(angles[index]):
            candidates = np.flatnonzero(active)
            candidates = np.array([neighbour for neighbour in candidates
                                   if tuple(sorted(polygon[[previous[neighbour], following[neighbour]]]))
                                   not in occupied],
                                  dtype=int)
            if not len(candidates):
                return np.empty((0, 3), dtype=int)
            # Folded or self-crossing loops may have no valid ear in this projection.
            ears = xyz[np.column_stack((previous[candidates], candidates, following[candidates]))]
            areas = np.linalg.norm(np.cross(ears[:, 1] - ears[:, 0], ears[:, 2] - ears[:, 0]), axis=1)
            index = candidates[areas.argmin()]
        before, after = previous[index], following[index]
        triangles.append(polygon[[before, index, after]])
        active[index], angles[index], following[before], previous[after] = False, np.inf, after, before
        for neighbour in (before, after):
            angles[neighbour] = _ear_angle(xy, polygon, previous, following, active, neighbour, occupied, tolerance)
    triangles.append(polygon[np.flatnonzero(active)])
    return np.array(triangles, dtype=int)


def _relax_patch(vertices, patch, selected=None, occupied=None):
    """Flip disjoint interior edges when triangle quality and winding improve."""
    for _ in range(5):
        directed_edges, edge_pairs = _adjacency(patch)
        face_pairs = edge_pairs // 3
        edge_start, edge_end = directed_edges[edge_pairs[:, 0]].T
        opposite_first = patch[face_pairs[:, 0], (edge_pairs[:, 0] % 3 + 2) % 3]
        opposite_second = patch[face_pairs[:, 1], (edge_pairs[:, 1] % 3 + 2) % 3]
        new = np.stack((np.column_stack((opposite_first, opposite_second, edge_end)),
                        np.column_stack((opposite_second, opposite_first, edge_start))), axis=1)
        old_xyz, new_xyz = vertices[patch[face_pairs]], vertices[new]
        old_normal = np.cross(old_xyz[:, :, 1] - old_xyz[:, :, 0], old_xyz[:, :, 2] - old_xyz[:, :, 0])
        new_normal = np.cross(new_xyz[:, :, 1] - new_xyz[:, :, 0], new_xyz[:, :, 2] - new_xyz[:, :, 0])
        old_quality = np.linalg.norm(old_normal, axis=2) / np.maximum(
            ((np.roll(old_xyz, -1, axis=2) - old_xyz) ** 2).sum(axis=(2, 3)), np.finfo(float).tiny)
        new_quality = np.linalg.norm(new_normal, axis=2) / np.maximum(
            ((np.roll(new_xyz, -1, axis=2) - new_xyz) ** 2).sum(axis=(2, 3)), np.finfo(float).tiny)
        existing = _edges(patch)[1]
        key = np.minimum(opposite_first, opposite_second) * len(vertices) + np.maximum(opposite_first, opposite_second)
        existing_keys = existing[:, 0] * len(vertices) + existing[:, 1]
        good = ((new_quality.min(axis=1) > 1.01 * old_quality.min(axis=1)) & ~np.isin(key, existing_keys)
                & (np.einsum('nki,nki->nk', new_normal, old_normal) > 0).all(axis=1))
        if selected is not None:
            good &= selected[face_pairs].any(axis=1)
        candidates = np.flatnonzero(good)
        if occupied is not None:
            diagonals = np.sort(np.column_stack((opposite_first[candidates], opposite_second[candidates])), axis=1)
            blocked = np.fromiter((tuple(diagonal) in occupied for diagonal in diagonals),
                                  dtype=bool, count=len(candidates))
            candidates = candidates[~blocked]
        _, unique_diagonals = np.unique(key[candidates], return_index=True)
        candidates = candidates[np.sort(unique_diagonals)]
        winner = np.full(len(patch), len(face_pairs))
        np.minimum.at(winner, face_pairs[candidates].ravel(), np.repeat(candidates, 2))
        chosen = candidates[(winner[face_pairs[candidates]] == candidates[:, None]).all(axis=1)]
        if not len(chosen):
            break
        patch[face_pairs[chosen].ravel()] = new[chosen].reshape(-1, 3)
    return patch


def fill_holes(vertices, faces, max_hole_edges=0, refine=True):
    """Ear-clip and refine boundary patches with fixed original vertices.

    Parameters
    ----------
    vertices, faces : ndarray
        Consistently oriented manifold mesh arrays.
    max_hole_edges : int
        Maximum cycle length to fill; zero fills every hole.
    refine : bool
        Refine at the boundary edge scale and harmonically fair inserted vertices.

    Returns
    -------
    vertices, faces : ndarray
        Filled mesh with original coordinates fixed; untriangulable holes remain open.
    """
    vertices, faces = _mesh(vertices, faces)
    if not isinstance(max_hole_edges, (int, np.integer)) or max_hole_edges < 0:
        raise ValueError('max_hole_edges must be a nonnegative integer')
    fixed_count = len(vertices)
    occupied = set(map(tuple, _edges(faces)[1]))
    patches = []
    for loop in boundary_loops(faces):
        if max_hole_edges and len(loop) > max_hole_edges:
            continue
        try:
            patch = _triangulate(vertices, loop, occupied)
        except np.linalg.LinAlgError:
            continue
        if not len(patch):
            continue
        if refine:
            edge_length = np.linalg.norm(vertices[loop] - vertices[np.roll(loop, 1)], axis=1).mean()
            if edge_length == 0:
                continue
            for _ in range(8):
                xyz = vertices[patch]
                area = np.linalg.norm(np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]), axis=1) / 2
                split = area > .65 * edge_length * edge_length
                if not split.any():
                    break
                centers = np.arange(len(vertices), len(vertices) + split.sum())
                vertices = np.vstack((vertices, xyz[split].mean(axis=1)))
                triangles = patch[split]
                patch = np.vstack((patch[~split], np.column_stack((triangles[:, 0], triangles[:, 1], centers)),
                                   np.column_stack((triangles[:, 1], triangles[:, 2], centers)),
                                   np.column_stack((triangles[:, 2], triangles[:, 0], centers))))
                patch = _relax_patch(vertices, patch, occupied=occupied)
        patches.append(patch)
        occupied.update(map(tuple, _edges(patch)[1]))
    if patches:
        faces = np.vstack([faces] + patches)
    if len(vertices) > fixed_count:
        edges = _edges(faces)[1]
        first_vertices, second_vertices = edges.T
        rows, columns = np.r_[first_vertices, second_vertices], np.r_[second_vertices, first_vertices]
        graph = sp.csr_matrix((np.ones(len(rows)), (rows, columns)), shape=(len(vertices), len(vertices)))
        laplacian = sp.eye(len(vertices)) - sp.diags(1 / np.maximum(graph.sum(axis=1).A.ravel(), 1)) @ graph
        # Positive harmonic weights keep each patch inside its boundary convex hull.
        fair = spla.spsolve(laplacian[fixed_count:, fixed_count:].tocsc(),
                            -laplacian[fixed_count:, :fixed_count] @ vertices[:fixed_count])
        if not np.isfinite(fair).all():
            raise ValueError('hole fairing produced non-finite coordinates')
        vertices[fixed_count:] = fair
    if patches:
        vertices, faces = remove_degenerate_faces(vertices, faces)
        if np.any(_edges(faces)[2] > 2):
            vertices, faces = fix_connectivity(vertices, faces)
        else:
            faces, _ = _orient_sheets(vertices, faces)
    return vertices, faces


# ---------------------------------------------------------------------------------------------------- Cleaning
def _bad_triangles(vertices, faces):
    """Mark triangles with near-zero area relative to their longest squared edge."""
    xyz = vertices[faces]
    lengths = ((np.roll(xyz, -1, axis=1) - xyz) ** 2).sum(axis=2)
    area2 = np.linalg.norm(np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]), axis=1)
    return area2 <= 1e-10 * lengths.max(axis=1)


def _remove_needles(vertices, faces):
    """Improve needle triangles by edge flips, then collapse their shortest edges."""
    for _ in range(5):
        bad = _bad_triangles(vertices, faces)
        if not bad.any():
            break
        faces = _relax_patch(vertices, faces, selected=bad)
        bad = _bad_triangles(vertices, faces)
        if not bad.any():
            break
        xyz = vertices[faces[bad]]
        shortest = ((np.roll(xyz, -1, axis=1) - xyz) ** 2).sum(axis=2).argmin(axis=1)
        edges = np.column_stack((faces[bad][np.arange(bad.sum()), shortest],
                                 faces[bad][np.arange(bad.sum()), (shortest + 1) % 3]))
        # Connected collapse sets are handled together, avoiding order-dependent remapping chains.
        graph = sp.coo_matrix((np.ones(edges.size), (edges.ravel(), edges[:, ::-1].ravel())),
                              shape=(len(vertices), len(vertices)))
        _, labels = csgraph.connected_components(graph, directed=False)
        _, first = np.unique(labels, return_index=True)
        vertices, faces = remove_degenerate_faces(vertices[first], labels[faces])
    return _compact(vertices, faces)


def mesh_clean(vertices, faces, max_iters=10, inner_loops=3):
    """Collapse degeneracies and repair crossings locally without selecting a largest component.

    Parameters
    ----------
    vertices, faces : ndarray, shapes (n, 3) and (m, 3)
    max_iters, inner_loops : int
        Cleaning attempts and retries: crossing faces, one edge ring, then one vertex ring.
        At most three retries are used; open fragments below max(4 faces, 0.1%) are discarded.

    Returns
    -------
    vertices, faces, success : ndarray, ndarray, bool
        Repaired arrays and whether a closed manifold without crossings or degeneracies was obtained.
    """
    vertices, faces = _mesh(vertices, faces)
    if any(not isinstance(value, (int, np.integer)) or value < 1 for value in (max_iters, inner_loops)):
        raise ValueError('max_iters and inner_loops must be positive integers')
    hits = None
    for _ in range(max_iters):
        if _bad_triangles(vertices, faces).any():
            vertices, faces = _remove_needles(vertices, faces)
            vertices, faces = fix_connectivity(vertices, faces)
            vertices, faces = _remove_debris(vertices, faces)
            vertices, faces = fill_holes(vertices, faces)
            hits = None
        if hits is None:
            hits = _intersecting_faces(vertices, faces)
        if not len(hits):
            if _bad_triangles(vertices, faces).any():
                continue
            return vertices, faces, _closed_manifold(vertices, faces)
        # Keep successful local edits; later retries target only the remaining crossings.
        for retry in range(min(inner_loops, 3)):
            original_vertices, original_faces = vertices, faces
            removed = np.zeros(len(faces), dtype=bool)
            removed[hits] = True
            if retry == 1:
                edge_neighbours = _adjacency(original_faces)[1] // 3
                touching = removed[edge_neighbours].any(axis=1)
                removed[edge_neighbours[touching].ravel()] = True
            elif retry == 2:
                touched_vertices = np.unique(original_faces[hits])
                removed = np.isin(original_faces, touched_vertices).any(axis=1)
            if removed.all():
                return vertices, faces, False
            # Cutting an existing sheet needs fan splitting, not a global coordinate weld.
            vertices, faces = _split_fans(original_vertices, original_faces[~removed])
            vertices, faces = _remove_debris(vertices, faces)
            kept_vertices, kept_faces = vertices, faces
            vertices, faces = fill_holes(vertices, faces)
            # Retained coordinates are fixed, but fan splitting can change which faces are adjacent.
            unchanged = (len(faces) >= len(kept_faces)
                         and np.array_equal(vertices[:len(kept_vertices)], kept_vertices)
                         and np.array_equal(np.sort(faces[:len(kept_faces)], axis=1), np.sort(kept_faces, axis=1)))
            selected = None
            if unchanged:
                _, vertex_ids, copies = np.unique(kept_vertices, axis=0, return_inverse=True, return_counts=True)
                detached = (copies[vertex_ids][kept_faces] > 1).any(axis=1)
                selected = np.r_[np.flatnonzero(detached), np.arange(len(kept_faces), len(faces))]
            hits = _intersecting_faces(vertices, faces, selected)
            if not len(hits):
                break
    return vertices, faces, bool(_closed_manifold(vertices, faces) and not _bad_triangles(vertices, faces).any()
                                 and not len(hits))


def _remove_debris(vertices, faces):
    """Drop tiny open fragments; retain every closed component and every substantial open sheet."""
    count, labels = _components(faces)
    sizes = np.bincount(labels)
    _, _, counts, order, offsets = _edges(faces)
    opened = np.zeros(count, dtype=bool)
    opened[labels[order[offsets[:-1][counts != 2]] // 3]] = True
    keep = (sizes >= max(4, .001 * len(faces))) | ~opened
    faces = faces[keep[labels]]
    return _compact(vertices, faces)


def _closed_manifold(vertices, faces):
    """Check closure, unique triangles, connected vertex fans and consistent winding."""
    if (not len(faces) or not np.all(_edges(faces)[2] == 2)
            or len(np.unique(np.sort(faces, axis=1), axis=0)) != len(faces)):
        return False
    _, split = _split_fans(vertices, faces)
    directed_edges, edge_pairs = _adjacency(faces)
    return bool(len(np.unique(split)) == len(np.unique(faces))
                and np.all(directed_edges[edge_pairs[:, 0], 0] == directed_edges[edge_pairs[:, 1], 1]))


def _stage_counts(vertices, faces, logger, stage):
    """Count stage geometry and optionally emit the same counts to a logger."""
    counts = dict(vertices=len(vertices), faces=len(faces), components=int(_components(faces)[0]),
                  boundary_loops=len(boundary_loops(faces)))
    if logger is not None:
        logger.info('%s: %d vertices, %d faces, %d components, %d boundary loops', stage, *counts.values())
    return counts


def repair_mesh(vertices, faces, logger=None, keep_largest=True, fill=True, clean=True):
    """Run the MeshFix loader, largest-shell, fill-all and meshclean sequence with stage counts.

    Parameters
    ----------
    vertices, faces : ndarray, shapes (n, 3) and (m, 3)
    logger : logging.Logger, optional
        Receives one info message for each completed stage.
    keep_largest, fill, clean : bool
        Enable shell selection, boundary filling and intersection cleaning respectively.

    Returns
    -------
    vertices, faces, info : ndarray, ndarray, dict
        Best mesh, per-stage counts and success for a closed manifold without crossings or degeneracies.
        Malformed arrays raise ValueError; failed geometric stages return success=False.
    """
    vertices, faces = _mesh(vertices, faces)
    info = {}
    try:
        vertices, faces = fix_connectivity(vertices, faces)
        info['connectivity'] = _stage_counts(vertices, faces, logger, 'connectivity')
        if keep_largest:
            vertices, faces = keep_largest_component(vertices, faces)
            info['largest_component'] = _stage_counts(vertices, faces, logger, 'largest_component')
        if fill:
            vertices, faces = fill_holes(vertices, faces)
            info['fill_holes'] = _stage_counts(vertices, faces, logger, 'fill_holes')
        if clean:
            vertices, faces, info['success'] = mesh_clean(vertices, faces)
            info['mesh_clean'] = _stage_counts(vertices, faces, logger, 'mesh_clean')
        else:
            info['success'] = bool(_closed_manifold(vertices, faces) and not _bad_triangles(vertices, faces).any()
                                   and not len(self_intersecting_faces(vertices, faces)))
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
        # Geometry failures retain the last completed stage; array validation above still raises.
        if logger is not None:
            logger.warning('Repair stopped with the best available mesh: %s', error)
        info['success'] = False
    return vertices, faces, info
