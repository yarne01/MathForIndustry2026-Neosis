"""Small synthetic regression tests for evaluation and mesh repair."""

from fractions import Fraction
from pathlib import Path
import sys

import numpy as np
import pytest
import trimesh
from skimage.measure import marching_cubes

MODULE_FOLDER = str(Path(__file__).resolve().parent)
if MODULE_FOLDER not in sys.path:
    sys.path.insert(0, MODULE_FOLDER)
import evaluation_utils as evaluation
import repair_utils as repair


# ---------------------------------------------------------------------------------------------------- Defect meshes
def _sphere():
    mesh = trimesh.creation.icosphere(subdivisions=2)
    return np.array(mesh.vertices), np.array(mesh.faces)


def _defect_meshes():
    vertices, faces = _sphere()
    holes = np.delete(faces, [0, 100], axis=0)
    duplicated = faces.copy()
    duplicated[0, 0] = len(vertices)
    flipped = faces.copy()
    flipped[:7] = flipped[:7, ::-1]
    small = trimesh.creation.icosphere(subdivisions=1, radius=0.1)
    floating_v = np.vstack((vertices, small.vertices + [3., 0., 0.]))
    floating_f = np.vstack((faces, small.faces + len(vertices)))
    crossing_v = np.vstack((vertices, vertices + [0.7, 0.13, 0.07]))
    crossing_f = np.vstack((faces, faces + len(vertices)))
    edge_v = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
                       [0., -1., 0.], [0., 0., 1.]])
    edge_f = np.array([[0, 1, 2], [1, 0, 3], [0, 1, 4]])
    return {
        "holes": (vertices.copy(), holes),
        "duplicate_vertices": (np.vstack((vertices, vertices[faces[0, 0]])), duplicated),
        "flipped_faces": (vertices.copy(), flipped),
        "nonmanifold_edge": (edge_v, edge_f),
        "floating_component": (floating_v, floating_f),
        "intersecting_spheres": (crossing_v, crossing_f),
        "degenerate_faces": (vertices.copy(), np.vstack((faces, [0, 0, 1], faces[0, ::-1]))),
    }


def _mesh(vertices, faces):
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


# ---------------------------------------------------------------------------------------------------- Repair
@pytest.mark.parametrize("case, key", [
    ("holes", "boundary_edges"),
    ("duplicate_vertices", "boundary_edges"),
    ("nonmanifold_edge", "non_manifold_edges"),
    ("floating_component", "components"),
    ("intersecting_spheres", "self_intersecting_faces"),
    ("degenerate_faces", "degenerate_faces"),
])
def test_quality_detects_defects(case, key):
    quality = evaluation.mesh_quality(*_defect_meshes()[case])
    assert quality[key] > (1 if case == "floating_component" else 0), quality


@pytest.mark.parametrize('vertices, zero_area, collinear', [
    ([[0., 0., 0.], [1., 0., 0.], [1., 1.e-14, 0.]], 0, 0),
    ([[0., 0., 0.], [1., 2., 3.], [2., 4., 6.]], 1, 1),
    ([[0., 0., 0.], [0., 0., 0.], [1., 2., 3.]], 1, 1),
    ([[0., 0., 0.], [1.e-170, 0., 0.], [0., 1.e-170, 0.]], 1, 0),
    ([[0., 0., 0.], [1., 1. + 2.**-52, 0.], [1. - 2.**-52, 1., 0.]], 1, 0),
    # Rounded subtraction makes these edges look equal, but the original corners are not collinear.
    ([[1., 0., 0.], [2.**54, 2.**54, 0.], [2.**55, 2.**55, 0.]], 1, 0),
])
@pytest.mark.parametrize('axes', [(0, 1, 2), (1, 2, 0), (2, 0, 1)])
def test_quality_distinguishes_float_zero_area_and_exact_collinearity(vertices, zero_area, collinear, axes):
    import evaluation_plot_utils as plots

    vertices = np.asarray(vertices)[:, axes]
    quality = evaluation.mesh_quality(vertices, np.array([[0, 1, 2]]), check_intersections=False)
    assert quality['zero_area_faces'] == zero_area
    assert quality['collinear_faces'] == quality['degenerate_faces'] == collinear
    quality['self_intersecting_faces'] = 0
    checks = plots.quality_checks(quality)
    assert checks['Nondegenerate triangles'] == (collinear == 0)
    assert checks['No zero-area faces'] == (zero_area == 0)


def test_collinearity_filter_matches_exact_determinants_across_scales():
    rng = np.random.default_rng(42)
    triangles = rng.integers(-16, 17, size=(60, 3, 3)).astype(float)
    triangles[:30, 2] = 3 * triangles[:30, 1] - 2 * triangles[:30, 0]
    triangles = np.ldexp(triangles, np.resize([-1070, -540, -200, 0, 500, 1000], 60)[:, None, None])
    expected = []
    for triangle in triangles:
        a, b, c = [[Fraction(float(value)) for value in corner] for corner in triangle]
        # Independent projected homogeneous determinants, without the production float filter.
        determinants = [a[i] * (b[j] - c[j]) + b[i] * (c[j] - a[j]) + c[i] * (a[j] - b[j])
                        for i, j in ((0, 1), (0, 2), (1, 2))]
        expected.append(all(value == 0 for value in determinants))
    assert any(expected) and not all(expected)
    for triangle, collinear in zip(triangles, expected):
        assert evaluation._count_collinear_faces(triangle, np.array([[0, 1, 2]])) == int(collinear)
    assert evaluation._count_collinear_faces(triangles.reshape(-1, 3), np.arange(180).reshape(-1, 3)) == sum(expected)


def test_merge_exact_duplicate_vertices():
    vertices, faces = _defect_meshes()["duplicate_vertices"]
    new_v, new_f = repair.merge_duplicate_vertices(vertices, faces)
    assert len(new_v) == len(vertices) - 1
    np.testing.assert_array_equal(new_v[new_f], vertices[faces])
    assert _mesh(new_v, new_f).is_watertight


def test_remove_repeated_index_and_duplicate_faces():
    vertices, faces = _defect_meshes()["degenerate_faces"]
    assert evaluation.mesh_quality(vertices, faces)["duplicate_faces"] == 1
    new_v, new_f = repair.remove_degenerate_faces(vertices, faces)
    assert len(new_f) == len(faces) - 2
    assert len(np.unique(np.sort(new_f, axis=1), axis=0)) == len(new_f)
    assert np.all(np.diff(np.sort(new_f, axis=1), axis=1) > 0)


def test_orient_faces_restores_consistency_and_outward_volume():
    vertices, faces = _defect_meshes()["flipped_faces"]
    assert not evaluation.mesh_quality(vertices, faces)["consistently_oriented"]
    for candidate in (faces, faces[:, ::-1]):
        new_f = repair.orient_faces(vertices, candidate)
        mesh = _mesh(vertices, new_f)
        assert mesh.is_winding_consistent
        assert mesh.volume > 0


def test_connectivity_splits_three_face_edge_without_deleting_faces():
    vertices, faces = _defect_meshes()["nonmanifold_edge"]
    new_v, new_f = repair.fix_connectivity(vertices, faces)
    edges = np.sort(new_f[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
    assert np.unique(edges, axis=0, return_counts=True)[1].max() <= 2
    assert len(new_f) == len(faces)
    assert len(np.unique(new_f)) == len(new_v)
    np.testing.assert_array_equal(new_v[new_f].mean(axis=1), vertices[faces].mean(axis=1))


def test_connectivity_splits_a_pinched_vertex():
    vertices, faces = _sphere()
    combined_v = np.vstack((vertices, vertices + [2., 0., 0.]))
    combined_f = np.vstack((faces, faces + len(vertices)))
    unique_v, inverse = np.unique(combined_v, axis=0, return_inverse=True)
    assert len(unique_v) == len(combined_v) - 1
    new_v, new_f = repair.fix_connectivity(unique_v, inverse[combined_f])
    assert len(new_v) == len(combined_v)
    assert len(new_f) == len(combined_f)
    assert _mesh(new_v, new_f).is_watertight


def test_keep_largest_edge_connected_component():
    new_v, new_f = repair.keep_largest_component(*_defect_meshes()["floating_component"])
    vertices, faces = _sphere()
    assert len(new_f) == len(faces)
    assert len(new_v) == len(vertices)
    assert _mesh(new_v, new_f).volume == pytest.approx(_mesh(vertices, faces).volume)


def test_ordered_boundary_loops_and_hole_filling():
    vertices, faces = _defect_meshes()["holes"]
    loops = repair.boundary_loops(faces)
    assert len(loops) == 2
    for loop in loops:
        assert loop.ndim == 1 and len(loop) == 3
        assert len(np.unique(loop)) == len(loop)
        for a, b in zip(loop, np.roll(loop, -1)):
            assert np.count_nonzero(np.any(faces == a, axis=1) & np.any(faces == b, axis=1)) == 1
    _, limited_f = repair.fill_holes(vertices, faces, max_hole_edges=2, refine=False)
    assert len(limited_f) == len(faces)
    new_v, new_f = repair.fill_holes(vertices, faces)
    assert repair.boundary_loops(new_f) == []
    assert _mesh(new_v, new_f).is_watertight


def test_self_intersections_and_mesh_clean():
    vertices, faces = _defect_meshes()["intersecting_spheres"]
    ids = repair.self_intersecting_faces(vertices, faces)
    assert ids.ndim == 1 and len(ids) > 0
    assert np.all((ids >= 0) & (ids < len(faces)))
    new_v, new_f, success = repair.mesh_clean(vertices, faces)
    assert success
    assert len(new_f) > 0
    assert len(repair.self_intersecting_faces(new_v, new_f)) == 0
    assert _mesh(new_v, new_f).is_watertight


def test_repair_pipeline_preserves_sphere_volume():
    original_v, original_f = _sphere()
    vertices, faces = _defect_meshes()["holes"]
    faces[:5] = faces[:5, ::-1]
    vertices = np.vstack((vertices, vertices[faces[0, 0]]))
    faces[0, 0] = len(vertices) - 1
    faces = np.vstack((faces, [0, 0, 1], faces[5, ::-1]))
    new_v, new_f, info = repair.repair_mesh(vertices, faces)
    mesh = _mesh(new_v, new_f)
    assert info["success"]
    stages = [counts for stage, counts in info.items() if stage != "success"]
    assert stages
    for counts in stages:
        assert {"vertices", "faces", "components", "boundary_loops"} <= counts.keys()
    assert mesh.is_watertight and mesh.is_winding_consistent
    assert mesh.volume == pytest.approx(_mesh(original_v, original_f).volume, rel=0.02)
    assert len(repair.self_intersecting_faces(new_v, new_f)) == 0


# ---------------------------------------------------------------------------------------------------- Distances
def test_surface_comparison_identity_and_known_scale_offset():
    vertices, faces = _sphere()
    same = evaluation.compare_surfaces(vertices, faces, vertices, faces, n_points=4000)
    assert same["chamfer_mean"] == pytest.approx(0., abs=1.e-6)
    scaled = evaluation.compare_surfaces(vertices, faces, 1.1 * vertices, faces, n_points=12000)
    # The polygonal sphere approximates concentric unit and radius-1.1 spheres.
    assert scaled["chamfer_mean"] == pytest.approx(0.1, abs=0.01)
    box = trimesh.creation.box(extents=[2., 2., 2.])
    scaled_box = evaluation.compare_surfaces(box.vertices, box.faces, box.vertices * 1.1, box.faces, n_points=1000)
    assert scaled_box["reference_to_mesh_mean"] == pytest.approx(0.1, abs=1.e-10)


def test_surface_sampling_is_deterministic():
    vertices, faces = _sphere()
    points, normals = evaluation.sample_surface(vertices, faces, 100, seed=7)
    again, _ = evaluation.sample_surface(vertices, faces, 100, seed=7)
    assert points.shape == normals.shape == (100, 3)
    np.testing.assert_array_equal(points, again)
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.)


def test_distance_matrix_is_symmetric_with_zero_diagonal():
    vertices, faces = _sphere()
    meshes = [(vertices, faces), (vertices * 1.2, faces), (vertices + [3., 0., 0.], faces)]
    distances = evaluation.mesh_distance_matrix(meshes, n_points=512)
    assert distances.shape == (3, 3)
    np.testing.assert_allclose(distances, distances.T)
    np.testing.assert_array_equal(np.diag(distances), 0.)
    assert np.all(distances[np.triu_indices(3, 1)] > 0)


def test_fscore_curve_known_precision_and_recall():
    result = evaluation.fscore_curve([0., 0.2], [0., 0.1, 0.3], [0.05, 0.25, 1.])
    np.testing.assert_allclose(result["precision"], [0.5, 1., 1.])
    np.testing.assert_allclose(result["recall"], [1. / 3., 2. / 3., 1.])
    np.testing.assert_allclose(result["fscore"], [0.4, 0.8, 1.])


@pytest.mark.parametrize("operation", [
    repair.merge_duplicate_vertices, repair.remove_degenerate_faces, repair.orient_faces,
    repair.fix_connectivity, repair.keep_largest_component, repair.fill_holes,
    repair.self_intersecting_faces, repair.mesh_clean, repair.repair_mesh,
    evaluation.mesh_quality, evaluation.mesh_descriptors, evaluation.curvature, evaluation.evaluate_mesh,
])
def test_operations_do_not_mutate_inputs(operation):
    vertices, faces = _sphere()
    before_v, before_f = vertices.copy(), faces.copy()
    operation(vertices, faces)
    np.testing.assert_array_equal(vertices, before_v)
    np.testing.assert_array_equal(faces, before_f)


def test_comparison_does_not_mutate_either_mesh():
    vertices, faces = _sphere()
    candidate_v, candidate_f = vertices * 1.1, faces.copy()
    originals = [a.copy() for a in (vertices, faces, candidate_v, candidate_f)]
    evaluation.compare_surfaces(vertices, faces, candidate_v, candidate_f, n_points=100)
    for actual, expected in zip((vertices, faces, candidate_v, candidate_f), originals):
        np.testing.assert_array_equal(actual, expected)


def test_real_stl_repair():
    paths = list((Path(__file__).parent / "test_files").rglob("*.stl"))
    if not paths:
        pytest.skip("No STL available in test_files")
    path = min(paths, key=lambda p: (p.stat().st_size, str(p)))
    mesh = trimesh.load_mesh(path, process=False)
    before = evaluation.mesh_quality(mesh.vertices, mesh.faces)
    vertices, faces, info = repair.repair_mesh(mesh.vertices, mesh.faces)
    after = evaluation.mesh_quality(vertices, faces)
    assert before and after
    assert info["success"], (str(path), info)
    assert _mesh(vertices, faces).is_watertight
    assert len(repair.self_intersecting_faces(vertices, faces)) == 0


# ---------------------------------------------------------------------------------------------------- Messy boundaries

def test_boundary_cycles_touching_at_one_vertex_are_split():
    faces = np.array([[0, 1, 2], [0, 3, 4]])
    loops = repair.boundary_loops(faces)
    assert sorted(map(len, loops)) == [3, 3]
    boundary_edges = np.vstack([np.column_stack((loop, np.roll(loop, -1))) for loop in loops])
    expected = faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
    assert set(map(tuple, boundary_edges)) == set(map(tuple, expected))


def test_int32_edge_keys_do_not_overflow_on_large_vertex_indices():
    faces = np.array([[1, 20000, 3], [20000, 1, 5], [61357, 67296, 69999]], dtype=np.int32)
    assert sorted(map(len, repair.boundary_loops(faces))) == [3, 4]


def test_uint64_edge_keys_retain_integer_precision():
    faces = np.arange(6, dtype=np.uint64).reshape(2, 3) + 1000000000
    assert sorted(map(len, repair.boundary_loops(faces))) == [3, 3]


def test_hole_with_collapsed_projection_does_not_raise():
    vertices = np.column_stack((np.arange(5.), np.zeros((5, 2))))
    faces = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4]])
    new_v, new_f = repair.fill_holes(vertices, faces)
    assert np.isfinite(new_v).all()
    assert new_f.shape[1] == 3
    _, _, info = repair.repair_mesh(vertices, faces)
    assert not info['success']


def test_hole_with_every_diagonal_occupied_is_left_open():
    # A seven-vertex torus has every possible edge; remove the fan of vertex zero.
    faces = np.array([[index, (index + 1) % 7, (index + 3) % 7] for index in range(7)]
                     + [[index, (index + 3) % 7, (index + 2) % 7] for index in range(7)])
    faces = faces[~(faces == 0).any(axis=1)]
    vertices = np.random.default_rng(4).normal(size=(7, 3))
    new_v, new_f = repair.fill_holes(vertices, faces, refine=False)
    np.testing.assert_array_equal(new_v, vertices)
    np.testing.assert_array_equal(new_f, faces)
    assert len(repair.boundary_loops(new_f)) == 1


def test_nonorientable_strip_is_cut_into_orientable_sheets():
    angles = np.arange(7) * 2 * np.pi / 7
    vertices = np.array([[(1 + width * np.cos(angle / 2)) * np.cos(angle),
                          (1 + width * np.cos(angle / 2)) * np.sin(angle),
                          width * np.sin(angle / 2)] for angle in angles for width in (-.2, .2)])
    faces = []
    for index in range(7):
        next_low, next_high = (2 * index + 2, 2 * index + 3) if index < 6 else (1, 0)
        faces.extend([[2 * index, next_low, 2 * index + 1], [2 * index + 1, next_low, next_high]])
    faces = np.array(faces)
    new_v, new_f = repair.fix_connectivity(vertices, faces)
    assert len(new_f) == len(faces)
    np.testing.assert_allclose(new_v[new_f].mean(axis=1), vertices[faces].mean(axis=1))
    repair.orient_faces(new_v, new_f)
    _, _, info = repair.repair_mesh(vertices, faces)
    assert isinstance(info['success'], bool)


@pytest.mark.parametrize('seed', [0, 1, 7])
def test_random_triangle_soup_returns_best_mesh(seed):
    rng = np.random.default_rng(seed)
    vertices = rng.normal(size=(45, 3))
    faces = rng.integers(0, len(vertices), size=(80, 3))
    connected_v, connected_f = repair.fix_connectivity(vertices, faces)
    filled_v, filled_f = repair.fill_holes(connected_v, connected_f)
    assert np.isfinite(filled_v).all()
    assert np.all(np.bincount(_mesh(filled_v, filled_f).edges_unique_inverse) <= 2)
    new_v, new_f, info = repair.repair_mesh(vertices, faces, keep_largest=False)
    assert np.isfinite(new_v).all()
    assert new_f.shape[1] == 3
    assert isinstance(info['success'], bool)
    if info['success']:
        assert _mesh(new_v, new_f).is_watertight
        assert not len(repair.self_intersecting_faces(new_v, new_f))


@pytest.mark.parametrize('vertices, faces', [
    (np.zeros((3, 2)), np.array([[0, 1, 2]])),
    (np.zeros((3, 3)), np.array([[0., 1., 2.]])),
    (np.zeros((3, 3)), np.array([[0, 1, 3]])),
    (np.zeros((3, 3)), np.array([[0, -1, 2]])),
    (np.full((3, 3), np.nan), np.array([[0, 1, 2]])),
])
def test_repair_still_rejects_malformed_arrays(vertices, faces):
    with pytest.raises(ValueError):
        repair.repair_mesh(vertices, faces)


def test_geometric_failure_returns_last_completed_stage(monkeypatch):
    vertices, faces = _defect_meshes()['holes']

    def failed_fill(*args):
        raise ValueError('untriangulable geometric boundary')

    monkeypatch.setattr(repair, 'fill_holes', failed_fill)
    new_v, new_f, info = repair.repair_mesh(vertices, faces)
    assert not info['success']
    assert info['largest_component']['faces'] == len(faces) == len(new_f)
    assert len(repair.boundary_loops(new_f)) == 2
    assert not np.shares_memory(new_v, vertices)
    assert not np.shares_memory(new_f, faces)


@pytest.mark.parametrize('case', ['noisy_spheres', 'thin_plate'])
def test_small_marching_cubes_fields(case):
    coordinates = np.linspace(-1., 1., 24)
    grid = np.stack(np.meshgrid(coordinates, coordinates, coordinates, indexing='ij'), axis=-1)
    if case == 'noisy_spheres':
        rng = np.random.default_rng(42)
        field = np.full(grid.shape[:-1], np.inf)
        for center in rng.uniform(-.4, .4, (4, 3)):
            field = np.minimum(field, np.linalg.norm(grid - center, axis=-1) - .4)
        field += rng.normal(scale=.06, size=field.shape)
    else:
        field = np.maximum.reduce([np.abs(grid[..., 2]) - .06, np.abs(grid[..., 0]) - .8,
                                   np.abs(grid[..., 1]) - .8])
        field = np.maximum(field, .3 - np.linalg.norm(grid[..., :2], axis=-1))
    vertices, faces, _, _ = marching_cubes(field, level=0.)
    new_v, new_f, info = repair.repair_mesh(vertices, faces)
    assert np.isfinite(new_v).all()
    assert len(new_f) > 0
    assert info['success']
    assert _mesh(new_v, new_f).is_watertight
    assert not len(repair.self_intersecting_faces(new_v, new_f))


def test_empty_mesh_is_an_unsuccessful_best_mesh():
    vertices, faces, info = repair.repair_mesh(np.empty((0, 3)), np.empty((0, 3), dtype=int))
    assert vertices.shape == faces.shape == (0, 3)
    assert not info['success']


def test_plot_quality_checks_and_pass_rates(tmp_path):
    import evaluation_plot_utils as plots

    vertices, faces = _sphere()
    checks = plots.quality_checks(evaluation.mesh_quality(vertices, faces))
    assert len(checks) == 8 and all(checks.values())
    opened = plots.quality_checks(evaluation.mesh_quality(vertices, faces[1:]))
    assert not opened['Closed']
    with pytest.raises(ValueError, match='intersection'):
        plots.quality_checks(evaluation.mesh_quality(vertices, faces, check_intersections=False))
    path = tmp_path / 'rates.png'
    plots.plot_pass_rates([(0, 1)], [(1, 1)], ['Sphere'], path)
    assert path.read_bytes().startswith(b'\x89PNG')
    with pytest.raises(ValueError, match='denominators'):
        plots.plot_pass_rates([(0, 1)], [(1, 2)], ['Sphere'], path)


def test_plot_check_counts_preserves_unknown(tmp_path):
    import matplotlib.pyplot as plt
    import evaluation_plot_utils as plots

    path = tmp_path / 'checks.png'
    plots.plot_check_matrix([{'Closed': 0, 'No self-intersection': np.nan},
                            {'Closed': 2, 'No self-intersection': 2}], ['Before n=2', 'After n=2'], path)
    assert path.read_bytes().startswith(b'\x89PNG')
    assert not plt.get_fignums()
    with pytest.raises(ValueError, match='same ordered'):
        plots.plot_check_matrix([{'Closed': 1}, {'Nonempty': 1}], ['A', 'B'], path)


def test_plot_mesh_stages_preserves_arrays(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    import evaluation_plot_utils as plots

    limits = []
    original_close = plt.close

    def capture_limits(figure):
        if hasattr(figure, 'axes'):
            limits.extend([(ax.get_xlim(), ax.get_ylim(), ax.get_zlim()) for ax in figure.axes])
        original_close(figure)

    monkeypatch.setattr(plt, 'close', capture_limits)
    vertices, faces = _sphere()
    saved_v, saved_f = vertices.copy(), faces.copy()
    path = tmp_path / 'meshes.png'
    plots.plot_mesh_stages([(vertices, faces[1:]), (vertices, faces)], ['Open', 'Closed'], path)
    assert path.read_bytes().startswith(b'\x89PNG')
    np.testing.assert_allclose(limits[0], limits[1])
    np.testing.assert_array_equal(vertices, saved_v)
    np.testing.assert_array_equal(faces, saved_f)
    with pytest.raises(ValueError, match='indices'):
        plots.plot_mesh_stages([(vertices, faces + len(vertices))], ['Invalid'], path)


# ---------------------------------------------------------------------------------------------------- Local repair

def test_clean_preserves_disconnected_valid_geometry_after_degeneracy_cleanup():
    first = trimesh.creation.icosphere(subdivisions=1)
    second = trimesh.creation.icosphere(subdivisions=1, radius=.5)
    vertices = np.vstack((first.vertices, second.vertices + [4., 0., 0.],
                          [[8., 0., 0.], [9., 0., 0.], [10., 0., 0.]]))
    faces = np.vstack((first.faces, second.faces + len(first.vertices),
                       np.arange(len(vertices) - 3, len(vertices))))
    new_v, new_f, success = repair.mesh_clean(vertices, faces)
    assert success
    assert repair._components(new_f)[0] == 2
    assert len(new_f) == len(first.faces) + len(second.faces)
    np.testing.assert_allclose(new_v.max(axis=0), [4.5, 1., 1.])


def test_hole_fairing_keeps_planar_cap_inside_boundary_bounds():
    cylinder = trimesh.creation.cylinder(radius=1., height=.1, sections=32)
    vertices = np.array(cylinder.vertices)
    faces = np.array(cylinder.faces[cylinder.face_normals[:, 2] < .9])
    new_v, new_f = repair.fill_holes(vertices, faces)
    assert len(new_v) > len(vertices)
    np.testing.assert_array_equal(new_v[:len(vertices)], vertices)
    np.testing.assert_allclose(new_v[len(vertices):, 2], .05, atol=1.e-12)
    assert _mesh(new_v, new_f).is_watertight
    assert not len(repair.self_intersecting_faces(new_v, new_f))


@pytest.mark.parametrize('selected', [[0], [1], [0, 1], []])
def test_local_intersection_query_checks_both_large_and_small_faces(selected):
    vertices = np.array([[-10., -10., 0.], [10., -10., 0.], [0., 10., 0.],
                         [-.1, 0., -.1], [.1, 0., -.1], [0., 0., .1]])
    vertices = np.vstack((vertices, vertices + [100., 0., 0.]))
    faces = np.arange(12).reshape(-1, 3)
    expected = [0, 1] if selected else []
    np.testing.assert_array_equal(repair._intersecting_faces(vertices, faces, selected), expected)
    np.testing.assert_array_equal(repair.self_intersecting_faces(vertices, faces), [0, 1, 2, 3])


def test_intersection_cleaning_preserves_a_remote_valid_component():
    vertices, faces = _defect_meshes()['intersecting_spheres']
    remote = trimesh.creation.icosphere(subdivisions=1, radius=.5)
    remote_vertices = np.array(remote.vertices) + [5., 0., 0.]
    combined_v = np.vstack((vertices, remote_vertices))
    combined_f = np.vstack((faces, np.array(remote.faces) + len(vertices)))
    new_v, new_f, success = repair.mesh_clean(combined_v, combined_f)
    assert success
    actual = new_v[new_f[np.all(new_v[new_f, 0] > 4., axis=1)]]
    assert len(actual) == len(remote.faces)
    expected = remote_vertices[remote.faces]
    np.testing.assert_allclose(np.sort(actual.reshape(-1, 3), axis=0), np.sort(expected.reshape(-1, 3), axis=0))
    assert not len(repair.self_intersecting_faces(new_v, new_f))


def test_clean_keeps_a_closed_component_below_the_debris_size_cutoff():
    large = trimesh.creation.icosphere(subdivisions=4)
    tiny_v = np.array([[4., 0., 0.], [4.5, 0., 0.], [4., .5, 0.], [4., 0., .5]])
    tiny_f = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    vertices = np.vstack((large.vertices, tiny_v, [[8., 0., 0.], [9., 0., 0.], [10., 0., 0.]]))
    faces = np.vstack((large.faces, tiny_f + len(large.vertices), np.arange(len(vertices) - 3, len(vertices))))
    new_v, new_f, success = repair.mesh_clean(vertices, faces)
    assert success
    assert len(new_f) == len(large.faces) + len(tiny_f)
    assert repair._components(new_f)[0] == 2
    assert new_v[:, 0].max() == 4.5


def test_repair_and_report_writes_json_for_reconstructed_mesh(tmp_path):
    import json
    import logging
    import open3d as o3d
    from skimage.measure import marching_cubes
    coordinates = np.linspace(-1., 1., 24)
    grid = np.stack(np.meshgrid(coordinates, coordinates, coordinates, indexing='ij'), axis=-1)
    field = np.linalg.norm(grid, axis=-1) - .6
    vertices, faces, _, _ = marching_cubes(field, level=0.)
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(faces))
    repaired = evaluation.repair_and_report(logging.getLogger('test'), mesh, 'reconstruction_0', str(tmp_path))
    report = json.loads((tmp_path / 'reconstruction_0_repair.json').read_text())
    assert set(report) == {'before', 'repair', 'after'}
    assert report['repair']['success'] and report['after']['watertight']
    assert len(np.asarray(repaired.triangles)) > 0
    assert len(np.asarray(mesh.triangles)) == len(faces)
