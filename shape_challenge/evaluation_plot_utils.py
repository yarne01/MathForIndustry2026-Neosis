import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------------------------------- Quality checks
def quality_checks(quality):
    """Map mesh_quality to the eight checks in the 2026-09-11 presentation manifest.

    Parameters
    ----------
    quality : dict
        mesh_quality output with intersection testing enabled.

    Returns
    -------
    checks : dict
        Boolean checks in historical order. Nondegenerate triangles uses exact
        collinearity; no zero-area faces uses float area == 0, matching the original
        definitions. Intersections use floating predicates, not CGAL. Orientability is
        inferred from defined genus or consistent winding; an inconsistently wound
        open orientable mesh can fail this conservative proxy. Closed means no
        boundary edges, as originally; manifoldness is checked separately.
        Neither single-component nor consistent-winding is an extra gate.
    """
    try:
        names = ('collinear_faces', 'zero_area_faces', 'self_intersecting_faces', 'boundary_edges',
                 'non_manifold_edges', 'n_vertices', 'n_faces', 'non_manifold_vertices')
        counts = [quality[name] for name in names]
        if any(value is None or not np.isscalar(value) or not np.isfinite(value)
               or value < 0 or int(value) != value for value in counts):
            raise ValueError('quality counts must be finite nonnegative integers; run intersection testing')
        if not isinstance(quality['consistently_oriented'], (bool, np.bool_)):
            raise ValueError('consistently_oriented must be boolean')
        orientable = quality['genus'] is not None or quality['consistently_oriented']
    except (KeyError, TypeError) as error:
        raise ValueError('quality must contain the complete mesh_quality output') from error
    return {'Nondegenerate triangles': bool(quality['collinear_faces'] == 0),
            'No self-intersection': bool(quality['self_intersecting_faces'] == 0),
            'Closed': bool(quality['boundary_edges'] == 0),
            'Edge manifold': bool(quality['non_manifold_edges'] == 0),
            'No zero-area faces': bool(quality['zero_area_faces'] == 0),
            'Nonempty': bool(quality['n_vertices'] > 0 and quality['n_faces'] > 0),
            'Orientable': bool(orientable),
            'Vertex manifold': bool(quality['non_manifold_vertices'] == 0)}


# ---------------------------------------------------------------------------------------------------- Pass rates
def plot_pass_rates(before, after, case_names, path):
    """Save grouped before/after pass rates with explicit denominators.

    Parameters
    ----------
    before, after : sequence of (passed, total) pairs
        Counts for each case, with matching positive totals.
    case_names : sequence of str
    path : str or Path

    Returns
    -------
    None
    """
    before, after = np.asarray(before, dtype=float), np.asarray(after, dtype=float)
    shape = (len(case_names), 2)
    if not len(case_names) or before.shape != shape or after.shape != shape:
        raise ValueError('before and after must contain one (passed, total) pair per case')
    for counts in (before, after):
        if (not np.isfinite(counts).all() or np.any(counts != np.floor(counts))
                or np.any(counts[:, 1] <= 0) or np.any(counts[:, 0] < 0)
                or np.any(counts[:, 0] > counts[:, 1])):
            raise ValueError('pass counts must be integers between zero and a positive total')
    if np.any(before[:, 1] != after[:, 1]):
        raise ValueError('before and after must use matching denominators')
    fig, ax = plt.subplots(figsize=(8, 4.5), layout='constrained')
    x = np.arange(len(case_names))
    for counts, shift, color, label in ((before, -.18, '#BAC3CC', 'Before repair'),
                                       (after, .18, '#147BB5', 'After repair')):
        bars = ax.bar(x + shift, 100 * counts[:, 0] / counts[:, 1], width=.34, color=color, label=label)
        ax.bar_label(bars, labels=[f'{int(p)}/{int(n)}' for p, n in counts], padding=3, fontsize=9)
    ax.set_xticks(x, case_names)
    ax.set(ylim=(0, 118), ylabel='Pass all eight checks (%)', title='Mesh repair across saved cases')
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_axisbelow(True)
    ax.grid(axis='y', color='#E8EDF1')
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(loc='upper center', bbox_to_anchor=(.5, 1.02), ncol=2, frameon=False)
    try:
        fig.savefig(path, dpi=180)
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------------------------------- Check matrix
def plot_check_matrix(rows, row_labels, path):
    """Save a heatmap of per-check pass counts; NaN cells are labelled UNKNOWN.

    Parameters
    ----------
    rows : sequence of dict
        Identically ordered check names mapped to nonnegative integer counts or NaN.
    row_labels : sequence of str
        Include each row's denominator in its label.
    path : str or Path

    Returns
    -------
    None
    """
    if not rows or len(rows) != len(row_labels) or not rows[0]:
        raise ValueError('nonempty rows and matching row_labels are required')
    names = list(rows[0])
    if any(list(row) != names for row in rows):
        raise ValueError('all rows must contain the same ordered checks')
    values = np.asarray([list(row.values()) for row in rows], dtype=float)
    known = values[~np.isnan(values)]
    if not np.isfinite(known).all() or np.any(known < 0) or np.any(known != np.floor(known)):
        raise ValueError('check counts must be nonnegative integers or NaN')
    fig, ax = plt.subplots(figsize=(max(8, 1.4 * len(names)), max(3, .42 * len(rows) + 2)), layout='constrained')
    cmap = plt.get_cmap('Blues').copy()
    cmap.set_bad('#dedede')
    maximum = max(1, known.max()) if len(known) else 1
    ax.imshow(values, cmap=cmap, vmin=0, vmax=maximum, aspect='auto')
    for (i, j), value in np.ndenumerate(values):
        ax.text(j, i, 'UNKNOWN' if np.isnan(value) else str(int(value)), ha='center', va='center',
                fontsize=9, color='white' if value > .6 * maximum else '#111827')
    ax.set_xticks(range(len(names)), [name.replace(' ', '\n') for name in names])
    ax.set_yticks(range(len(rows)), row_labels)
    ax.set_title('Structural checks: number passing')
    try:
        fig.savefig(path, dpi=180)
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------------------------------- Mesh stages
def plot_mesh_stages(meshes, titles, path):
    """Render meshes with common physical bounds and an orthographic camera.

    Parameters
    ----------
    meshes : sequence of (vertices, faces) pairs
    titles : sequence of str
    path : str or Path

    Returns
    -------
    None
    """
    if not meshes or len(meshes) != len(titles):
        raise ValueError('nonempty meshes and matching titles are required')
    arrays = [(np.asarray(v), np.asarray(f)) for v, f in meshes]
    for v, f in arrays:
        if (v.ndim != 2 or v.shape[1] != 3 or not len(v) or v.dtype.kind not in 'fiu'
                or f.ndim != 2 or f.shape[1] != 3 or not len(f) or f.dtype.kind not in 'iu'):
            raise ValueError('meshes must contain real (n, 3) vertices and integer (m, 3) faces')
        if not np.isfinite(v).all() or np.any(f < 0) or np.any(f >= len(v)):
            raise ValueError('mesh coordinates must be finite and face indices in range')
    low, high = np.min([v.min(axis=0) for v, _ in arrays], axis=0), np.max([v.max(axis=0) for v, _ in arrays], axis=0)
    span = np.maximum(high - low, max(float(np.max(high - low)) * .01, 1e-12))
    center = (low + high) / 2
    columns = 3 if len(arrays) % 3 == 0 else min(4, len(arrays))
    fig = plt.figure(figsize=(3.6 * columns, 3.7 * int(np.ceil(len(arrays) / columns))), layout='constrained')
    for index, ((v, f), title) in enumerate(zip(arrays, titles)):
        ax = fig.add_subplot(int(np.ceil(len(arrays) / columns)), columns, index + 1, projection='3d')
        ax.plot_trisurf(v[:, 0], v[:, 1], v[:, 2], triangles=f, color='#9dc7d3', linewidth=.08,
                       edgecolor='#3b6270', shade=True)
        ax.set(xlim=center[0] + np.array([-.5, .5]) * span[0],
               ylim=center[1] + np.array([-.5, .5]) * span[1],
               zlim=center[2] + np.array([-.5, .5]) * span[2], title=title)
        ax.set_box_aspect(span.copy())
        ax.view_init(elev=65, azim=-90)
        ax.set_proj_type('ortho')
        ax.set_axis_off()
    try:
        fig.savefig(path, dpi=180)
    finally:
        plt.close(fig)
