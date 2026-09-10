"""UMAP visualization helpers with latent-space k-nearest-neighbor links.

Pipeline usage
--------------
Once the PCA latent vectors are available, this module can be used to explore
the latent space and generate synthetic samples. The main workflow is

        


The entry point is ``plot_training_latent_umap_with_clustering``.
Its main parameters are:

        - ``training_latent``: array with shape ``(n_samples, n_features)``.
        - ``clustering_method``: ``"mst"`` or ``"hdbscan"``.
        - ``evaluate_*`` and ``make_plot_*`` booleans enable or disable workflow
                    stages such as reconstruction, distance analysis, SOM, UMAP, and sampling.
        - ``compare_clusters`` enables both MST and HDBSCAN center comparisons.
        - ``cluster_center_neighbors`` sets the number of original sample meshes
                    selected for each cluster center.
        - ``n_neighbors``: latent-space kNN size for MST or HDBSCAN.
        - ``umap_neighbors`` and ``min_dist``: UMAP embedding settings.
        - ``metric``: distance metric used by UMAP and clustering.
        - ``pruning_method``: ``None``, ``"threshold"``,
            ``"shortest_percent"``, or ``"iqr"`` for MST pruning.
        - ``pruning_value``: threshold or percentage for the first two methods.
        - ``iqr_multiplier``: upper-fence multiplier for ``"iqr"`` pruning.
        - ``compare_clusters``: enables both comparison methods.
        - ``compare_mst`` and ``compare_hdbscan``: enable methods individually.
        - ``center_n_neighbors``: number of original samples selected per center.
        - ``selected_center``: optional center index to inspect exclusively.
        - ``source_meshes``: original meshes aligned with ``training_latent``.
        - ``show_meshes``: displays the selected meshes in a colored 3D grid.
        - ``seed``: random seed for reproducible UMAP results.
        - ``case`` and ``save_folder``: output naming and save location.
        - ``show``: displays figures when ``True``; figures are still saved when
            ``save_folder`` is provided.
        - ``selected_cluster_center`` selects one center by index, or ``None``
            compares every center.

When comparisons are enabled, the workflow can save the final UMAP PNG and
interactive MST/HDBSCAN Plotly HTML files in ``save_folder``.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import umap
from matplotlib.lines import Line2D
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
from sklearn.cluster import HDBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


STANDARD_SEED = 42



def compute_mst_edges(latent, n_neighbors=3, metric="euclidean"):
    """Return the MST edges of a kNN graph in latent space.

    Each returned row contains ``(sample_a, sample_b, distance, component)``.
    The graph is first restricted to k-nearest-neighbor edges, then a minimum
    spanning tree is computed for each connected component.
    """
    latent = np.asarray(latent, dtype=float)
    if latent.ndim != 2:
        raise ValueError("latent must be a two-dimensional array")
    if latent.shape[0] < 2:
        raise ValueError("latent must contain at least two samples")
    if not 1 <= n_neighbors < latent.shape[0]:
        raise ValueError("n_neighbors must be between 1 and n_samples - 1")

    nearest_neighbors = NearestNeighbors(
        n_neighbors=n_neighbors + 1,
        metric=metric,
    )
    nearest_neighbors.fit(latent)
    distances, indices = nearest_neighbors.kneighbors(latent)

    source = np.repeat(np.arange(latent.shape[0]), n_neighbors)
    target = indices[:, 1:].ravel()
    weights = distances[:, 1:].ravel()

    # Add reverse edges so the kNN graph is treated as undirected.
    graph = csr_matrix((weights, (source, target)), shape=(latent.shape[0], latent.shape[0]))
    graph = graph.maximum(graph.T)
    tree = minimum_spanning_tree(graph)
    n_components, component_labels = connected_components(tree, directed=False)

    rows, columns = tree.nonzero()
    edge_components = component_labels[rows]
    return np.column_stack((rows, columns, np.asarray(tree[rows, columns]).ravel(), edge_components))


def compute_full_knn_graph(latent, n_neighbors=None, metric="euclidean"):
    """Return the full symmetric kNN graph and directed neighbor arrays.

    Unlike :func:`compute_mst_edges`, this keeps every requested kNN edge.
    The return value is ``(graph, distances, indices)`` where ``graph`` is a
    symmetric CSR matrix and the other arrays exclude each sample's self
    neighbor.
    """
    latent = np.asarray(latent, dtype=float)
    if latent.ndim != 2 or latent.shape[0] < 2:
        raise ValueError("latent must be a two-dimensional array with at least two samples")
    if n_neighbors is None:
        n_neighbors = latent.shape[0] - 1
    if not 1 <= n_neighbors < latent.shape[0]:
        raise ValueError("n_neighbors must be between 1 and n_samples - 1")

    nearest_neighbors = NearestNeighbors(n_neighbors=n_neighbors + 1, metric=metric)
    nearest_neighbors.fit(latent)
    distances, indices = nearest_neighbors.kneighbors(latent)
    source = np.repeat(np.arange(latent.shape[0]), n_neighbors)
    graph = csr_matrix(
        (distances[:, 1:].ravel(), (source, indices[:, 1:].ravel())),
        shape=(latent.shape[0], latent.shape[0]),
    )
    return graph.maximum(graph.T), distances[:, 1:], indices[:, 1:]


def show_full_knn_graph(latent, n_neighbors):
    """Display the full symmetric kNN graph in a two-dimensional UMAP view."""
    latent = np.asarray(latent, dtype=float)
    if latent.ndim != 2 or latent.shape[0] < 2:
        raise ValueError("latent must be a two-dimensional array with at least two samples")
    if not 1 <= n_neighbors < latent.shape[0]:
        raise ValueError("n_neighbors must be between 1 and n_samples - 1")

    scaler = StandardScaler()
    latent_scaled = scaler.fit_transform(latent)
    embedding = umap.UMAP(
        n_neighbors=min(n_neighbors, latent.shape[0] - 1),
        n_components=2,
        random_state=STANDARD_SEED,
    ).fit_transform(latent_scaled)
    graph, _, _ = compute_full_knn_graph(latent, n_neighbors=n_neighbors)

    figure, axes = plt.subplots(figsize=(9, 8))
    rows, columns = graph.nonzero()
    for sample_a, sample_b in zip(rows, columns):
        if sample_a < sample_b:
            axes.plot(
                embedding[[sample_a, sample_b], 0],
                embedding[[sample_a, sample_b], 1],
                color="gray",
                linewidth=0.8,
                alpha=0.35,
                zorder=1,
            )

    axes.scatter(
        embedding[:, 0],
        embedding[:, 1],
        color="tab:blue",
        edgecolors="black",
        s=55,
        zorder=2,
    )
    axes.set_title(f"Full kNN graph in UMAP space (k={n_neighbors})")
    axes.set_xlabel("UMAP 1")
    axes.set_ylabel("UMAP 2")
    axes.grid(True, alpha=0.3)
    figure.tight_layout()
    plt.show()
    return figure, axes


def order_mst_trees_by_size(mst_edges, n_samples):
    """Return MST forest trees ordered from fewest to most vertices."""
    mst_edges = np.asarray(mst_edges, dtype=float)
    if mst_edges.ndim != 2 or mst_edges.shape[1] != 4:
        raise ValueError("mst_edges must have shape (n_edges, 4)")
    if n_samples < 1:
        raise ValueError("n_samples must be positive")

    if len(mst_edges):
        rows = mst_edges[:, 0].astype(int)
        columns = mst_edges[:, 1].astype(int)
        graph = csr_matrix(
            (np.ones(2 * len(mst_edges)), (np.r_[rows, columns], np.r_[columns, rows])),
            shape=(n_samples, n_samples),
        )
    else:
        graph = csr_matrix((n_samples, n_samples))
    component_count, node_components = connected_components(graph, directed=False)

    trees = []
    for component_id in range(component_count):
        vertices = np.flatnonzero(node_components == component_id)
        edge_mask = np.isin(mst_edges[:, 0].astype(int), vertices) if len(mst_edges) else np.zeros(0, dtype=bool)
        trees.append({
            "component": component_id,
            "vertices": vertices,
            "edges": mst_edges[edge_mask],
            "n_vertices": len(vertices),
        })
    return sorted(trees, key=lambda tree: (tree["n_vertices"], tree["component"]))


def generate_leaf_bridge_samples(
    latent,
    mst_edges,
    n_samples=1,
    n_external_neighbors=1,
    interpolation_values=(0.1, 0.25, 0.5, 0.75, 0.9),
    metric="euclidean",
):
    """Interpolate leaf nodes in the smallest MST trees to external neighbors.

    For each selected leaf, external nearest neighbors are searched in the
    full kNN graph, excluding every vertex in that leaf's tree. Each returned
    sample is ``t * leaf + (1 - t) * external_neighbor``.
    """
    latent = np.asarray(latent, dtype=float)
    if latent.ndim != 2 or latent.shape[0] < 2:
        raise ValueError("latent must be a two-dimensional array with at least two samples")
    if n_samples < 1 or n_external_neighbors < 1:
        raise ValueError("n_samples and n_external_neighbors must be positive")
    if np.isscalar(interpolation_values):
        interpolation_values = (float(interpolation_values),)
    else:
        interpolation_values = tuple(float(value) for value in interpolation_values)
    if not interpolation_values or any(not 0 <= value <= 1 for value in interpolation_values):
        raise ValueError("interpolation_values must contain values between 0 and 1")

    trees = order_mst_trees_by_size(mst_edges, len(latent))[:n_samples]
    _, distances, indices = compute_full_knn_graph(latent, metric=metric)
    synthetic_samples = []
    records = []
    for tree in trees:
        tree_vertices = set(tree["vertices"].tolist())
        tree_edge_rows = np.asarray(tree["edges"][:, :2], dtype=int) if len(tree["edges"]) else np.empty((0, 2), dtype=int)
        degrees = np.bincount(tree_edge_rows.ravel(), minlength=len(latent))
        leaves = tree["vertices"][degrees[tree["vertices"]] <= 1]
        for leaf in leaves:
            external_neighbors = [
                neighbor
                for neighbor in indices[leaf]
                if int(neighbor) not in tree_vertices
            ][:n_external_neighbors]
            for neighbor in external_neighbors:
                for interpolation_value in interpolation_values:
                    synthetic_samples.append(
                        (1- interpolation_value) * latent[leaf]
                        + interpolation_value * latent[neighbor]
                    )
                    records.append({
                        "tree_component": tree["component"],
                        "tree_size": tree["n_vertices"],
                        "leaf_index": int(leaf),
                        "external_neighbor_index": int(neighbor),
                        "external_neighbor_distance": float(
                            distances[leaf, np.flatnonzero(indices[leaf] == neighbor)[0]]
                        ),
                        "t": interpolation_value,
                    })
    return np.asarray(synthetic_samples, dtype=float), records


def reconstruct_synthetic_meshes(
    synthetic_samples,
    pca,
    logger,
    n_grids_x,
    n_grids_y,
    n_grids_z,
    grid_size,
    make_plot=False,
    title_prefix="Synthetic sample",
    sample_titles=None,
):
    """Reconstruct Open3D meshes from synthetic PCA latent samples."""
    from sdf_pca_utils import reconstruct_mesh

    synthetic_samples = np.asarray(synthetic_samples, dtype=float)
    if synthetic_samples.size == 0:
        return []
    if synthetic_samples.ndim != 2:
        raise ValueError("synthetic_samples must be a two-dimensional array")
    if sample_titles is not None and len(sample_titles) != len(synthetic_samples):
        raise ValueError("sample_titles must have one entry per synthetic sample")
    return [
        reconstruct_mesh(
            logger,
            pca,
            sample.reshape(1, -1),
            None,
            n_grids_x,
            n_grids_y,
            n_grids_z,
            None,
            None,
            grid_size,
            make_plot_=make_plot,
            title_=(sample_titles[index] if sample_titles is not None else f"{title_prefix} {index}"),
        )
        for index, sample in enumerate(synthetic_samples)
    ]


def show_synthetic_mesh_between_samples(
    source_mesh,
    synthetic_mesh,
    target_mesh,
    title,
    spacing_factor=1.5,
    save_folder=None,
):
    """Display and optionally save source, synthetic, and target meshes."""
    import copy
    import open3d as o3d

    meshes = [copy.deepcopy(mesh) for mesh in (source_mesh, synthetic_mesh, target_mesh)]
    source_extent = np.asarray(meshes[0].get_axis_aligned_bounding_box().get_extent())
    spacing = float(np.max(source_extent) * spacing_factor)
    colors = ([0.2, 0.4, 0.9], [0.9, 0.2, 0.2], [0.2, 0.7, 0.3])

    for mesh_index, mesh in enumerate(meshes):
        center = mesh.get_axis_aligned_bounding_box().get_center()
        mesh.translate(np.array([mesh_index * spacing, 0.0, 0.0]) - center)
        mesh.paint_uniform_color(colors[mesh_index])

    o3d.visualization.draw_geometries(
        meshes,
        window_name=title,
        mesh_show_wireframe=True,
    )

    if save_folder is not None:
        import plot_utils

        reference_folder = os.path.join(save_folder, "synthetic meshes with reference")
        os.makedirs(reference_folder, exist_ok=True)
        plot_utils.save_geometries_image(
            meshes,
            os.path.join(reference_folder, f"{title}.png"),
            window_title=title,
            axis="z",
        )


def generate_synthetic_meshes_from_mst(
    latent,
    pca,
    logger,
    n_grids_x,
    n_grids_y,
    n_grids_z,
    grid_size,
    n_neighbors=3,
    metric="euclidean",
    pruning_method=None,
    pruning_value=None,
    iqr_multiplier=1.5,
    n_trees=1,
    n_external_neighbors=1,
    interpolation_values=(0.1, 0.25, 0.5, 0.75, 0.9),
    make_plot=False,
    title_prefix="Synthetic sample",
    save_folder=None,
    prefix="",
    show=True,
    sample_labels=None,
    source_meshes=None,
    show_interpolation_meshes=False,
):
    """Build an MST, bridge its smallest trees, and reconstruct their meshes.

    The returned dictionary retains each pipeline stage so the selected trees,
    bridge metadata, and synthetic latent vectors can be inspected before or
    alongside the reconstructed Open3D meshes.
    """
    latent = np.asarray(latent, dtype=float)
    if latent.ndim != 2 or latent.shape[0] < 2:
        raise ValueError("latent must be a two-dimensional array with at least two samples")
    mst_edges = compute_mst_edges(latent, n_neighbors=n_neighbors, metric=metric)
    pruned_edges = prune_mst_edges(
        mst_edges,
        method=pruning_method,
        value=pruning_value,
        iqr_multiplier=iqr_multiplier,
    )
    ordered_trees = order_mst_trees_by_size(pruned_edges, len(latent))
    synthetic_samples, records = generate_leaf_bridge_samples(
        latent,
        pruned_edges,
        n_samples=n_trees,
        n_external_neighbors=n_external_neighbors,
        interpolation_values=interpolation_values,
        metric=metric,
    )
    if synthetic_samples.size == 0:
        synthetic_samples = np.empty((0, latent.shape[1]), dtype=float)
    if sample_labels is None:
        sample_labels = [str(index) for index in range(len(latent))]
    if len(sample_labels) != len(latent):
        raise ValueError("sample_labels must have one entry per latent sample")
    sample_labels = [
        label if str(label).lower().startswith("exp_") else f"exp_{label}"
        for label in sample_labels
    ]
    if source_meshes is not None and len(source_meshes) != len(latent):
        raise ValueError("source_meshes must have one entry per latent sample")
    sample_titles = [
        f"{sample_labels[record['leaf_index']]}_{sample_labels[record['external_neighbor_index']]}_{record['t']}"
        for record in records
    ]
    pruned_component_edges, component_count, node_components = relabel_mst_components(
        pruned_edges,
        len(latent),
    )
    meshes = reconstruct_synthetic_meshes(
        synthetic_samples,
        pca=pca,
        logger=logger,
        n_grids_x=n_grids_x,
        n_grids_y=n_grids_y,
        n_grids_z=n_grids_z,
        grid_size=grid_size,
        make_plot=make_plot and not show_interpolation_meshes,
        title_prefix=title_prefix,
        sample_titles=sample_titles,
    )

    if save_folder is not None and meshes:
        import plot_utils

        synthetic_mesh_folder = os.path.join(save_folder, "synthetic meshes")
        os.makedirs(synthetic_mesh_folder, exist_ok=True)
        for mesh, sample_title in zip(meshes, sample_titles):
            base_path = os.path.join(synthetic_mesh_folder, f"{sample_title}.png")
            for axis in ("x", "y", "z"):
                plot_utils.save_geometries_image(
                    [mesh],
                    base_path,
                    window_title=sample_title,
                    axis=axis,
                )

    if show_interpolation_meshes and source_meshes is not None:
        for mesh, record, sample_title in zip(meshes, records, sample_titles):
            show_synthetic_mesh_between_samples(
                source_meshes[record["leaf_index"]],
                mesh,
                source_meshes[record["external_neighbor_index"]],
                title=sample_title,
                save_folder=save_folder,
            )

    scaler = StandardScaler()
    latent_scaled = scaler.fit_transform(latent)
    reducer = umap.UMAP(
        n_neighbors=min(max(2, n_neighbors), len(latent) - 1),
        n_components=2,
        random_state=STANDARD_SEED,
    )
    embedding = reducer.fit_transform(latent_scaled)
    synthetic_embedding = (
        reducer.transform(scaler.transform(synthetic_samples))
        if len(synthetic_samples)
        else np.empty((0, 2), dtype=float)
    )

    figure, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=True, sharey=True)
    component_colors = plt.get_cmap("tab10")
    for sample_a, sample_b, _, component in pruned_component_edges:
        sample_a = int(sample_a)
        sample_b = int(sample_b)
        edge_color = component_colors(int(component) % component_colors.N)
        for axis in axes:
            axis.plot(
                embedding[[sample_a, sample_b], 0],
                embedding[[sample_a, sample_b], 1],
                color=edge_color,
                linewidth=1.4,
                alpha=0.75,
                zorder=1,
            )

    for component_id in range(component_count):
        sample_mask = node_components == component_id
        for axis in axes:
            axis.scatter(
                embedding[sample_mask, 0],
                embedding[sample_mask, 1],
                color=component_colors(component_id % component_colors.N),
                edgecolors="black",
                linewidths=0.5,
                s=48,
                zorder=3,
                label=f"Component {component_id}",
            )

    interpolation_paths = {}
    for synthetic_index, record in enumerate(records):
        path_key = (record["leaf_index"], record["external_neighbor_index"])
        interpolation_paths.setdefault(path_key, []).append(
            (record["t"], synthetic_embedding[synthetic_index])
        )

    for (leaf, neighbor), path_points in interpolation_paths.items():
        path_points.sort(key=lambda point: point[0])
        path_embedding = np.vstack(
            [embedding[leaf], *(point for _, point in path_points), embedding[neighbor]]
        )
        axes[1].plot(
            path_embedding[:, 0],
            path_embedding[:, 1],
            color="tab:orange",
            linestyle=":",
            linewidth=1.2,
            alpha=0.65,
            zorder=2,
        )

    axes[0].set_title("Pruned MST")
    axes[1].set_title("Pruned MST with synthetic bridges")
    axes[1].scatter(
        embedding[:, 0],
        embedding[:, 1],
        color="none",
        edgecolors="black",
        s=20,
        alpha=0.5,
        zorder=4,
    )
    if len(synthetic_embedding):
        axes[1].scatter(
            synthetic_embedding[:, 0],
            synthetic_embedding[:, 1],
            color="tab:red",
            marker="*",
            s=45,
            alpha=0.7,
            edgecolors="black",
            label="Synthetic samples",
            zorder=4,
        )
    axes[0].set_xlabel("UMAP 1")
    axes[1].set_xlabel("UMAP 1")
    axes[0].set_ylabel("UMAP 2")
    axes[1].legend(loc="best")
    for axis in axes:
        axis.grid(True, alpha=0.3)
    figure.tight_layout()

    if save_folder is not None:
        os.makedirs(save_folder, exist_ok=True)
        filename = f"{prefix}_synthetic_k={n_neighbors}_{pruning_method}_{pruning_value}.png"
        figure.savefig(os.path.join(save_folder, filename), dpi=300, bbox_inches="tight")
    if show:
        plt.show()

    return {
        "mst_edges": mst_edges,
        "pruned_edges": pruned_component_edges,
        "ordered_trees": ordered_trees,
        "synthetic_samples": synthetic_samples,
        "records": records,
        "meshes": meshes,
        "embedding": embedding,
        "synthetic_embedding": synthetic_embedding,
        "figure": figure,
        "axes": axes,
    }


def prune_mst_edges_fixed_threshold(mst_edges, threshold):
    """Keep MST edges whose length is at most ``threshold``."""
    mst_edges = np.asarray(mst_edges)
    if mst_edges.ndim != 2 or mst_edges.shape[1] != 4:
        raise ValueError("mst_edges must have shape (n_edges, 4)")
    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    return mst_edges[mst_edges[:, 2] <= threshold]


def prune_mst_edges_shortest_percent(mst_edges, percentage):
    """Keep the shortest ``percentage`` of MST edges."""
    mst_edges = np.asarray(mst_edges)
    if mst_edges.ndim != 2 or mst_edges.shape[1] != 4:
        raise ValueError("mst_edges must have shape (n_edges, 4)")
    if not 0 <= percentage <= 100:
        raise ValueError("percentage must be between 0 and 100")
    if len(mst_edges) == 0 or percentage == 0:
        return mst_edges[:0]

    number_to_keep = int(np.ceil(len(mst_edges) * percentage / 100))
    order = np.argsort(mst_edges[:, 2], kind="stable")
    return mst_edges[order[:number_to_keep]]


def prune_mst_edges_iqr(mst_edges, multiplier=1.5):
    """Reject edges longer than the upper IQR fence.

    The upper fence is ``Q3 + multiplier * (Q3 - Q1)``. The default
    multiplier of 1.5 is the conventional outlier criterion.
    """
    mst_edges = np.asarray(mst_edges)
    if mst_edges.ndim != 2 or mst_edges.shape[1] != 4:
        raise ValueError("mst_edges must have shape (n_edges, 4)")
    if multiplier < 0:
        raise ValueError("multiplier must be non-negative")
    if len(mst_edges) == 0:
        return mst_edges.copy()

    lengths = mst_edges[:, 2]
    first_quartile, third_quartile = np.percentile(lengths, [25, 75])
    upper_fence = third_quartile + multiplier * (third_quartile - first_quartile)
    return mst_edges[lengths <= upper_fence]


def prune_mst_edges(mst_edges, method=None, value=None, iqr_multiplier=1.5):
    """Apply a selected MST pruning strategy.

    ``method`` can be ``None``, ``"threshold"``, ``"shortest_percent"``,
    or ``"iqr"``. The ``value`` is the length threshold or percentage for
    the first two methods; ``iqr_multiplier`` configures the IQR method.
    """
    if method is None:
        return np.asarray(mst_edges)
    if method == "threshold":
        if value is None:
            raise ValueError("value is required for threshold pruning")
        return prune_mst_edges_fixed_threshold(mst_edges, value)
    if method == "shortest_percent":
        if value is None:
            raise ValueError("value is required for shortest_percent pruning")
        return prune_mst_edges_shortest_percent(mst_edges, value)
    if method == "iqr":
        return prune_mst_edges_iqr(mst_edges, iqr_multiplier)
    raise ValueError("method must be None, 'threshold', 'shortest_percent', or 'iqr'")


def relabel_mst_components(mst_edges, n_samples):
    """Recompute component IDs from the edges remaining after pruning."""
    mst_edges = np.asarray(mst_edges)
    if mst_edges.ndim != 2 or mst_edges.shape[1] != 4:
        raise ValueError("mst_edges must have shape (n_edges, 4)")
    if n_samples < 1:
        raise ValueError("n_samples must be positive")

    if len(mst_edges) == 0:
        return mst_edges.copy(), n_samples, np.arange(n_samples)

    rows = mst_edges[:, 0].astype(int)
    columns = mst_edges[:, 1].astype(int)
    graph = csr_matrix(
        (np.ones(2 * len(mst_edges)), (np.r_[rows, columns], np.r_[columns, rows])),
        shape=(n_samples, n_samples),
    )
    _, node_components = connected_components(graph, directed=False)
    relabeled_edges = mst_edges.copy()
    relabeled_edges[:, 3] = node_components[rows]
    component_count = len(np.unique(node_components))
    return relabeled_edges, component_count, node_components


def _select_cluster_center_neighbors(latent, centers, n_neighbors=1, selected_center=None):
    latent = np.asarray(latent, dtype=float)
    centers = np.asarray(centers, dtype=float)
    if latent.ndim != 2 or len(latent) < 1:
        raise ValueError("latent must contain at least one sample")
    if centers.ndim != 2 or len(centers) < 1 or centers.shape[1] != latent.shape[1]:
        raise ValueError("centers must match the feature dimension of latent")
    if not 1 <= n_neighbors <= len(latent):
        raise ValueError("n_neighbors must be between 1 and the number of samples")
    if selected_center is not None and not 0 <= selected_center < len(centers):
        raise ValueError("selected_center must be a valid center index")

    if len(centers) == 1:
        center_indices = [selected_center] if selected_center is not None else [0]
        return {int(center_index): np.empty(0, dtype=int) for center_index in center_indices}

    nearest_neighbors = NearestNeighbors(n_neighbors=n_neighbors)
    nearest_neighbors.fit(latent)
    _, indices = nearest_neighbors.kneighbors(centers)
    center_indices = [selected_center] if selected_center is not None else range(len(centers))
    return {
        int(center_index): indices[center_index].astype(int)
        for center_index in center_indices
    }


def _show_cluster_neighbor_meshes(source_meshes, center_neighbors, center_clusters, labels, title):
    import copy
    import open3d as o3d

    if source_meshes is None:
        return
    if len(source_meshes) != len(labels):
        raise ValueError("source_meshes must have one entry per sample")
    meshes = []
    mesh_labels = []
    mesh_colors = []
    mesh_positions = []
    colors = plt.get_cmap("tab10")
    for row_index, (center_index, neighbor_indices) in enumerate(center_neighbors.items()):
        cluster_color = colors(int(center_clusters[center_index]) % colors.N)[:3]
        for column_index, neighbor_index in enumerate(neighbor_indices):
            mesh = copy.deepcopy(source_meshes[int(neighbor_index)])
            mesh_labels.append(
                f"{labels[int(neighbor_index)]} (center {center_index}, neighbor {column_index})"
            )
            meshes.append(mesh)
            mesh_colors.append(cluster_color)
            mesh_positions.append((row_index, column_index))
    if not meshes:
        return

    extents = [np.asarray(mesh.get_axis_aligned_bounding_box().get_extent()) for mesh in meshes]
    spacing = max(float(np.max(extent)) for extent in extents) * 1.5
    for (row_index, column_index), (mesh, color) in zip(mesh_positions, zip(meshes, mesh_colors)):
        mesh_center = mesh.get_axis_aligned_bounding_box().get_center()
        mesh.translate(
            np.array([column_index * spacing, -row_index * spacing, 0.0]) - mesh_center
        )
        mesh.paint_uniform_color(color)

    o3d.visualization.draw_geometries(
        meshes,
        window_name=f"{title} (one row per cluster center): {', '.join(mesh_labels)}",
        mesh_show_wireframe=True,
    )


def plot_cluster_center_neighbors_3d(
    latent,
    cluster_labels,
    centers,
    center_neighbors,
    labels=None,
    center_labels=None,
    center_clusters=None,
    title="Cluster center nearest neighbours",
    show=True,
):
    """Display samples and selected cluster-center neighbours in 3D Plotly."""
    import plotly.graph_objects as go

    latent = np.asarray(latent, dtype=float)
    centers = np.asarray(centers, dtype=float)
    cluster_labels = np.asarray(cluster_labels)
    if latent.ndim != 2 or latent.shape[1] < 3:
        raise ValueError("latent must have at least three dimensions for a 3D plot")
    if len(cluster_labels) != len(latent):
        raise ValueError("cluster_labels must have one entry per sample")
    if centers.ndim != 2 or centers.shape[1] != latent.shape[1]:
        raise ValueError("centers must have the same feature dimension as latent")
    if labels is None:
        labels = [str(index) for index in range(len(latent))]
    if len(labels) != len(latent):
        raise ValueError("labels must have one entry per sample")
    if center_labels is None:
        center_labels = [f"center {index}" for index in range(len(centers))]

    unique_clusters = [cluster for cluster in np.unique(cluster_labels) if cluster != -1]
    colors = plt.get_cmap("tab10")
    figure = go.Figure()
    for color_index, cluster in enumerate(unique_clusters):
        sample_mask = cluster_labels == cluster
        rgba = colors(color_index % colors.N)
        color = f"rgba({int(rgba[0] * 255)},{int(rgba[1] * 255)},{int(rgba[2] * 255)},{rgba[3]})"
        figure.add_trace(go.Scatter3d(
            x=latent[sample_mask, 0], y=latent[sample_mask, 1], z=latent[sample_mask, 2],
            mode="markers+text", text=np.asarray(labels)[sample_mask], textposition="top center",
            marker={"size": 5, "color": color}, name=f"Cluster {cluster}",
        ))
    noise_mask = cluster_labels == -1
    if np.any(noise_mask):
        figure.add_trace(go.Scatter3d(
            x=latent[noise_mask, 0], y=latent[noise_mask, 1], z=latent[noise_mask, 2],
            mode="markers+text", text=np.asarray(labels)[noise_mask], textposition="top center",
            marker={"size": 5, "color": "gray"}, name="Noise",
        ))

    for center_index, neighbor_indices in center_neighbors.items():
        for neighbor_index in neighbor_indices:
            points = np.vstack((centers[center_index], latent[neighbor_index]))
            figure.add_trace(go.Scatter3d(
                x=points[:, 0], y=points[:, 1], z=points[:, 2],
                mode="lines", line={"color": "rgba(80,80,80,0.35)", "width": 2},
                showlegend=False,
            ))
    if center_clusters is None:
        center_clusters = np.arange(len(centers))
    center_colors = [
        f"rgb({int(colors(int(cluster) % colors.N)[0] * 255)},"
        f"{int(colors(int(cluster) % colors.N)[1] * 255)},"
        f"{int(colors(int(cluster) % colors.N)[2] * 255)})"
        for cluster in center_clusters
    ]
    figure.add_trace(go.Scatter3d(
        x=centers[:, 0], y=centers[:, 1], z=centers[:, 2],
        mode="markers+text", text=center_labels, textposition="top center",
        marker={"size": 9, "color": center_colors, "symbol": "diamond"}, name="Cluster centers",
    ))
    figure.update_layout(
        title=title,
        scene={"xaxis_title": "Latent 1", "yaxis_title": "Latent 2", "zaxis_title": "Latent 3"},
    )
    if show:
        figure.show()
    return figure


def _overlay_cluster_centers_on_umap(axes, embedding, center_embedding, center_neighbors, center_clusters):
    colors = plt.get_cmap("tab10")
    for center_index, neighbor_indices in center_neighbors.items():
        for neighbor_index in neighbor_indices:
            axes.plot(
                [center_embedding[center_index, 0], embedding[neighbor_index, 0]],
                [center_embedding[center_index, 1], embedding[neighbor_index, 1]],
                color=colors(int(center_clusters[center_index]) % colors.N),
                linewidth=0.8,
                alpha=0.35,
                zorder=1,
            )
    axes.scatter(
        center_embedding[:, 0], center_embedding[:, 1],
        color=[colors(int(cluster) % colors.N) for cluster in center_clusters],
        edgecolors="black", marker="D", s=70, zorder=4, label="Cluster centers",
    )


def compare_mst_clusters(
    latent,
    n_neighbors=3,
    metric="euclidean",
    pruning_method=None,
    pruning_value=None,
    iqr_multiplier=1.5,
    center_n_neighbors=1,
    selected_center=None,
    labels=None,
    source_meshes=None,
    show_meshes=False,
    save_folder=None,
    show=True,
):
    """Compare nearest neighbours of post-pruning MST component centers."""
    latent = np.asarray(latent, dtype=float)
    mst_edges = compute_mst_edges(latent, n_neighbors, metric)
    pruned_edges = prune_mst_edges(mst_edges, pruning_method, pruning_value, iqr_multiplier)
    pruned_edges, _, cluster_labels = relabel_mst_components(pruned_edges, len(latent))
    component_ids = np.unique(cluster_labels)
    centers = np.vstack([latent[cluster_labels == component_id].mean(axis=0) for component_id in component_ids])
    center_neighbors = _select_cluster_center_neighbors(latent, centers, center_n_neighbors, selected_center)
    if labels is None:
        labels = [str(index) for index in range(len(latent))]
    figure = plot_cluster_center_neighbors_3d(
        latent, cluster_labels, centers, center_neighbors, labels=labels,
        center_labels=[f"MST center {component_id}" for component_id in component_ids],
        center_clusters=component_ids,
        title="MST cluster center nearest neighbours", show=show,
    )
    if save_folder is not None:
        os.makedirs(save_folder, exist_ok=True)
        figure.write_html(os.path.join(save_folder, "mst_cluster_center_neighbors.html"))
    if show_meshes:
        _show_cluster_neighbor_meshes(source_meshes, center_neighbors, component_ids, labels,
                                      "MST cluster center nearest neighbours")
    return {"mst_edges": mst_edges, "pruned_edges": pruned_edges, "cluster_labels": cluster_labels,
            "centers": centers, "center_neighbors": center_neighbors,
            "center_clusters": component_ids, "figure": figure}


def compare_hdbscan_clusters(
    latent,
    min_cluster_size=3,
    metric="euclidean",
    center_n_neighbors=1,
    selected_center=None,
    labels=None,
    source_meshes=None,
    show_meshes=False,
    save_folder=None,
    show=True,
):
    """Compare nearest neighbours of HDBSCAN cluster centers."""
    latent = np.asarray(latent, dtype=float)
    clusterer = HDBSCAN(min_cluster_size=max(2, min_cluster_size), metric=metric, store_centers="both")
    cluster_labels = clusterer.fit_predict(latent)
    component_ids = np.asarray([cluster for cluster in np.unique(cluster_labels) if cluster != -1])
    if hasattr(clusterer, "centroids_"):
        centers = np.asarray(clusterer.centroids_, dtype=float)
    else:
        centers = np.vstack([latent[cluster_labels == cluster].mean(axis=0) for cluster in component_ids])
    center_neighbors = _select_cluster_center_neighbors(latent, centers, center_n_neighbors, selected_center)
    if labels is None:
        labels = [str(index) for index in range(len(latent))]
    figure = plot_cluster_center_neighbors_3d(
        latent, cluster_labels, centers, center_neighbors, labels=labels,
        center_labels=[f"HDBSCAN center {cluster}" for cluster in component_ids],
        center_clusters=component_ids,
        title="HDBSCAN cluster center nearest neighbours", show=show,
    )
    if save_folder is not None:
        os.makedirs(save_folder, exist_ok=True)
        figure.write_html(os.path.join(save_folder, "hdbscan_cluster_center_neighbors.html"))
    if show_meshes:
        _show_cluster_neighbor_meshes(source_meshes, center_neighbors, component_ids, labels,
                                      "HDBSCAN cluster center nearest neighbours")
    return {"clusterer": clusterer, "cluster_labels": cluster_labels, "centers": centers,
            "center_neighbors": center_neighbors, "center_clusters": component_ids,
            "figure": figure}


def plot_umap_with_clustering(
    embedding,
    latent,
    labels=None,
    clustering_method="mst",
    n_neighbors=3,
    metric="euclidean",
    pruning_method=None,
    pruning_value=None,
    iqr_multiplier=1.5,
    title="Training latent space UMAP with {clustering_method}",
    case=None,
    save_folder=None,
    show=True,
):
    """Plot an existing UMAP embedding with MST or HDBSCAN clusters."""
    embedding = np.asarray(embedding, dtype=float)
    latent = np.asarray(latent, dtype=float)
    if embedding.ndim != 2 or embedding.shape[1] != 2:
        raise ValueError("embedding must have shape (n_samples, 2)")
    if latent.ndim != 2 or latent.shape[0] != embedding.shape[0]:
        raise ValueError("latent and embedding must have the same number of samples")
    if labels is not None and len(labels) != len(embedding):
        raise ValueError("labels must have one entry per sample")
    if not isinstance(clustering_method, str):
        raise ValueError("clustering_method must be 'mst' or 'hdbscan'")
    clustering_method = clustering_method.lower()

    if clustering_method == "mst":
        mst_edges = compute_mst_edges(latent, n_neighbors, metric)
        mst_edges = prune_mst_edges(mst_edges, pruning_method, pruning_value, iqr_multiplier)
        mst_edges, component_count, node_components = relabel_mst_components(mst_edges, len(latent))
        component_ids = list(range(component_count))
        unique_labels = None
    elif clustering_method == "hdbscan":
        cluster_labels = HDBSCAN(min_cluster_size=max(2, n_neighbors), metric=metric).fit_predict(latent)
        unique_labels = np.unique(cluster_labels)
        node_components = np.searchsorted(unique_labels, cluster_labels)
        component_ids = list(range(len(unique_labels)))
        mst_edges = np.empty((0, 4), dtype=float)
        component_count = len(unique_labels)
    else:
        raise ValueError("clustering_method must be 'mst' or 'hdbscan'")

    figure, axes = plt.subplots(figsize=(9, 8))

    component_colors = plt.get_cmap("tab10")
    for edge_index, (sample_a, sample_b, _, component) in enumerate(mst_edges):
        sample_a = int(sample_a)
        sample_b = int(sample_b)
        edge_color = component_colors(int(component) % component_colors.N)
        axes.plot(
            embedding[[sample_a, sample_b], 0],
            embedding[[sample_a, sample_b], 1],
            color=edge_color,
            linewidth=1.5,
            alpha=0.8,
            zorder=1,
        )

    component_colors = plt.get_cmap("tab10")
    for component_id in component_ids:
        sample_mask = node_components == component_id
        label = "Noise" if clustering_method == "hdbscan" and unique_labels[component_id] == -1 else f"Component {component_id}"
        axes.scatter(
            embedding[sample_mask, 0],
            embedding[sample_mask, 1],
            color=component_colors(component_id % component_colors.N),
            edgecolors="black",
            linewidths=0.5,
            s=55,
            zorder=2,
            label=label,
        )

    if labels is not None:
        for point, label in zip(embedding, labels):
            axes.annotate(str(label), point, xytext=(4, 4), textcoords="offset points", fontsize=8)

    legend_handles = []
    for component_id in component_ids:
        component_label = (
            "Noise"
            if clustering_method == "hdbscan" and unique_labels[component_id] == -1
            else f"Component {component_id}"
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=component_colors(component_id % component_colors.N),
                markeredgecolor="black",
                markersize=7,
                label=component_label,
            )
        )
    axes.legend(handles=legend_handles)
    result_label = "connected component(s) after pruning" if clustering_method == "mst" else "HDBSCAN cluster(s)"
    axes.set_title(f"{title}\n{component_count} {result_label}")
    axes.set_xlabel("UMAP 1")
    axes.set_ylabel("UMAP 2")
    axes.grid(True, alpha=0.3)
    figure.tight_layout()

    if save_folder is not None:
        os.makedirs(save_folder, exist_ok=True)
        prefix = f"{case}_" if case else ""
        figure.savefig(os.path.join(save_folder, f"{prefix}_k={n_neighbors}_{pruning_method}_{pruning_value}.png"), dpi=300, bbox_inches="tight")
    if show:
        plt.show()

    return figure, axes, mst_edges


def plot_training_latent_umap_with_clustering(
    training_latent,
    labels=None,
    clustering_method="mst",
    n_neighbors=5,
    umap_neighbors=10,
    min_dist=0.1,
    metric="euclidean",
    pruning_method=None,
    pruning_value=None,
    iqr_multiplier=1.5,
    compare_clusters=False,
    compare_mst=False,
    compare_hdbscan=False,
    center_n_neighbors=3,
    selected_center=None,
    source_meshes=None,
    show_meshes=False,
    seed=STANDARD_SEED,
    title="Training latent space UMAP with {clustering_method}",
    case=None,
    save_folder=None,
    show=True,
):
    """Create the training UMAP and overlay its latent-space clustering.

      ``pruning_method`` can be ``None``, ``"threshold"``, ``"shortest_percent"``,
        or ``"iqr"``. The ``pruning_value`` is the length threshold or percentage for
        the first two methods; ``iqr_multiplier`` configures the IQR method."""
    training_latent = np.asarray(training_latent, dtype=float)
    if training_latent.ndim != 2:
        raise ValueError("training_latent must be a two-dimensional array")
    if not 2 <= umap_neighbors <= training_latent.shape[0]:
        raise ValueError("umap_neighbors must be between 2 and n_samples")

    scaler = StandardScaler()
    latent_scaled = scaler.fit_transform(training_latent)
    reducer = umap.UMAP(
        n_neighbors=umap_neighbors,
        min_dist=min_dist,
        n_components=2,
        metric=metric,
        random_state=seed,
    )
    embedding = reducer.fit_transform(latent_scaled)

    figure, axes, mst_edges = plot_umap_with_clustering(
        embedding,
        training_latent,
        labels=labels,
        clustering_method=clustering_method,
        n_neighbors=n_neighbors,
        metric=metric,
        pruning_method=pruning_method,
        pruning_value=pruning_value,
        iqr_multiplier=iqr_multiplier,
        title=title.format(clustering_method=clustering_method),
        case=case,
        save_folder=save_folder,
        show=show,
    )

    if compare_clusters:
        compare_mst = True
        compare_hdbscan = True
    comparison_result = None
    if compare_mst:
        mst_comparison = compare_mst_clusters(
            training_latent,
            n_neighbors=n_neighbors,
            metric=metric,
            pruning_method=pruning_method,
            pruning_value=pruning_value,
            iqr_multiplier=iqr_multiplier,
            center_n_neighbors=center_n_neighbors,
            selected_center=selected_center,
            labels=labels,
            source_meshes=source_meshes,
            show_meshes=show_meshes,
            save_folder=save_folder,
            show=show,
        )
        if clustering_method == "mst":
            comparison_result = mst_comparison
    if compare_hdbscan:
        hdbscan_comparison = compare_hdbscan_clusters(
            training_latent,
            min_cluster_size=n_neighbors,
            metric=metric,
            center_n_neighbors=center_n_neighbors,
            selected_center=selected_center,
            labels=labels,
            source_meshes=source_meshes,
            show_meshes=show_meshes,
            save_folder=save_folder,
            show=show,
        )
        if clustering_method == "hdbscan":
            comparison_result = hdbscan_comparison

    if comparison_result is not None:
        center_embedding = reducer.transform(scaler.transform(comparison_result["centers"]))
        _overlay_cluster_centers_on_umap(
            axes,
            embedding,
            center_embedding,
            comparison_result["center_neighbors"],
            comparison_result["center_clusters"],
        )
        axes.legend(loc="best")
        figure.tight_layout()
        if save_folder is not None:
            os.makedirs(save_folder, exist_ok=True)
            prefix = f"{case}_" if case else ""
            figure.savefig(
                os.path.join(save_folder, f"{prefix}umap_with_cluster_centers.png"),
                dpi=300,
                bbox_inches="tight",
            )

    return figure, axes, embedding, mst_edges, reducer, scaler
