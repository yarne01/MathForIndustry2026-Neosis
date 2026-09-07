import copy
import os
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
from matplotlib.colors import Normalize


def plot_geometries(logger, source_mesh_, target_meshes_=None, distance_values=None,
                    window_title="Source (Y) | Target (B)", offset_margin=1.0, include_reference=False):
    """
    Plot geometry using o3d. Source mesh must be provided, targets can be a list of geometries
    :param logger: logger object
    :param source_mesh_: o3d mesh, main one, plotted in yellow at the center
    :param target_meshes_: list of additional meshes to be plotted, counterclockwise from x-axis
    :param distance_values: list, comparison metric that relates similarities/differences among source and other meshes, if provided
    :param window_title: str, name for the plot
    :param offset_margin: offset to be used when potting multiple meshes
    :param include_reference: bool, add axis
    :return: none
    """

    # Copy meshes
    src_n = copy.deepcopy(source_mesh_)
    src_n.vertices = o3d.utility.Vector3dVector(np.asarray(source_mesh_.vertices))
    src_n.paint_uniform_color([1, 0.7, 0.1])   # yellow
    bb_src = src_n.get_axis_aligned_bounding_box()
    src_extent = bb_src.get_extent()

    # determine longest axis
    longest_axis = np.argmin(src_extent)

    # characteristic spacing
    spacing = offset_margin * np.max(src_extent)
    geometries = [src_n]

    if target_meshes_:

        offsets = []
        n_targets = len(target_meshes_)

        # case == 1 -> side by side
        if n_targets == 1:

            offset = np.zeros(3)
            offset[longest_axis] = spacing

            offsets.append(offset)

        # case >=2 -> circular arrangement
        else:
            # plane perpendicular to longest axis
            remaining_axes = [
                i for i in range(3)
                if i != longest_axis
            ]

            for i in range(n_targets):

                theta = 2.0 * np.pi * i / n_targets

                offset = np.zeros(3)

                offset[remaining_axes[0]] = spacing * np.cos(theta)
                offset[remaining_axes[1]] = spacing * np.sin(theta)

                offsets.append(offset)

        norm = []   # used only if distance_values
        if distance_values:
            norm = Normalize(
                vmin=np.min(distance_values),
                vmax=np.max(distance_values)
            )

        # cmap = cm.get_cmap("viridis")
        cmap = cm.get_cmap("jet")
        for i, (new_mesh, offset) in enumerate(zip(target_meshes_, offsets)):
            tgt_n = copy.deepcopy(new_mesh)
            tgt_n.vertices = o3d.utility.Vector3dVector(np.asarray(new_mesh.vertices))

            if distance_values:
                value = distance_values[i]
                color = cmap(norm(value))[:3]
            else:
                color = cmap(i / max(1, n_targets - 1))[:3]

            tgt_n.paint_uniform_color(color)  # blue
            tgt_n.translate(offset)

            geometries.append(tgt_n)

    # Coordinate frame
    if include_reference:
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        geometries.append(axis)

    # Display
    o3d.visualization.draw_geometries(
        geometries,
        window_name=window_title,
        mesh_show_wireframe=True,
    )

    logger.debug("Visualization complete.")


def save_geometries_image(geometries, filename, window_title="Open3D",
                          wireframe=True, width=1200, height=900, axis=None):
    """
    Save an image of Open3D geometries from a selected viewing axis.

    Parameters
    ----------
    geometries : list
        List of Open3D TriangleMesh objects.
    filename : str
        Output image path.
    window_title: str
        Name for the rendered image
    wireframe: bool
        Plot wireframe, default True.
    width: int
        Width of the image.
    height: int
        Height of the image.
    axis : str or None
        Camera viewing direction:
            'x'  -> look from +x toward origin
            '-x' -> look from -x toward origin
            'y'
            '-y'
            'z'
            '-z'
    """

    o3d_geometries = []

    for geometry in geometries:
        o3d_geometry = copy.deepcopy(geometry)
        o3d_geometry.vertices = o3d.utility.Vector3dVector(np.asarray(geometry.vertices))
        o3d_geometry.paint_uniform_color([1, 0.7, 0.1])
        o3d_geometries.append(o3d_geometry)
    vis = o3d.visualization.Visualizer()

    vis.create_window(window_name=window_title, width=width, height=height, visible=True)

    for g in o3d_geometries:
        vis.add_geometry(g)

    opt = vis.get_render_option()
    opt.mesh_show_wireframe = wireframe

    # Camera setup

    ctr = vis.get_view_control()

    if axis is not None:
        axis = axis.lower()
        filename = filename.replace(".png", f"_{axis}.png")

        if axis == 'x':
            front = (-1, 0, 0)

        elif axis == '-x':
            front = (1, 0, 0)

        elif axis == 'y':
            front = (0, -1, 0)

        elif axis == '-y':
            front = (0, 1, 0)

        elif axis == 'z':
            front = (0, 0, -1)

        elif axis == '-z':
            front = (0, 0, 1)

        else:
            raise ValueError(
                "axis must be one of "
                "'x', '-x', 'y', '-y', 'z', '-z'"
            )

        ctr.set_front(front)

        # keep z upward when possible
        if axis in ['z', '-z']:
            ctr.set_up((0, 1, 0))
        else:
            ctr.set_up((0, 0, 1))

        # fit geometry nicely
        ctr.set_zoom(0.8)

    # Render

    vis.poll_events()
    vis.update_renderer()

    # capture image
    vis.capture_screen_image(filename)

    # close automatically
    vis.destroy_window()


def apply_heatmap(mesh, values):
    """
    Generates heatmap for provided mesh accoring to values.

    Parameters
    ----------
    mesh: o3d.geometry.TriangleMesh
        Reference mesh object.
    values: numpy array, list of values to be used as base for the colormap.
    Returns
    -------
    out: o3d.utility.Vector3d
        Colourmap.
    """

    values = (values - values.min()) / (values.max() - values.min() + 1e-12)
    colors_ = mpl.colormaps["turbo"](values)[:, :3]

    out = copy.deepcopy(mesh)
    out.vertex_colors = o3d.utility.Vector3dVector(colors_)
    return out


def plot_som_simple(mapped_som_points, som_size=5, labels=None, title="Geometry SOM", case=None, save_folder=None):
    """
    Simple SOM plot using scatter. Rectangular grid
    :param mapped_som_points: array, (n_exp, 2) coordinates of the points
    :param som_size: side of the grid (must be consistent with som map)
    :param labels: array of str, names for the experiments
    :param title: name for the plot
    :param case: str, name of the usecase
    :param save_folder: str, path where figure will be saved
    :return: None
    """

    plt.figure(figsize=(8, 8))

    for i, w in enumerate(mapped_som_points):

        label = str(i)
        if labels:
            label = labels[i]

        # small random offset to avoid overlap
        offset_param = 0.05
        dx = np.random.uniform(-1.*offset_param, offset_param)
        dy = np.random.uniform(-1.*offset_param, offset_param)

        plt.scatter(w[0] + dx, w[1] + dy)
        plt.text(w[0] + dx,
                 w[1] + dy,
                 str(label),
                 fontsize=8)

    plt.xlim(-1, som_size)
    plt.ylim(-1, som_size)

    plt.grid()
    plt.title(title)

    if save_folder:
        plt.savefig(os.path.join(save_folder, f"{case}_{title}.png"), dpi=300, bbox_inches="tight")

    plt.show()


def plot_umap(embedding, new_embedding=None, labels=None, new_labels=None, title="Geometry UMAP", case=None, save_folder=None):
    """
    Scatter plot with UMAP points
    :param embedding: position of the experiments in 2 dimensions (n_exp, 2)
    :param new_embedding: position of the new experiments in 2 dimensions (n_exp, 2)
    :param labels: array of str, names for each point
    :param new_labels: array of str, names for each point
    :param title: str, title of the plot with extra information, like geometry / parameters or UMAP settings
    :param case: str, name of the usecase
    :param save_folder: str, path where figure will be saved
    :return: None
    """
    plt.figure(figsize=(8, 8))

    plt.scatter(embedding[:, 0], embedding[:, 1])
    for i in range(len(embedding)):
        label = str(i)
        if labels:
            label = labels[i]

        plt.text(embedding[i, 0], embedding[i, 1], str(label), fontsize=8)

    if new_embedding is not None:
        plt.scatter(new_embedding[:, 0], new_embedding[:, 1], color='r')
        for i in range(len(new_embedding)):
            label = str(i)
            if new_labels:
                label = new_labels[i]

            plt.text(new_embedding[i, 0], new_embedding[i, 1], str(label), fontsize=8)

    plt.title(title)
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")

    plt.grid()

    if save_folder:
        plt.savefig(os.path.join(save_folder, f"{case}_{title}.png"), dpi=300, bbox_inches="tight")

    plt.show()


def plot_overlapped_umap(embedding, labels=None, title="Geometry UMAP", legend_labels=None,
                         show_quivers=False, quiver_alpha=0.5, case=None, save_folder=None):
    """
    Scatter plot with UMAP points
    :param embedding: array of position of the experiments in 2 dimensions (n_exp, 2), number_of_maps
    :param labels: array of str, names for each point
    :param title: str, title of the plot with extra information, like geometry / parameters or UMAP settings
    :param legend_labels: list of str, name for each layer
    :param show_quivers: bool, plot quivers linking points
    :param quiver_alpha: float, 0.1 to 1, alpha for the arrows
    :param case: str, name of the usecase
    :param save_folder: str, path where figure will be saved
    :return: None
    """

    plt.figure(figsize=(8, 8))

    cmap = plt.get_cmap("tab10")

    for layer_idx, layer in enumerate(embedding):

        color = cmap(layer_idx % cmap.N)
        layer_name = f"Layer {layer_idx}"

        if legend_labels is not None:
            layer_name = legend_labels[layer_idx]

        plt.scatter(layer[:, 0], layer[:, 1], color=color, label=layer_name)

        for i in range(len(layer)):
            label = str(i)
            if labels:
                label = labels[i]
            plt.text(layer[i, 0], layer[i, 1], str(label), fontsize=8, color=color)

    # add quivers to keep track of how they move
    if show_quivers and len(embedding) > 1:

        for layer_idx in range(len(embedding) - 1):

            p0 = embedding[layer_idx]
            p1 = embedding[layer_idx + 1]

            plt.quiver(p0[:, 0], p0[:, 1], p1[:, 0] - p0[:, 0], p1[:, 1] - p0[:, 1],
                       angles='xy', scale_units='xy', scale=1, alpha=quiver_alpha, color='gray', width=0.002)

    plt.title(title)
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.grid(True)

    if len(embedding) > 1:
        plt.legend()

    plt.tight_layout()

    if save_folder:
        plt.savefig(os.path.join(save_folder, f"{case}_{title}.png"), dpi=300, bbox_inches="tight")

    plt.show()


def plot_parallel_coordinates(latent_, labels_=None, max_curves_=None, plot_title=None, component_names_=None,
                              highlighted_experiments_bold=None,
                              highlighted_experiments_dashed=None,
                              case=None, save_folder=None):
    """
    Parallel coodinates
    :param latent_: (n_exp, n_params) numpy array with projections in reduced space or experiments in design space
    :param labels_: array of names for the legend
    :param max_curves_: int, limit maximum number of curves
    :param plot_title: name to be included in the title (e.g. parameters, geometry or combined)
    :param component_names_: list[str], optional, names to be shown on the x-axis. If None, components are numbered 1..N.
    :param highlighted_experiments_bold: optional, list or None, experiments (indices) that will receive a thicker line
    :param highlighted_experiments_dashed: optional, list or None, experiments (indices) that will receive a dashed line
    :param case: str, name of the usecase
    :param save_folder: str, path where figure will be saved
    :return: None
    """

    plt.figure(figsize=(10, 6))
    n_samples, n_components = latent_.shape
    x = np.arange(n_components)

    # validate labels
    if component_names_ is not None:
        if len(component_names_) != n_components:
            raise ValueError(f"Expected {n_components} component names, got {len(component_names_)}")
    else:
        component_names_ = [str(i + 1) for i in range(n_components)]

    # optionally limit number of curves
    if max_curves_ is not None:
        n_samples = min(n_samples, max_curves_)

    for i in range(n_samples):
        linewidth = 1.
        linestyle = '-'
        if highlighted_experiments_bold and i in highlighted_experiments_bold:
            linewidth = 4
        if highlighted_experiments_dashed and i in highlighted_experiments_dashed:
            linestyle = '--'
            linewidth = 4
        if labels_ is not None:
            plt.plot(x, latent_[i], marker='o', linewidth=linewidth, linestyle=linestyle, label=labels_[i])
        else:
            plt.plot(x, latent_[i], marker='o', linewidth=linewidth, linestyle=linestyle)

    plt.xlabel("Component")
    plt.ylabel("Value")

    if plot_title:
        plt.title(plot_title)
    else:
        plt.title("Parallel coordinates")
    plt.grid(True)

    plt.xticks(x, component_names_, rotation=45, ha='right')

    # avoid huge legends
    if labels_ is not None and (n_samples <= 20 and (highlighted_experiments_bold or highlighted_experiments_dashed)):
        plt.legend()
        plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))

    plt.tight_layout()

    if save_folder:
        plt.savefig(os.path.join(save_folder, f"{case}_{plot_title}.png"), dpi=300, bbox_inches="tight")

    plt.show()


def plot_parallel_coordinates_statistics(latent_, plot_title=None, component_names_=None, case=None, save_folder=None):
    """
    Simple plot of max, mix and standard deviation for each component.
    :param latent_: (n_exp, n_params) numpy array with projections in reduced space
    :param plot_title: name to be included in the title (e.g. parameters, geometry or combined)
    :param component_names_: list[str], optional, names to be shown on the x-axis. If None, components are numbered 1..N.
    :param case: str, name of the usecase
    :param save_folder: str, path where figure will be saved
    :return: None
    """

    mean_ = np.mean(latent_, axis=0)
    std_ = np.std(latent_, axis=0)
    n_samples, n_components = latent_.shape

    x = np.arange(n_components)

    # validate labels
    if component_names_ is not None:
        if len(component_names_) != n_components:
            raise ValueError(f"Expected {n_components} component names, got {len(component_names_)}")
    else:
        component_names_ = [str(i + 1) for i in range(n_components)]

    plt.figure(figsize=(9, 5))
    plt.plot(x, mean_, marker='o')
    plt.fill_between(x, mean_ - std_, mean_ + std_, alpha=0.3)

    plt.xlabel("Component")
    plt.ylabel("Value")

    plt.xticks(x, component_names_, rotation=45, ha='right')

    full_title = f"Mean {plot_title} ± std"
    plt.title(full_title)

    plt.grid(True)
    plt.tight_layout()

    if save_folder:
        plt.savefig(os.path.join(save_folder, f"{case}_{full_title}.png"), dpi=300, bbox_inches="tight")

    plt.show()
