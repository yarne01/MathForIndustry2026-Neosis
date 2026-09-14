import os
import re
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from skimage.measure import marching_cubes
import open3d as o3d
import scipy.ndimage


# ==================================================================================================
# 1. Model Architecture
# ==================================================================================================
class SDFAutoencoder(nn.Module):
    def __init__(self, grid_shape, latent_dim=16):
        super().__init__()
        self.grid_shape = grid_shape

        self.encoder = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, stride=(2, 2, 1), padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv3d(16, 32, kernel_size=3, stride=(2, 2, 1), padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv3d(32, 64, kernel_size=3, stride=(2, 2, 1), padding=1),
            nn.LeakyReLU(0.2)
        )

        dummy = torch.zeros(1, 1, *grid_shape)
        encoded_dummy = self.encoder(dummy)
        self.encoded_shape = encoded_dummy.shape[2:]
        self.flattened_size = encoded_dummy.numel()

        self.fc1 = nn.Linear(self.flattened_size, latent_dim)
        self.fc2 = nn.Linear(latent_dim, self.flattened_size)

        self.decoder = nn.Sequential(
            nn.Conv3d(64, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=(2, 2, 1), mode='trilinear', align_corners=False),

            nn.Conv3d(32, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=(2, 2, 1), mode='trilinear', align_corners=False),

            nn.Conv3d(16, 1, kernel_size=3, padding=1),
            nn.Upsample(size=grid_shape, mode='trilinear', align_corners=False)
        )

    def forward(self, x):
        features = self.encoder(x)
        features_flat = torch.flatten(features, 1)
        latent = self.fc1(features_flat)
        expanded = self.fc2(latent).view(-1, 64, *self.encoded_shape)
        return self.decoder(expanded)


# ==================================================================================================
# 2. Geometry & Metric Helpers
# ==================================================================================================
def extract_mesh_raw(sdf_array, enforce_symmetry=True):
    """Extracts raw mesh from SDF to expose topological defects without artificial cleaning."""
    if enforce_symmetry:
        sdf_array = (sdf_array + np.flip(sdf_array, axis=1)) / 2.0

    smoothed = scipy.ndimage.gaussian_filter(sdf_array, sigma=0.5)
    try:
        verts, faces, normals, _ = marching_cubes(smoothed, level=0.0)
    except RuntimeError:
        # Fails if 0-level set does not exist (empty space)
        return o3d.geometry.TriangleMesh()

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()
    return mesh


def get_original_mesh(case_name, label, fallback_sdf, enforce_sym):
    """
    Brute-force searches all available STLs in the dataset directory, scaling each one
    to voxel space and calculating the Chamfer distance against the target voxel grid.
    Returns the STL with the absolute lowest geometric error.
    """
    import os
    import numpy as np
    import open3d as o3d
    from scipy.spatial import cKDTree

    print(f"\n  -> Searching all STLs to find the closest physical match for [{label}]...")

    # 1. Generate our search target (the exact voxel representation of the network)
    voxel_mesh = extract_mesh_raw(fallback_sdf, enforce_symmetry=enforce_sym)

    # Sample 5,000 points from the voxel surface and build a KD-Tree for fast distance queries
    target_pcd = voxel_mesh.sample_points_uniformly(number_of_points=5000)
    target_points = np.asarray(target_pcd.points)

    if len(target_points) == 0:
        return voxel_mesh  # Safety fallback for empty geometries

    target_tree = cKDTree(target_points)

    base_dir = os.path.join("test_files", case_name, "Cases", "Case1", "TrainingBlock1", "ImportedData")

    best_stl_mesh = None
    best_distance = float('inf')
    best_stl_path = ""

    if os.path.exists(base_dir):
        # Pre-calculate target scaling metrics to apply to every candidate
        voxel_bbox = voxel_mesh.get_axis_aligned_bounding_box()
        vox_ext = voxel_bbox.get_extent().max()
        vox_center = voxel_bbox.get_center()

        # 2. Iterate through every single Exp folder
        for folder_name in os.listdir(base_dir):
            folder_path = os.path.join(base_dir, folder_name)

            if os.path.isdir(folder_path):
                for file in os.listdir(folder_path):
                    if file.lower().endswith(".stl"):
                        stl_path = os.path.join(folder_path, file)

                        # Load candidate STL
                        candidate_mesh = o3d.io.read_triangle_mesh(stl_path)
                        stl_bbox = candidate_mesh.get_axis_aligned_bounding_box()
                        stl_ext = stl_bbox.get_extent().max()

                        # Scale and align the candidate STL to the Voxel space
                        scale_factor = vox_ext / stl_ext if stl_ext > 0 else 1.0
                        candidate_mesh.translate(-stl_bbox.get_center())
                        candidate_mesh.scale(scale_factor, center=(0, 0, 0))
                        candidate_mesh.translate(vox_center)

                        # Sample points on the aligned candidate
                        candidate_pcd = candidate_mesh.sample_points_uniformly(number_of_points=5000)
                        candidate_points = np.asarray(candidate_pcd.points)

                        if len(candidate_points) == 0:
                            continue

                        # 3. Calculate symmetric Chamfer Distance
                        candidate_tree = cKDTree(candidate_points)
                        dist_target_to_cand, _ = candidate_tree.query(target_points)
                        dist_cand_to_target, _ = target_tree.query(candidate_points)

                        chamfer_dist = np.mean(dist_target_to_cand) + np.mean(dist_cand_to_target)

                        # 4. Keep the absolute best match
                        if chamfer_dist < best_distance:
                            best_distance = chamfer_dist
                            best_stl_mesh = candidate_mesh
                            best_stl_path = stl_path
                        break  # Only evaluate the first .stl in each folder

    if best_stl_mesh is not None:
        folder_match = os.path.basename(os.path.dirname(best_stl_path))
        print(f"  -> Found closest match! (Folder: {folder_match} | Chamfer Error: {best_distance:.4f})")
        best_stl_mesh.compute_vertex_normals()
        return best_stl_mesh
    else:
        print(f"  [!] Search failed. Falling back to voxel parent.")
        return voxel_mesh


def compute_mesh_metrics(mesh):
    """Computes physical volume, surface area, and topological connectivity."""
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        return {"volume": 0.0, "surface_area": 0.0, "num_bodies": 0, "is_watertight": False}

    is_watertight = mesh.is_watertight()
    surface_area = mesh.get_surface_area()

    # Calculate volume (fallback to AABB bounding box volume if not strictly watertight)
    try:
        if is_watertight:
            volume = mesh.get_volume()
        else:
            bbox = mesh.get_axis_aligned_bounding_box()
            volume = np.prod(bbox.get_extent())
    except Exception:
        bbox = mesh.get_axis_aligned_bounding_box()
        volume = np.prod(bbox.get_extent())

    # Identify disconnected floating bodies (ghosting artifacts)
    _, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
    num_bodies = len(cluster_n_triangles)

    return {
        "volume": volume,
        "surface_area": surface_area,
        "num_bodies": num_bodies,
        "is_watertight": is_watertight,
        "n_vertices": len(mesh.vertices),
        "n_triangles": len(mesh.triangles)
    }


# ==================================================================================================
# 3. Main Comparison Pipeline
# ==================================================================================================
if __name__ == "__main__":
    Plane = True
    sdf_folder_path = r"results\sdfs\Blended"
    model_path = os.path.join(sdf_folder_path, "trained_sdf_ae.pth")

    # --- CONFIGURE YOUR HYBRID HERE ---
    source_indices = [42]  # Indices of geometries to blend
    weights = [1]  # Blending ratio (must sum to 1.0)
    enforce_y_symmetry = Plane  # Apply (SDF + flip) / 2

    # 1. Load Dataset
    valid_sdfs, file_labels = [], []
    valid_keywords = ('case', 'exp', 'valve')

    for filename in sorted(os.listdir(sdf_folder_path)):
        lower = filename.lower()
        if lower.endswith(".npz") and any(k in lower for k in valid_keywords):
            filepath = os.path.join(sdf_folder_path, filename)
            try:
                with np.load(filepath) as data:
                    valid_sdfs.append(data[data.files[0]])
                    file_labels.append(filename.split('.')[0])
            except Exception:
                pass

    training_sdfs = np.array(valid_sdfs)
    n_samples, nx, ny, nz = training_sdfs.shape
    grid_shape = (nx, ny, nz)
    print(f"Loaded {n_samples} geometries with grid shape {grid_shape}")

    # 2. Fit 16-Component Linear PCA Baseline
    sdfs_flat = training_sdfs.reshape(n_samples, -1)
    pca_dim = min(16, n_samples)
    pca = PCA(n_components=pca_dim)
    pca_latents = pca.fit_transform(sdfs_flat)
    print(
        f"Fitted baseline PCA with {pca_dim} components (Explained Var: {pca.explained_variance_ratio_.sum() * 100:.2f}%)")

    # 3. Load 16D Autoencoder
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    model = SDFAutoencoder(grid_shape=grid_shape, latent_dim=16).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Encode all geometries through the AE
    sdf_tensor = torch.tensor(training_sdfs, dtype=torch.float32).unsqueeze(1).to(device)
    with torch.no_grad():
        ae_features = model.encoder(sdf_tensor)
        ae_latents = model.fc1(torch.flatten(ae_features, 1)).cpu().numpy()

    # 4. Generate Latent Blends
    weights = np.array(weights, dtype=np.float32)
    weights /= weights.sum()

    # --- Method A: Linear PCA Reconstruction ---
    z_pca_hybrid = np.zeros(pca_dim)
    for idx, w in zip(source_indices, weights):
        z_pca_hybrid += w * pca_latents[idx]

    sdf_pca_flat = pca.inverse_transform(z_pca_hybrid)
    sdf_pca_3d = sdf_pca_flat.reshape(grid_shape)
    mesh_pca = extract_mesh_raw(sdf_pca_3d, enforce_symmetry=enforce_y_symmetry)

    # --- Method B: Non-Linear Autoencoder Reconstruction ---
    z_ae_hybrid = np.zeros(16)
    for idx, w in zip(source_indices, weights):
        z_ae_hybrid += w * ae_latents[idx]

    with torch.no_grad():
        z_tensor = torch.tensor(z_ae_hybrid, dtype=torch.float32).unsqueeze(0).to(device)
        expanded = model.fc2(z_tensor).view(-1, 64, *model.encoded_shape)
        sdf_ae_3d = model.decoder(expanded).squeeze().cpu().numpy()

    mesh_ae = extract_mesh_raw(sdf_ae_3d, enforce_symmetry=enforce_y_symmetry)

    # 5. Extract Parent Meshes for Baseline Reference
    parent_metrics = [compute_mesh_metrics(extract_mesh_raw(training_sdfs[i], enforce_y_symmetry)) for i in
                      source_indices]
    expected_vol = sum(w * m['volume'] for w, m in zip(weights, parent_metrics))

    # 6. Compute Comparative Metrics
    metrics_pca = compute_mesh_metrics(mesh_pca)
    metrics_ae = compute_mesh_metrics(mesh_ae)

    # ----------------------------------------------------------------------------------------------
    # 7. Print Comparative Metrics Report
    # ----------------------------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(
        f" HYBRID GEOMETRY EVALUATION: {' + '.join([f'{w * 100:.0f}% {file_labels[i]}' for i, w in zip(source_indices, weights)])}")
    print("=" * 80)
    print(f"{'Metric':<32} | {'Linear PCA (16D)':<20} | {'Autoencoder (16D)':<20}")
    print("-" * 80)
    print(f"{'Physical Volume (voxels^3)':<32} | {metrics_pca['volume']:<20.2f} | {metrics_ae['volume']:<20.2f}")
    print(f"{'Expected Linear Volume':<32} | {expected_vol:<20.2f} | {expected_vol:<20.2f}")

    vol_ret_pca = (metrics_pca['volume'] / expected_vol * 100) if expected_vol > 0 else 0
    vol_ret_ae = (metrics_ae['volume'] / expected_vol * 100) if expected_vol > 0 else 0
    print(f"{'Volume Retention Ratio':<32} | {vol_ret_pca:<19.1f}% | {vol_ret_ae:<19.1f}%")

    print(
        f"{'Surface Area (voxels^2)':<32} | {metrics_pca['surface_area']:<20.2f} | {metrics_ae['surface_area']:<20.2f}")
    print(f"{'Isolated Bodies (Ghosting)':<32} | {metrics_pca['num_bodies']:<20d} | {metrics_ae['num_bodies']:<20d}")
    print(
        f"{'Watertight Surface':<32} | {str(metrics_pca['is_watertight']):<20} | {str(metrics_ae['is_watertight']):<20}")
    print(f"{'Triangle Count':<32} | {metrics_pca['n_triangles']:<20d} | {metrics_ae['n_triangles']:<20d}")
    print("=" * 80)

    # ----------------------------------------------------------------------------------------------
    # 8. Side-by-Side Visual Inspection with Ground Truth Parents
    # ----------------------------------------------------------------------------------------------
    mesh_pca.paint_uniform_color([0.85, 0.25, 0.25])  # Red
    mesh_ae.paint_uniform_color([0.20, 0.50, 0.80])  # Blue

    # Calculate offset based on the size of the generated mesh
    bbox = mesh_ae.get_axis_aligned_bounding_box()
    extent = bbox.get_extent()
    offset_x = extent[0] * 1.5 if extent[0] > 0 else 100.0

    display_meshes = []
    case_name = os.path.basename(os.path.normpath(sdf_folder_path))
    print("\nLoading Anchor Geometries...")

    if len(source_indices) == 2:
        # Layout: [Parent A] --> [PCA Hybrid] --> [AE Hybrid] <-- [Parent B]

        # Load Ground Truth A
        mesh_p0 = get_original_mesh(case_name, file_labels[source_indices[0]], training_sdfs[source_indices[0]],
                                    enforce_y_symmetry)
        mesh_p0.paint_uniform_color([0.65, 0.65, 0.65])  # Grey
        mesh_p0.translate([-offset_x * 1.5, 0.0, 0.0])

        mesh_pca.translate([-offset_x * 0.5, 0.0, 0.0])
        mesh_ae.translate([offset_x * 0.5, 0.0, 0.0])

        # Load Ground Truth B
        mesh_p1 = get_original_mesh(case_name, file_labels[source_indices[1]], training_sdfs[source_indices[1]],
                                    enforce_y_symmetry)
        mesh_p1.paint_uniform_color([0.65, 0.65, 0.65])  # Grey
        mesh_p1.translate([offset_x * 1.5, 0.0, 0.0])

        display_meshes.extend([mesh_p0, mesh_pca, mesh_ae, mesh_p1])

        print("\nVisual Key (Left to Right):")
        print(f"  1. [Grey] Ground Truth Parent A ({file_labels[source_indices[0]]})")
        print("  2. [Red]  Linear PCA Hybrid")
        print("  3. [Blue] Autoencoder Hybrid")
        print(f"  4. [Grey] Ground Truth Parent B ({file_labels[source_indices[1]]})")

    else:
        # N-Way Blend Layout: All Parents on the left, then PCA, then AE
        for i, idx in enumerate(source_indices):
            m = get_original_mesh(case_name, file_labels[idx], training_sdfs[idx], enforce_y_symmetry)
            m.paint_uniform_color([0.65, 0.65, 0.65])
            m.translate([-offset_x * (len(source_indices) - i), 0.0, 0.0])
            display_meshes.append(m)

        mesh_pca.translate([0.0, 0.0, 0.0])
        mesh_ae.translate([offset_x, 0.0, 0.0])
        display_meshes.extend([mesh_pca, mesh_ae])

        print("\nVisual Key (Left to Right):")
        print("  - [Grey] Ground Truth Anchor Geometries")
        print("  - [Red]  Linear PCA Hybrid")
        print("  - [Blue] Autoencoder Hybrid")

    o3d.visualization.draw_geometries(
        display_meshes,
        window_name="Latent Space Comparison: PCA vs Autoencoder",
        mesh_show_wireframe=True
    )