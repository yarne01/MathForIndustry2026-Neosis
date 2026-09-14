import os
import re
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
import open3d as o3d
import trimesh
from scipy.spatial import cKDTree
import scipy.ndimage
from skimage.measure import marching_cubes
from scipy.stats import wasserstein_distance


# ==================================================================================================
# 1. Model Architecture & Data Extraction
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
        latent = self.fc1(torch.flatten(features, 1))
        expanded = self.fc2(latent).view(-1, 64, *self.encoded_shape)
        return self.decoder(expanded)


def extract_mesh_raw(sdf_array, enforce_symmetry=True):
    if enforce_symmetry:
        sdf_array = (sdf_array + np.flip(sdf_array, axis=1)) / 2.0
    smoothed = scipy.ndimage.gaussian_filter(sdf_array, sigma=0.5)
    try:
        verts, faces, normals, _ = marching_cubes(smoothed, level=0.0)
    except RuntimeError:
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
    Returns the STL with the absolute lowest geometric error for metric evaluation.
    """
    import os
    import numpy as np
    import open3d as o3d
    from scipy.spatial import cKDTree

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

                        try:
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
                        except Exception:
                            # Silently skip any corrupted STL files
                            pass
                        break  # Only evaluate the first .stl in each folder

    if best_stl_mesh is not None:
        folder_match = os.path.basename(os.path.dirname(best_stl_path))
        print(f"  -> GT Match: Folder {folder_match} (Fit Error: {best_distance:.4f})")
        best_stl_mesh.compute_vertex_normals()
        return best_stl_mesh
    else:
        print(f"  [!] Search failed for {label}. Falling back to voxel parent.")
        return voxel_mesh


# ==================================================================================================
# 2. Geometry & Mesh Utilities (Yarne's Setup)
# ==================================================================================================
def o3d_to_trimesh(mesh_o3d):
    vertices = np.asarray(mesh_o3d.vertices)
    faces = np.asarray(mesh_o3d.triangles)
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def _get_sample_points(mesh, n_points=10000):
    t_mesh = o3d_to_trimesh(mesh) if not isinstance(mesh, trimesh.Trimesh) else mesh
    points, _ = trimesh.sample.sample_surface(t_mesh, n_points)
    return points


class MeshInfo:
    def __init__(self, mesh, n_samples=10000):
        self.mesh = mesh
        self.sample_points = _get_sample_points(self.mesh, n_samples)
        self.tree = cKDTree(self.sample_points)


# ==================================================================================================
# 3. Yarne's Custom Metric Suite
# ==================================================================================================
def get_distance_of_meshes(meshinfo_a, meshinfo_b, metric="Hausdorff"):
    if meshinfo_a == meshinfo_b:
        return 0.0

    if metric == "Hausdorff":
        dist_a, _ = meshinfo_b.tree.query(meshinfo_a.sample_points)
        dist_b, _ = meshinfo_a.tree.query(meshinfo_b.sample_points)
        return np.max(dist_a ** 2) + np.max(dist_b ** 2)

    if metric == "Chamfer":
        dist_a, _ = meshinfo_b.tree.query(meshinfo_a.sample_points)
        dist_b, _ = meshinfo_a.tree.query(meshinfo_b.sample_points)
        return np.mean(dist_a) + np.mean(dist_b)

    if metric in ["IntNorm-uniform(Yarne)", "IntNorm(Yarne)"]:
        box_dimension = [None, None, None]
        for idx_dim in range(3):
            array_at_index = [a[idx_dim] for a in np.vstack((meshinfo_a.sample_points, meshinfo_b.sample_points))]
            box_dimension[idx_dim] = (min(array_at_index), max(array_at_index))

        total_amount_points = 10 ** 3
        length_per_dim = [dimm[1] - dimm[0] for dimm in box_dimension]

        if metric == "IntNorm-uniform(Yarne)":
            ratio = np.power(total_amount_points / np.prod(length_per_dim), 1 / 3)
            amount_points_per_dimm = [max(int(np.floor(l * ratio)), 3) for l in length_per_dim]
        else:
            amount_points_per_dimm = [int(np.power(total_amount_points, 1 / 3))] * 3

        CoordsTODO = [np.linspace(box_dimension[i][0], box_dimension[i][1], amount_points_per_dimm[i]) for i in
                      range(3)]
        listCoords = [[x, y, z] for x in CoordsTODO[0] for y in CoordsTODO[1] for z in CoordsTODO[2]]

        d_a, _ = meshinfo_a.tree.query(listCoords)
        d_b, _ = meshinfo_b.tree.query(listCoords)
        d = np.power(np.abs(d_a - d_b), 2)
        return np.sqrt(np.sum(d) * np.prod(np.array(length_per_dim) / np.array(amount_points_per_dimm)))

    if metric == "Wasserstein-POT":
        weight_a = np.ones(len(meshinfo_a.sample_points)) / len(meshinfo_a.sample_points)
        weight_b = np.ones(len(meshinfo_b.sample_points)) / len(meshinfo_b.sample_points)
        cost_matrix = ot.dist(meshinfo_a.sample_points, meshinfo_b.sample_points, metric='euclidean')
        return ot.emd2(weight_a, weight_b, cost_matrix)

    if metric == "Wasserstein-Sliced":
        NUMBER_OF_PROJECTIONS = 100
        directions = np.random.randn(NUMBER_OF_PROJECTIONS, 3)
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)

        distances = []
        for d in directions:
            proj1 = meshinfo_a.sample_points @ d
            proj2 = meshinfo_b.sample_points @ d
            distances.append(wasserstein_distance(proj1, proj2))
        return np.mean(distances)

    raise Exception(f"Unknown metric: {metric}")


# ==================================================================================================
# 4. Main Benchmark Loop for All Scenarios (2x2 Grid Output)
# ==================================================================================================
if __name__ == "__main__":
    cases = ["Blended", "Tesla_valve", "Sokaris", "Fan"]
    metrics_to_run = ["Hausdorff", "Chamfer", "IntNorm-uniform(Yarne)", "Wasserstein-Sliced"]

    # Nested dictionary to store all final averages: results[case][metric]["PCA" or "AE"]
    results = {case: {metric: {"PCA": 0.0, "AE": 0.0, "valid_samples": 0} for metric in metrics_to_run} for case in
               cases}

    print("\n" + "=" * 90)
    print(" STARTING FULL DATASET BENCHMARK (4 SCENARIOS)")
    print("=" * 90)

    for case_name in cases:
        print(f"\n>>> PROCESSING SCENARIO: {case_name.upper()} <<<")
        sdf_folder_path = rf"results\sdfs\{case_name}"
        model_path = os.path.join(sdf_folder_path, "trained_sdf_ae.pth")

        # Only enforce symmetry if we are processing the Blended airplane dataset
        enforce_sym = (case_name.lower() == "blended")

        # 1. Load Dataset
        valid_sdfs, file_labels = [], []
        valid_keywords = ('case', 'exp', 'valve', 'fan', 'sokaris')

        if not os.path.exists(sdf_folder_path):
            print(f"  [!] Folder {sdf_folder_path} not found. Skipping.")
            continue

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

        if not valid_sdfs:
            print(f"  [!] No valid .npz files found for {case_name}. Skipping.")
            continue

        training_sdfs = np.array(valid_sdfs)
        n_samples, nx, ny, nz = training_sdfs.shape
        grid_shape = (nx, ny, nz)
        print(f"  -> Loaded {n_samples} geometries.")

        # 2. Fit PCA
        sdfs_flat = training_sdfs.reshape(n_samples, -1)
        pca_dim = min(16, n_samples)
        pca = PCA(n_components=pca_dim)
        pca_latents = pca.fit_transform(sdfs_flat)

        # 3. Load AE
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            checkpoint = torch.load(model_path, map_location=device)
            model = SDFAutoencoder(grid_shape=grid_shape, latent_dim=16).to(device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
        except Exception as e:
            print(f"  [!] Could not load AE model for {case_name}: {e}")
            continue

        sdf_tensor = torch.tensor(training_sdfs, dtype=torch.float32).unsqueeze(1).to(device)
        with torch.no_grad():
            ae_features = model.encoder(sdf_tensor)
            ae_latents = model.fc1(torch.flatten(ae_features, 1)).cpu().numpy()

        # 4. Evaluate Every Sample
        for idx in range(n_samples):
            label = file_labels[idx]

            gt_mesh = get_original_mesh(case_name, label, training_sdfs[idx], enforce_sym)
            info_gt = MeshInfo(gt_mesh)

            z_pca = pca_latents[idx]
            sdf_pca = pca.inverse_transform(z_pca).reshape(grid_shape)
            pca_mesh = extract_mesh_raw(sdf_pca, enforce_symmetry=enforce_sym)
            info_pca = MeshInfo(pca_mesh)

            z_ae = ae_latents[idx]
            with torch.no_grad():
                z_tensor = torch.tensor(z_ae, dtype=torch.float32).unsqueeze(0).to(device)
                expanded = model.fc2(z_tensor).view(-1, 64, *model.encoded_shape)
                sdf_ae = model.decoder(expanded).squeeze().cpu().numpy()
            ae_mesh = extract_mesh_raw(sdf_ae, enforce_symmetry=enforce_sym)
            info_ae = MeshInfo(ae_mesh)

            for metric in metrics_to_run:
                err_pca = get_distance_of_meshes(info_gt, info_pca, metric)
                err_ae = get_distance_of_meshes(info_gt, info_ae, metric)

                results[case_name][metric]["PCA"] += err_pca
                results[case_name][metric]["AE"] += err_ae

            for metric in metrics_to_run:
                results[case_name][metric]["valid_samples"] += 1

    # ==================================================================================================
    # 5. Print the 2x2 Final Presentation Grid
    # ==================================================================================================
    # Calculate final averages safely
    for case in cases:
        for metric in metrics_to_run:
            n = results[case][metric]["valid_samples"] / len(
                metrics_to_run)  # Adjusting since we incremented per metric
            if n > 0:
                results[case][metric]["PCA"] /= n
                results[case][metric]["AE"] /= n


    def get_str(val):
        return f"{val:.4f}" if val > 0 else "N/A"


    def get_winner(pca_val, ae_val):
        if pca_val == 0 and ae_val == 0: return "N/A"
        return "AE" if ae_val < pca_val else "PCA"


    # Define the 2x2 grid layout
    row_1_cases = ["Blended", "Tesla_valve"]
    row_2_cases = ["Sokaris", "Fan"]

    print("\n\n")
    print("=" * 122)
    print(f"{'':<30} | {row_1_cases[0].upper():^42} | {row_1_cases[1].replace('_', ' ').upper():^42} |")
    print("-" * 122)
    print(f"{'Metric':<30} | {'PCA':^12} | {'AE':^12} | {'Better':^12} | {'PCA':^12} | {'AE':^12} | {'Better':^12} |")
    print("=" * 122)

    for metric in metrics_to_run:
        c1, c2 = row_1_cases[0], row_1_cases[1]

        p1, a1 = results[c1][metric]["PCA"], results[c1][metric]["AE"]
        p2, a2 = results[c2][metric]["PCA"], results[c2][metric]["AE"]

        win1 = get_winner(p1, a1)
        win2 = get_winner(p2, a2)

        print(
            f"{metric:<30} | {get_str(p1):^12} | {get_str(a1):^12} | {win1:^12} | {get_str(p2):^12} | {get_str(a2):^12} | {win2:^12} |")

    print("=" * 122)
    print(f"{'':<30} | {row_2_cases[0].upper():^42} | {row_2_cases[1].upper():^42} |")
    print("-" * 122)
    print(f"{'Metric':<30} | {'PCA':^12} | {'AE':^12} | {'Better':^12} | {'PCA':^12} | {'AE':^12} | {'Better':^12} |")
    print("=" * 122)

    for metric in metrics_to_run:
        c1, c2 = row_2_cases[0], row_2_cases[1]

        p1, a1 = results[c1][metric]["PCA"], results[c1][metric]["AE"]
        p2, a2 = results[c2][metric]["PCA"], results[c2][metric]["AE"]

        win1 = get_winner(p1, a1)
        win2 = get_winner(p2, a2)

        print(
            f"{metric:<30} | {get_str(p1):^12} | {get_str(a1):^12} | {win1:^12} | {get_str(p2):^12} | {get_str(a2):^12} | {win2:^12} |")
    print("=" * 122)
    print("\n")