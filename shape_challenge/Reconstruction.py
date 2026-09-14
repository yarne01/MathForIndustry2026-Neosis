import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from skimage.measure import marching_cubes
import open3d as o3d
import scipy.ndimage


# ==================================================================================================
# 1. The Network Architecture (Required to load the state_dict)
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
        reconstructed_sdf = self.decoder(expanded)
        return reconstructed_sdf


def extract_mesh(sdf_array):
    smoothed_sdf = scipy.ndimage.gaussian_filter(sdf_array, sigma=0.5)
    verts, faces, normals, values = marching_cubes(smoothed_sdf, level=0.0)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()
    return mesh


# ==================================================================================================
# 2. Initialization and Data Loading
# ==================================================================================================
if __name__ == "__main__":
    # --- 1. Point this to the case folder you want to explore ---
    Plane = True
    sdf_folder_path = r"results\sdfs\Blended"
    model_path = os.path.join(sdf_folder_path, "trained_sdf_ae.pth")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Could not find trained model at {model_path}. Run training script first!")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on {device}...")

    # Load the checkpoint dictionary
    checkpoint = torch.load(model_path, map_location=device)
    grid_shape = checkpoint['grid_shape']

    # Rebuild the exact network architecture dynamically
    model = SDFAutoencoder(grid_shape=grid_shape, latent_dim=16).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Model successfully loaded. Grid shape locked to: {grid_shape}")

    # Load the original data to populate the Latent Space
    valid_sdfs = []
    file_labels = []
    valid_keywords = ('case', 'exp', 'valve')

    for filename in os.listdir(sdf_folder_path):
        lower_name = filename.lower()
        if lower_name.endswith(".npz") and any(keyword in lower_name for keyword in valid_keywords):
            filepath = os.path.join(sdf_folder_path, filename)
            try:
                with np.load(filepath) as data:
                    arr = data[data.files[0]]
                if arr.shape == grid_shape:
                    valid_sdfs.append(arr)
                    file_labels.append(filename.split('.')[0])  # Store name without extension for the plot
            except Exception:
                pass

    training_sdfs_3d = np.array(valid_sdfs)
    sdf_tensor = torch.tensor(training_sdfs_3d, dtype=torch.float32).unsqueeze(1).to(device)

    # ==================================================================================================
    # 3. Map the Latent Space (PCA)
    # ==================================================================================================
    print("\nEncoding geometries to Latent Space...")

    all_latents = []
    with torch.no_grad():
        for i in range(len(sdf_tensor)):
            single_geom = sdf_tensor[i:i + 1]
            features = model.encoder(single_geom)
            latent = model.fc1(torch.flatten(features, 1))
            all_latents.append(latent.squeeze().cpu().numpy())

    all_latents = np.array(all_latents)  # Shape: (Num_Geometries, 16)

    pca = PCA(n_components=2)
    latents_2d = pca.fit_transform(all_latents)

    # Plotting
    plt.figure(figsize=(12, 9))
    plt.scatter(latents_2d[:, 0], latents_2d[:, 1], c='crimson', marker='o', s=120, edgecolors='black')

    for i, label in enumerate(file_labels):
        plt.annotate(f"{i}: {label}", (latents_2d[i, 0], latents_2d[i, 1]),
                     xytext=(5, 5), textcoords='offset points', fontsize=9)

    plt.title(f"2D PCA of Latent Space ({sdf_folder_path.split(os.sep)[-1]})", fontsize=14)
    plt.xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)", fontsize=12)
    plt.ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    print("Close the matplotlib window to proceed to 3D extrapolation.")
    plt.show()

    # ==================================================================================================
    # 4. Multi-Mesh Interpolation (Convex Combination)
    # ==================================================================================================

    # Define the geometries you want to blend by their index
    source_indices = [30, 37, 11]

    # Define how much of each geometry to include.
    # CRITICAL: To guarantee topological stability (a single solid body), these must sum to 1.0
    weights = [0.1, 0.6, 0.3]

    if abs(sum(weights) - 1.0) > 1e-6:
        print("\nWARNING: Weights do not sum to 1.0! The geometry may become unstable or disjointed.")

    print(f"\nBlending {len(source_indices)} geometries...")

    # Calculate the weighted linear combination in the 16D space
    z_novel = np.zeros_like(all_latents[0])
    for idx, weight in zip(source_indices, weights):
        z_novel += weight * all_latents[idx]
        print(f"  + {weight * 100}% of [{file_labels[idx]}]")

    with torch.no_grad():
        z_novel_tensor = torch.tensor(z_novel, dtype=torch.float32).unsqueeze(0).to(device)

        # Decode the novel vector back to 3D
        expanded = model.fc2(z_novel_tensor).view(-1, 64, *model.encoded_shape)
        predicted_sdf = model.decoder(expanded).squeeze().cpu().numpy()

        # --- THE FIX: FORCE PERFECT X-AXIS SYMMETRY ---
        # Averages the SDF array with a flipped version of itself along the X-axis (axis=0)
        # If your symmetry is along the Y-axis instead, change axis=0 to axis=1
        if Plane == True:
            predicted_sdf = (predicted_sdf + np.flip(predicted_sdf, axis=1)) / 2.0

        # Extract the mesh
        extrapolated_mesh = extract_mesh(predicted_sdf)

        # Build a safe filename and window title
        combo_name = "_".join([f"s{idx}w{w:.2f}" for idx, w in zip(source_indices, weights)])
        window_title = f"N-Way Hybrid | {combo_name}"

        o3d.visualization.draw_geometries([extrapolated_mesh],
                                          window_name=window_title,
                                          mesh_show_wireframe=True)

        # # --- EXPORT TO MESH FILE ---
        # output_filename = os.path.join(sdf_folder_path, f"novel_mesh_s{source_idx}_t{target_idx}_a{alpha}.obj")
        # o3d.io.write_triangle_mesh(output_filename, extrapolated_mesh)
        # print(f"Novel geometry saved to: {output_filename}")