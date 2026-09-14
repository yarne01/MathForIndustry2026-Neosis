import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from skimage.measure import marching_cubes
import open3d as o3d
import scipy.ndimage
import os


# ==================================================================================================
# 1. The Dynamic, Shape-Adaptive Architecture
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
    # Light blur helps close Marching Cubes artifacts on thin meshes
    smoothed_sdf = scipy.ndimage.gaussian_filter(sdf_array, sigma=0.5)
    verts, faces, normals, values = marching_cubes(smoothed_sdf, level=0.0)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()
    return mesh


# ==================================================================================================
# 2. Train and Save
# ==================================================================================================
if __name__ == "__main__":
    # --- UPDATE THIS PATH FOR EACH CASE (e.g., planes, valves, etc.) ---
    sdf_folder_path = r"results\sdfs\Sokaris"

    valid_sdfs = []
    target_shape = None

    print(f"Scanning folder: {os.path.abspath(sdf_folder_path)} for geometries...")
    valid_keywords = ('case', 'exp', 'valve')

    for filename in os.listdir(sdf_folder_path):
        lower_name = filename.lower()
        if not lower_name.endswith(".npz"):
            continue

        if any(keyword in lower_name for keyword in valid_keywords):
            filepath = os.path.join(sdf_folder_path, filename)
            try:
                with np.load(filepath) as data:
                    arr = data[data.files[0]]

                if target_shape is None:
                    target_shape = arr.shape
                    print(f"\n  >>> Auto-Locked onto Grid Shape: {target_shape} <<<\n")

                if arr.shape != target_shape:
                    continue

                valid_sdfs.append(arr)
                print(f"  [Included] {filename}")

            except Exception as e:
                pass

    training_sdfs_3d = np.array(valid_sdfs)
    print(f"\nSuccessfully loaded dataset shape: {training_sdfs_3d.shape}")

    sdf_tensor = torch.tensor(training_sdfs_3d, dtype=torch.float32).unsqueeze(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sdf_tensor = sdf_tensor.to(device)

    dataset = TensorDataset(sdf_tensor, sdf_tensor)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = SDFAutoencoder(grid_shape=target_shape, latent_dim=16).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    epochs = 400
    print(f"\nStarting training on {device}...")

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_data, target_data in dataloader:
            optimizer.zero_grad()
            reconstructed = model(batch_data)
            loss = criterion(reconstructed, target_data)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / len(dataloader):.6f}")

    # --- SAVE THE MODEL AND METADATA ---
    save_path = os.path.join(sdf_folder_path, "trained_sdf_ae.pth")
    save_dict = {
        'grid_shape': target_shape,
        'model_state_dict': model.state_dict()
    }
    torch.save(save_dict, save_path)
    print(f"\n[SUCCESS] Model and metadata successfully saved to: {save_path}")

    # --- VISUAL SANITY CHECK (1 Hybrid Mesh) ---
    print("Generating 1 test hybrid to verify network...")
    model.eval()
    with torch.no_grad():
        valve_A = sdf_tensor[0:1]
        valve_B = sdf_tensor[1:2]

        features_A = model.encoder(valve_A)
        latent_A = model.fc1(torch.flatten(features_A, 1))

        features_B = model.encoder(valve_B)
        latent_B = model.fc1(torch.flatten(features_B, 1))

        latent_hybrid = 0.5 * latent_A + 0.5 * latent_B

        expanded = model.fc2(latent_hybrid).view(-1, 64, *model.encoded_shape)
        predicted_sdf = model.decoder(expanded).squeeze().cpu().numpy()

        hybrid_mesh = extract_mesh(predicted_sdf)
        o3d.visualization.draw_geometries([hybrid_mesh], window_name="Sanity Check: Generated Hybrid",
                                          mesh_show_wireframe=True)