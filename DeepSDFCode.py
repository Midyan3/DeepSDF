import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import trimesh
from skimage import measure
import matplotlib.pyplot as plt
import glob
import os
import random
from torch.utils.data import Dataset


class DeepSDF(nn.Module):
    def __init__(self, latent_size=128, hidden_dim=256):
        super().__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(3 + latent_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, coords, latent_code):
        latent_expanded = latent_code.expand(coords.size(0), -1)
        inputs = torch.cat([coords, latent_expanded], dim=1)
        sdf = self.fc_layers(inputs)
        return sdf


def sample_sdf(mesh_file, num_samples=50000):
    mesh = trimesh.load(mesh_file)

    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.dump())
    """
    if len(mesh.faces) > 100000:
        print(f"Simplifying complex mesh with {len(mesh.faces)} faces")
        mesh = mesh.simplify_quadratic_decimation(100000)
    
    """

    if mesh.is_empty or mesh.vertices.shape[0] == 0:
        raise ValueError(f"Mesh is empty or invalid: {mesh_file}")

    mesh.apply_translation(-mesh.centroid)
    scale_factor = max(mesh.extents)
    if scale_factor == 0:
        raise ValueError(f"Mesh has zero scale: {mesh_file}")
    mesh.apply_scale(2.0 / scale_factor)

    points_surface, _ = trimesh.sample.sample_surface(mesh, num_samples // 2)

    points_random = np.random.uniform(-1, 1, (num_samples // 2, 3))
    points = np.vstack((points_surface, points_random))

    sdf = mesh.nearest.signed_distance(points)

    return torch.tensor(points, dtype=torch.float32), torch.tensor(
        sdf, dtype=torch.float32
    ).unsqueeze(1)


class SDFDataset(Dataset):
    def __init__(self, shape_files, num_samples=100000):
        self.shape_files = shape_files
        self.num_samples = num_samples

        self.all_points = []
        self.all_sdfs = []

        for idx, mesh_file in enumerate(self.shape_files):
            try:
                points, sdf = sample_sdf(mesh_file, self.num_samples)
                self.all_points.append(points)
                self.all_sdfs.append(sdf)
                print(f"Sampled shape {idx+1}/{len(self.shape_files)}: {mesh_file}")
            except ValueError as e:
                print(f"Warning: Failed to sample shape at index {idx}: {e}")

                self.all_points.append(
                    torch.zeros((self.num_samples, 3), dtype=torch.float32)
                )
                self.all_sdfs.append(
                    torch.zeros((self.num_samples, 1), dtype=torch.float32)
                )

    def __len__(self):
        return len(self.shape_files)

    def __getitem__(self, idx):
        return self.all_points[idx], self.all_sdfs[idx], idx


def load_shape_dataset(dataset_path):
    obj_files = sorted(glob.glob(os.path.join(dataset_path, "*.obj")))
    print(f"Found {len(obj_files)} shape files in {dataset_path}")
    return obj_files


def train_multi_shape(
    model, dataset, latent_codes, epochs=5000, lr=1e-4, save_dir="checkpoints"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = model.to(device)
    latent_codes = latent_codes.to(device)

    optimizer = optim.Adam(
        [{"params": model.parameters()}, {"params": latent_codes.parameters()}], lr=lr
    )

    loss_fn = nn.MSELoss()

    os.makedirs(save_dir, exist_ok=True)

    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for shape_idx in range(len(dataset)):
            points, sdf_gt, _ = dataset[shape_idx]

            points = points.to(device)
            sdf_gt = sdf_gt.to(device)

            latent_code = latent_codes(torch.tensor([shape_idx], device=device))

            optimizer.zero_grad()

            pred_sdf = model(points, latent_code)

            loss = loss_fn(pred_sdf, sdf_gt)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            print(
                f"Epoch {epoch+1}/{epochs}, Shape {shape_idx+1}/{len(dataset)}, Loss: {loss.item():.6f}"
            )

        avg_loss = epoch_loss / len(dataset)

        print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "latent_codes_state_dict": latent_codes.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_loss,
            }
            torch.save(checkpoint, os.path.join(save_dir, "best_model.pth"))
            print(f"Saved new best model with loss {best_loss:.6f}")

        if (epoch + 1) % 500 == 0:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "latent_codes_state_dict": latent_codes.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
            }
            torch.save(
                checkpoint, os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pth")
            )

            if epoch > 0:
                for vis_idx in range(min(3, len(dataset))):
                    latent_code = latent_codes(torch.tensor([vis_idx], device=device))
                    mesh = extract_mesh(model, latent_code, resolution=256)

                    vis_dir = os.path.join(save_dir, f"validation_epoch_{epoch+1}")
                    os.makedirs(vis_dir, exist_ok=True)

                    obj_path = os.path.join(vis_dir, f"shape_{vis_idx}.obj")
                    mesh.export(obj_path)

    torch.save(model.state_dict(), os.path.join(save_dir, "final_model.pth"))
    torch.save(
        latent_codes.state_dict(), os.path.join(save_dir, "final_latent_codes.pth")
    )

    print(f"Training completed. Final model saved to {save_dir}")

    return model, latent_codes


def extract_mesh(model, latent_code, resolution=128, level=0.0, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    latent_code = latent_code.to(device)
    model.eval()

    grid = np.linspace(-1, 1, resolution)
    x, y, z = np.meshgrid(grid, grid, grid)
    points = np.vstack((x.flatten(), y.flatten(), z.flatten())).T

    batch_size = 32768
    num_batches = (points.shape[0] + batch_size - 1) // batch_size

    sdf_values = []
    with torch.no_grad():
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, points.shape[0])

            batch_points = torch.tensor(
                points[start_idx:end_idx], dtype=torch.float32
            ).to(device)
            batch_sdf = model(batch_points, latent_code).cpu().numpy()
            sdf_values.append(batch_sdf)

    sdf_values = np.vstack(sdf_values).reshape(resolution, resolution, resolution)

    try:
        verts, faces, normals, _ = measure.marching_cubes(
            sdf_values, level=level, spacing=(2 / resolution,) * 3
        )
        verts -= 1
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, normals=normals)
        return mesh
    except Exception as e:
        print(f"Error in marching cubes: {e}")
        return None


def extract_mesh_multi_level(model, latent_code, resolution=256, device=None):
    """Try multiple isosurface levels to find the best mesh. This is used to extract the mesh from the SDF values for complex shapes."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    latent_code = latent_code.to(device)
    model.eval()

    grid = np.linspace(-1, 1, resolution)
    x, y, z = np.meshgrid(grid, grid, grid)
    points = np.vstack((x.flatten(), y.flatten(), z.flatten())).T

    batch_size = 32768
    num_batches = (points.shape[0] + batch_size - 1) // batch_size

    sdf_values = []
    with torch.no_grad():
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, points.shape[0])

            batch_points = torch.tensor(
                points[start_idx:end_idx], dtype=torch.float32
            ).to(device)
            batch_sdf = model(batch_points, latent_code).cpu().numpy()
            sdf_values.append(batch_sdf)

    sdf_values = np.vstack(sdf_values).reshape(resolution, resolution, resolution)

    for level in [0.0, 0.01, -0.01, 0.02, -0.02]:
        try:
            verts, faces, normals, _ = measure.marching_cubes(
                sdf_values, level=level, spacing=(2 / resolution,) * 3
            )
            if len(faces) < 100:
                print(
                    f"Level {level} produced only {len(faces)} faces, trying next level"
                )
                continue

            verts -= 1
            mesh = trimesh.Trimesh(vertices=verts, faces=faces, normals=normals)
            print(
                f"Successfully extracted mesh at level {level} with {len(faces)} faces"
            )
            return mesh
        except Exception as e:
            print(f"Failed at level {level}: {e}")
            continue

    return None


def visualize_reconstruction(original_mesh, reconstructed_mesh, save_path=None):
    """Visualize original and reconstructed meshes side by side"""
    scene = trimesh.Scene()

    original_mesh.visual.face_colors = [255, 100, 100, 255]
    original_transform = np.eye(4)
    original_transform[0, 3] = -1.2
    scene.add_geometry(original_mesh, transform=original_transform)

    reconstructed_mesh.visual.face_colors = [100, 100, 255, 255]
    reconstructed_transform = np.eye(4)
    reconstructed_transform[0, 3] = 1.2
    scene.add_geometry(reconstructed_mesh, transform=reconstructed_transform)

    if save_path:
        png = scene.save_image(resolution=(800, 600))
        with open(save_path, "wb") as f:
            f.write(png)
    else:
        scene.show()


def load_checkpoint(model, latent_codes, checkpoint_path):
    """Load model, latent codes, and optimizer state from a checkpoint"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    latent_codes.load_state_dict(checkpoint["latent_codes_state_dict"])

    print(
        f"Loaded checkpoint from epoch {checkpoint['epoch']} with loss {checkpoint['loss']:.6f}"
    )

    return model, latent_codes, checkpoint["optimizer_state_dict"], checkpoint["epoch"]


def process_validation_shape(
    model, latent_codes, shape_idx, mesh_file, resolution=256, save_dir="results"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    latent_codes = latent_codes.to(device)
    model.eval()

    os.makedirs(save_dir, exist_ok=True)

    print(f"Processing validation shape: {mesh_file}")
    try:
        original_mesh = trimesh.load(mesh_file)

        if isinstance(original_mesh, trimesh.Scene):
            original_mesh = trimesh.util.concatenate(original_mesh.dump())
        if not hasattr(original_mesh, "visual") or original_mesh.visual is None:
            original_mesh.visual = trimesh.visual.ColorVisuals(original_mesh)

        latent_code = latent_codes(torch.tensor([shape_idx], device=device))

        reconstructed_mesh = extract_mesh_multi_level(
            model, latent_code, resolution=resolution, device=device
        )

        if reconstructed_mesh is not None:
            obj_path = os.path.join(save_dir, f"reconstructed_{shape_idx}.obj")
            reconstructed_mesh.export(obj_path)

            png_path = os.path.join(save_dir, f"comparison_{shape_idx}.png")
            visualize_reconstruction(original_mesh, reconstructed_mesh, png_path)

            print(
                f"Saved OBJ and PNG for shape {shape_idx} at {obj_path} and {png_path}"
            )
            return True
        else:
            print(f"Failed to extract mesh for shape {shape_idx}")
            return False

    except Exception as e:
        print(f"Error processing shape {shape_idx}: {e}")
        return False


def get_partial_shape(points, sdfs, view="front", ratio=0.5):
    """
    Get a partial shape by keeping only points from a specific view.

    Args:
        points: Full point cloud
        sdfs: SDF values for points
        view: Which part to keep ("front", "back", "left", "right", "top", "bottom")
        ratio: How much of the shape to keep (0-1)

    Returns:
        Partial points and SDFs
    """
    if view == "front":
        mask = points[:, 2] > 0
    elif view == "back":
        mask = points[:, 2] < 0
    elif view == "left":
        mask = points[:, 0] < 0
    elif view == "right":
        mask = points[:, 0] > 0
    elif view == "top":
        mask = points[:, 1] > 0
    elif view == "bottom":
        mask = points[:, 1] < 0
    else:
        raise ValueError(f"Unknown view: {view}")

    indices = np.where(mask)[0]
    if ratio < 1.0:
        num_to_keep = int(len(indices) * ratio)
        indices = np.random.choice(indices, num_to_keep, replace=False)
        new_mask = np.zeros_like(mask)
        new_mask[indices] = True
        mask = new_mask

    return points[mask], sdfs[mask]


def complete_shape(
    model, partial_points, partial_sdfs, num_iterations=1000, lr=1e-3, latent_size=128
):
    """
    Find the optimal latent code for a partial shape observation.

    Args:
        model: Trained DeepSDF model
        partial_points: Tensor of 3D points from partial observation [N, 3]
        partial_sdfs: Tensor of SDF values for those points [N, 1]
        num_iterations: Number of optimization steps
        lr: Learning rate for optimization
        latent_size: Size of latent vector

    Returns:
        Optimized latent code for this partial shape
    """
    device = next(model.parameters()).device

    partial_points = partial_points.to(device)
    partial_sdfs = partial_sdfs.to(device)

    latent_code = torch.zeros(1, latent_size, device=device, requires_grad=True)

    optimizer = optim.Adam([latent_code], lr=lr)

    for i in range(num_iterations):
        optimizer.zero_grad()

        pred_sdf = model(partial_points, latent_code)

        loss = torch.mean(torch.abs(pred_sdf - partial_sdfs))

        reg_loss = 0.01 * torch.sum(latent_code**2)
        total_loss = loss + reg_loss

        total_loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(f"Iteration {i}, Loss: {total_loss.item():.6f}")

    return latent_code.detach()


def get_best_matching_shape(model, partial_points, partial_sdfs, latent_codes, device):
    partial_points = partial_points.to(device)
    partial_sdfs = partial_sdfs.to(device)

    model.eval()
    with torch.no_grad():
        losses = []
        total_shapes = latent_codes.weight.shape[0]

        for idx in range(total_shapes):
            idx_tensor = torch.tensor([idx], device=device)

            lat_code = latent_codes(idx_tensor)

            pred_sdf = model(partial_points, lat_code)

            loss_val = torch.mean(torch.abs(pred_sdf - partial_sdfs)).item()
            losses.append((idx, loss_val))

    losses.sort(key=lambda x: x[1])
    return losses


def test_model_from_checkpoint(
    checkpoint_epoch=None,
    resolution=256,
    test_shape_idx=0,
    view="front",
    obj_path=None,
    is_single_obj=False,
    num_random_shapes=32,
    results_dir=None,
):
    """
    Test model from a specific checkpoint or the final model.

    Args:
        checkpoint_epoch: Specific epoch to load checkpoint from (None for best/final model)
        resolution: Resolution for mesh extraction
        test_shape_idx: Index of shape to test
        view: Which view to use for shape completion test
        obj_path: Path to an OBJ file or directory with OBJs (if None, uses default dataset_path)
        is_single_obj: If True, treats obj_path as single object; if False, treats as directory
        num_random_shapes: Number of random shapes to select if using directory
        results_dir: Custom directory to save results (optional)
    """

    default_dataset_path = "F:/shapeOfChair"  # Default dataset path. Replace with your own path when using this code.
    save_dir = "deepsdf_results_single1chair4973.obj_1" # This as well. Replace with your own path when using this code.
    checkpoint_dir = os.path.join(save_dir, "checkpoints")

    if checkpoint_epoch is not None:
        model_path = os.path.join(
            checkpoint_dir, f"checkpoint_epoch_{checkpoint_epoch}.pth"
        )
        if not os.path.exists(model_path):
            print(f"Checkpoint at epoch {checkpoint_epoch} not found!")
            return False

        latent_path = None
    else:
        model_path = os.path.join(checkpoint_dir, "best_model.pth")
        if not os.path.exists(model_path):
            model_path = os.path.join(checkpoint_dir, "final_model.pth")

        latent_path = os.path.join(checkpoint_dir, "final_latent_codes.pth")
        if not os.path.exists(latent_path):
            latent_path = None

    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}!")
        return False

    latent_size = 128
    hidden_dim = 256
    model = DeepSDF(latent_size=latent_size, hidden_dim=hidden_dim)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    if results_dir is None:
        if checkpoint_epoch is not None:
            out_dir = f"test_results_epoch_{checkpoint_epoch}"
        else:
            out_dir = "test_results_final"
    else:
        out_dir = results_dir

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if is_single_obj and obj_path is not None:
        if not os.path.isabs(obj_path):
            obj_path = os.path.join(os.getcwd(), obj_path)

        if not os.path.exists(obj_path):
            print(f"Single object file not found: {obj_path}")
            return False

        print(f"Processing single object: {obj_path}")

        try:
            points, sdfs = sample_sdf(obj_path)

            partial_points, partial_sdfs = get_partial_shape(points, sdfs, view=view)
            print(
                f"Full shape: {points.shape[0]} points, Partial shape: {partial_points.shape[0]} points"
            )

            latent_code = torch.zeros(1, latent_size, device=device, requires_grad=True)

            print("\nOptimizing latent code for single object...")
            new_latent = complete_shape(
                model,
                partial_points.to(device),
                partial_sdfs.to(device),
                num_iterations=10000,
                latent_size=latent_size,
            )

            completed_mesh = extract_mesh_multi_level(model, new_latent)

            single_obj_dir = os.path.join(out_dir, "single_obj_results")
            os.makedirs(single_obj_dir, exist_ok=True)

            if completed_mesh is not None:
                completed_mesh.export(
                    os.path.join(single_obj_dir, "completion_optimized.obj")
                )
                print(
                    f"Saved completed mesh to {os.path.join(single_obj_dir, 'completion_optimized.obj')}"
                )

                original_mesh = trimesh.load(obj_path)
                if isinstance(original_mesh, trimesh.Scene):
                    original_mesh = trimesh.util.concatenate(original_mesh.dump())

                visualize_reconstruction(
                    original_mesh,
                    completed_mesh,
                    os.path.join(single_obj_dir, "comparison.png"),
                )

                partial_cloud = trimesh.points.PointCloud(partial_points.numpy())
                partial_cloud.export(os.path.join(single_obj_dir, "partial_points.ply"))

                print(
                    f"Single object processing complete. Results saved to {single_obj_dir}"
                )
            else:
                print("Failed to extract mesh for the single object.")

            return True
        except Exception as e:
            print(f"Error processing single object: {e}")
            return False

    dataset_path = obj_path if obj_path is not None else default_dataset_path

    if not os.path.isabs(dataset_path):
        dataset_path = os.path.join(os.getcwd(), dataset_path)

    all_shape_files = load_shape_dataset(dataset_path)

    if not all_shape_files:
        print(f"No OBJ files found in {dataset_path}")
        return False

    if obj_path is not None:
        num_shapes = min(num_random_shapes, len(all_shape_files))
        selected_indices = random.sample(range(len(all_shape_files)), num_shapes)
        shape_files = [all_shape_files[i] for i in selected_indices]

        selected_shapes_file = os.path.join(out_dir, "selected_shapes.txt")
        with open(selected_shapes_file, "w") as f:
            for i, path in zip(selected_indices, shape_files):
                f.write(f"{i}: {path}\n")
    else:
        selected_shapes_file = os.path.join(save_dir, "selected_shapes.txt")
        selected_indices = []

        if os.path.exists(selected_shapes_file):
            with open(selected_shapes_file, "r") as f:
                for line in f:
                    parts = line.strip().split(": ")
                    if len(parts) >= 1:
                        try:
                            idx = int(parts[0])
                            selected_indices.append(idx)
                        except:
                            pass

        if not selected_indices:
            selected_indices = list(range(min(num_random_shapes, len(all_shape_files))))

        shape_files = [all_shape_files[i] for i in selected_indices]

    num_shapes = len(shape_files)
    latent_codes = torch.nn.Embedding(num_shapes, latent_size)

    if latent_path and os.path.exists(latent_path):
        print(f"Loading latent codes from {latent_path}")
        latent_codes.load_state_dict(torch.load(latent_path, map_location=device))
    elif "latent_codes_state_dict" in checkpoint:
        print(f"Extracting latent codes from checkpoint")
        latent_codes.load_state_dict(checkpoint["latent_codes_state_dict"])
    else:
        print("Warning: Could not find latent codes in checkpoint!")
        return False

    latent_codes.to(device)

    test_shape_idx = min(test_shape_idx, num_shapes - 1)
    success = run_shape_completion_test_custom(
        model,
        latent_codes,
        shape_files,
        test_shape_idx=test_shape_idx,
        view=view,
        save_dir=os.path.join(out_dir, f"shape_{test_shape_idx}_{view}"),
    )

    visualization_dir = os.path.join(out_dir, "visualizations")
    os.makedirs(visualization_dir, exist_ok=True)

    for idx in range(min(5, num_shapes)):
        print(f"Extracting mesh for shape {idx}...")
        latent_code = latent_codes(torch.tensor([idx], device=device))

        mesh = extract_mesh_multi_level(
            model, latent_code, resolution=resolution, device=device
        )

        if mesh is not None:
            output_path = os.path.join(visualization_dir, f"shape_{idx}.obj")
            mesh.export(output_path)
            print(f"Saved mesh to {output_path}")
        else:
            print(f"Failed to extract mesh for shape {idx}")

    return True


def run_shape_completion_test_custom(
    model, latent_codes, shape_files, test_shape_idx=0, view="front", save_dir=None
):
    """
    Modified version of run_shape_completion_test that accepts model and latent_codes directly.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if save_dir is None:
        save_dir = f"shape_completion_test_{test_shape_idx}_{view}"
    os.makedirs(save_dir, exist_ok=True)

    print(f"Sampling points from {shape_files[test_shape_idx]}")
    try:
        points, sdfs = sample_sdf(shape_files[test_shape_idx])

        partial_points, partial_sdfs = get_partial_shape(points, sdfs, view=view)
        print(
            f"Full shape: {points.shape[0]} points, Partial shape: {partial_points.shape[0]} points"
        )

        print("\nMethod 1: Finding best matching shape")
        device = next(model.parameters()).device

        top_matches = get_best_matching_shape(
            model, partial_points, partial_sdfs, latent_codes, device
        )

        for i, (match_idx, loss) in enumerate(top_matches[:3]):
            print(f"Match {i+1}: Shape {match_idx}, Loss: {loss:.6f}")

            latent_code = latent_codes(torch.tensor([match_idx], device=device))

            mesh = extract_mesh_multi_level(model, latent_code)

            if mesh is not None:
                mesh.export(
                    os.path.join(save_dir, f"match_{i+1}_shape_{match_idx}.obj")
                )

        print("\nMethod 2: Optimizing new latent code")
        new_latent = complete_shape(
            model,
            partial_points,
            partial_sdfs,
            num_iterations=10000,
            latent_size=latent_codes.weight.shape[1],
        )

        completed_mesh = extract_mesh_multi_level(model, new_latent)

        if completed_mesh is not None:
            completed_mesh.export(os.path.join(save_dir, "completion_optimized.obj"))

        original_mesh = trimesh.load(shape_files[test_shape_idx])
        if isinstance(original_mesh, trimesh.Scene):
            original_mesh = trimesh.util.concatenate(original_mesh.dump())

        partial_cloud = trimesh.points.PointCloud(partial_points.numpy())
        partial_cloud.export(os.path.join(save_dir, "partial_points.ply"))

        if completed_mesh is not None:
            visualize_reconstruction(
                original_mesh,
                completed_mesh,
                os.path.join(save_dir, "comparison_optimized.png"),
            )

        best_match_mesh = extract_mesh_multi_level(
            model, latent_codes(torch.tensor([top_matches[0][0]], device=device))
        )

        if best_match_mesh is not None:
            visualize_reconstruction(
                original_mesh,
                best_match_mesh,
                os.path.join(save_dir, "comparison_best_match.png"),
            )

        print(f"\nShape completion test completed. Results saved to {save_dir}")
        return True

    except Exception as e:
        print(f"Error in shape completion test: {e}")
        return False


if __name__ == "__main__":
    NOT_RECONSTRUCTED = True # When this is true, the code will not reconstruct the shape, but will only test the model on a single shape. When you set this to false, it will reconstruct the shape and save the results. 
    #It should be set to true when you want to test the model on a single shape, and false when you want to reconstruct the shape.

    obj_path = "79571.obj"  # Path to single OBJ or directory with OBJs (can be relative to current dir). Replace with None to use default dataset path. If you have a obj file that
    is_single_obj = True
    num_random_shapes = 1

    if NOT_RECONSTRUCTED:
        dataset_path = "F:/shapeOfChairs" if obj_path is None else obj_path # Path to dataset or single OBJ file. Replace with the path to your dataset or single OBJ file. I used ShapeNet dataset for this and extracted the obj files from it that I wanted to use.

        all_shape_files = load_shape_dataset(dataset_path)

        num_train_shapes = num_random_shapes
        save_dir = f"deepsdf_results_{'single1' + dataset_path if is_single_obj else 'multi'}_{num_train_shapes}"
        os.makedirs(save_dir, exist_ok=True)

        if is_single_obj:
            if not os.path.isabs(obj_path):
                obj_path = os.path.join(os.getcwd(), obj_path)

            if not os.path.exists(obj_path):
                print(f"Single object file not found: {obj_path}")
                exit(1)

            shape_files = [obj_path]

            with open(os.path.join(save_dir, "selected_shapes.txt"), "w") as f:
                f.write(f"0: {obj_path}\n")
        else:
            if len(all_shape_files) < num_train_shapes:
                print(
                    f"Not enough shapes in {dataset_path} (found {len(all_shape_files)})"
                )
                exit(1)

            selected_indices = random.sample(
                range(len(all_shape_files)), num_train_shapes
            )
            shape_files = [all_shape_files[i] for i in selected_indices]

            with open(os.path.join(save_dir, "selected_shapes.txt"), "w") as f:
                for i, path in zip(selected_indices, shape_files):
                    f.write(f"{i}: {path}\n")

        latent_size = 128
        hidden_dim = 512
        model = DeepSDF(latent_size=latent_size, hidden_dim=hidden_dim)

        latent_codes = torch.nn.Embedding(len(shape_files), latent_size)
        torch.nn.init.normal_(latent_codes.weight, mean=0.0, std=0.01)

        dataset = SDFDataset(shape_files, num_samples=50000)

        checkpoint_path = os.path.join(save_dir, "checkpoints", "best_model.pth")
        checkpoint_dir = os.path.join(save_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        resume_training = True
        starting_epoch = 0

        if os.path.exists(checkpoint_path) and resume_training:
            print(f"Found checkpoint at {checkpoint_path}, resuming training...")
            model, latent_codes, optimizer_state, starting_epoch = load_checkpoint(
                model, latent_codes, checkpoint_path
            )

            print(f"Continuing training from epoch {starting_epoch}")
        else:
            print("Starting new training run...")

        model, latent_codes = train_multi_shape(
            model, dataset, latent_codes, epochs=10000, lr=1e-4, save_dir=checkpoint_dir
        )

        validation_dir = os.path.join(save_dir, "final_validation")
        os.makedirs(validation_dir, exist_ok=True)

        print("Performing final validation...")
        for idx in range(len(dataset)):
            process_validation_shape(
                model,
                latent_codes,
                idx,
                shape_files[idx],
                resolution=384,
                save_dir=validation_dir,
            )

        print("Training and validation complete!")
    else:
        checkpoint_epoch = 0
        test_shape_idx = 0
        view = "front"

        results_dir = f"test_results_{'single' if is_single_obj else 'multi'}"

        test_model_from_checkpoint(
            checkpoint_epoch=checkpoint_epoch,
            resolution=384,
            test_shape_idx=test_shape_idx,
            view=view,
            obj_path=obj_path,
            is_single_obj=is_single_obj,
            num_random_shapes=num_random_shapes,
            results_dir=results_dir,
        )

    print("All tasks completed.")


    #Ignore the code below. It is not used in the code above. Here for reference only.
    """
    dataset_path = 'F:/shapeOfChair'
    shape_files = load_shape_dataset(dataset_path)
    

    selected_shapes_file = "deepsdf_results_simple(chairs = 30)/selected_shapes.txt"
    selected_indices = []
    
    if os.path.exists(selected_shapes_file):
        with open(selected_shapes_file, 'r') as f:
            for line in f:
                parts = line.strip().split(': ')
                if len(parts) >= 1:
                    try:
                        idx = int(parts[0])
                        selected_indices.append(idx)
                    except:
                        pass
    
    if not selected_indices:

        selected_indices = list(range(min(30, len(shape_files))))
    
    shape_files = [shape_files[i] for i in selected_indices]
    

    run_shape_completion_test(model_path, latent_path, shape_files, test_shape_idx=0, view="front")
    """
