"""Evaluate and repair a small batch of STL meshes, saving one JSON summary."""

from pathlib import Path
import json
import time

import numpy as np
import trimesh

import evaluation_utils as evaluation
import repair_utils as repair


# ---------------------------------------------------------------------------------------------------- Settings
BASE_FOLDER = Path(__file__).resolve().parent
INPUT_FOLDER = BASE_FOLDER / "test_files"
OUTPUT_FOLDER = BASE_FOLDER / "results" / "jsons" / "evaluation"
N_MESHES = 3
N_POINTS = 2000
SEED = 42


def _numpy_value(value):
    """Convert numpy values encountered by the JSON encoder."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot encode {type(value).__name__} as JSON")


# ---------------------------------------------------------------------------------------------------- Evaluation
def main():
    """
    Load the first N_MESHES STLs and write their before/after measurements.

    Returns
    -------
    output_path : pathlib.Path
        Location of the JSON summary.
    """
    if not isinstance(N_MESHES, int) or N_MESHES < 1:
        raise ValueError("N_MESHES must be a positive integer")
    paths = sorted(p for p in INPUT_FOLDER.rglob("*") if p.suffix.lower() == ".stl")[:N_MESHES]
    if not paths:
        raise ValueError(f"No STL meshes found in {INPUT_FOLDER}")

    summaries = []
    for index, path in enumerate(paths, start=1):
        print(f"[{index}/{len(paths)}] {path.relative_to(INPUT_FOLDER)}", flush=True)
        mesh = trimesh.load_mesh(path, process=False)
        vertices = np.array(mesh.vertices)
        faces = np.array(mesh.faces)
        started = time.perf_counter()

        before = evaluation.mesh_quality(vertices, faces)
        repaired_v, repaired_f, repair_info = repair.repair_mesh(vertices, faces)
        after = evaluation.mesh_quality(repaired_v, repaired_f)
        comparison = evaluation.compare_surfaces(
            vertices, faces, repaired_v, repaired_f, n_points=N_POINTS, seed=SEED
        )

        summaries.append({
            "mesh": str(path.relative_to(INPUT_FOLDER)),
            "before": before,
            "repair": repair_info,
            "after": after,
            "surface_comparison": comparison,
            "elapsed_seconds": time.perf_counter() - started,
        })
        print(f"  vertices: {len(vertices)} -> {len(repaired_v)}; "
              f"faces: {len(faces)} -> {len(repaired_f)}; success: {repair_info['success']}")

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_FOLDER / "summary.json"
    summary = {
        "input_folder": str(INPUT_FOLDER),
        "n_meshes": len(summaries),
        "n_sample_points": N_POINTS,
        "seed": SEED,
        "meshes": summaries,
    }
    output_path.write_text(json.dumps(summary, indent=2, default=_numpy_value, allow_nan=False) + "\n")
    print(f"Summary written to {output_path}")
    return output_path


if __name__ == "__main__":
    main()
