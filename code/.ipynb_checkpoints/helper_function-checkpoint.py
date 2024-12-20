import json
from pathlib import Path
import gudhi
import numpy as np
import matplotlib.pyplot as plt
import time

def extract_embeddings(paths=['ivyClass/ant'], base_dir='/scratch/zzhang30/cs420/data', exclude_files=True):
    """
    Extracts embeddings from JSON files based on the given paths.

    Args:
        paths (list): List of paths within the data directories.
        base_dir (str): Base directory where the data folders are located.
        exclude_files (bool): If True, excludes JSON files not in the specified folder structure.

    Returns:
        np.array: Extracted embeddings as a point cloud.
    """
    embeddings = []

    for path in paths:
        full_path = Path(base_dir) / path
        print(f"Processing path: {full_path}")  # Print the directory being processed

        if not full_path.exists():
            print(f"Warning: Path does not exist - {full_path}")
            continue

        # Traverse all JSON files within the given path
        files = list(full_path.glob('**/*.json'))
        if not files:
            print(f"Warning: No JSON files found in {full_path}")
            continue

        for file in files:
            # Check if the file is within the full_path hierarchy
            if exclude_files and not file.is_relative_to(full_path):
                print(f"Skipping file: {file} (not in {full_path})")
                continue

            print(f"Parsing file: {file}")
            with file.open() as f:
                try:
                    data = json.load(f)
                    embeddings.append(data["embedding"])  
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Error reading file {file}: {e}")
    embeddings = np.asarray(embeddings)
    print(f'Extracted point cloud contains {embeddings.shape[0]} points')
    embeddings = embeddings.reshape(embeddings.shape[0], -1) 
    return embeddings

def build_rips_complex(point_cloud, max_edge, sparse=None, max_dim=3, verbose=False):
    """
    Constructs a Rips complex from a point cloud.

    Args:
        point_cloud (np.array): Input point cloud.
        max_edge (float): Maximum edge length in the Rips complex.
        sparse (float, optional): Sparsification parameter.
        max_dim (int): Maximum dimension for the Rips complex.
        verbose (bool): If True, prints details about the simplex tree.

    Returns:
        gudhi.SimplexTree: Generated simplex tree.
    """
    
    rips = gudhi.RipsComplex(points=point_cloud, max_edge_length=max_edge, sparse=sparse)
    simplex_tree = rips.create_simplex_tree(max_dimension=max_dim)

    print(f"Rips complex: dimension {simplex_tree.dimension()}, simplices {simplex_tree.num_simplices()}, vertices {simplex_tree.num_vertices()}")

    return simplex_tree

def calculate_betti_curves(simplex_tree, interval, max_dim=3):
    """
    Calculates Betti curves from a simplex tree.

    Args:
        simplex_tree (gudhi.SimplexTree): Input simplex tree.
        interval (np.array): Interval for Betti curves.
        max_dim (int): Maximum dimension for Betti curves.

    Returns:
        np.array: Betti curves for each dimension.
    """
    simplex_tree.persistence(persistence_dim_max=True, homology_coeff_field=2)
    diagrams = [simplex_tree.persistence_intervals_in_dimension(i) for i in range(max_dim + 1)]
    betti_curves = []

    step = interval[1] - interval[0]
    for diagram in diagrams:
        curve = np.zeros(len(interval))
        if len(diagram) > 0:
            diagram_intervals = np.clip(np.ceil((diagram[:, :2] - interval[0]) / step), 0, len(interval)).astype(int)
            for start, end in diagram_intervals:
                curve[start:end] += 1
        betti_curves.append(curve)

    return np.reshape(betti_curves, (max_dim+1, len(interval)))

def time_execution(func, *args, **kwargs):
    """
    Measures the execution time of a function.

    Args:
        func (callable): Function to execute.
        *args: Positional arguments for the function.
        **kwargs: Keyword arguments for the function.

    Returns:
        Any: Output of the function.
    """
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    print(f"{func.__name__} executed in {end - start:.4f} seconds")
    return result

def visualize_betti_curves(betti_curves, interval, output_name="betti_curve"):
    """
    Visualizes Betti curves over the specified interval with separate graphs for each dimension.

    Args:
        betti_curves (np.array): Betti curves for each dimension.
        interval (np.array): Interval for visualization.
        output_name (str): Base name for output file.
    """
    fig = plt.figure(figsize=(15, 5))
    for dim in range(len(betti_curves)):
        ax = fig.add_subplot(1, len(betti_curves), dim + 1)
        ax.step(interval, betti_curves[dim])
        ax.set_title(f"Dimension {dim}")
        ax.set_xlabel("Scale")
        ax.set_ylabel("Betti Number")

    output_dir = Path("/scratch/zzhang30/cs420/figure")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"{output_name}.png"
    plt.savefig(output_path)
    print(f"Saved Betti curves to {output_path}")
    plt.show()
    
def plot_and_save_barcode(simplex_tree, output_name="barcode"):
    """
    Plots and saves the barcode from the simplex tree.

    Args:
        simplex_tree (gudhi.SimplexTree): Input simplex tree.
        output_name (str): Name of the output file.
    """
    simplex_tree.persistence()
    gudhi.plot_persistence_barcode(simplex_tree.persistence_intervals_in_dimension(0))
    output_dir = Path("/scratch/zzhang30/cs420/figure")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"{output_name}.png"
    plt.savefig(output_path)
    print(f"Saved barcode plot to {output_path}")
    plt.show()

def plot_and_save_persistence_diagram(simplex_tree, output_name="persistence_diagram"):
    """
    Plots and saves the persistence diagram from the simplex tree.

    Args:
        simplex_tree (gudhi.SimplexTree): Input simplex tree.
        output_name (str): Name of the output file.
    """
    simplex_tree.persistence()
    gudhi.plot_persistence_diagram(simplex_tree.persistence_intervals_in_dimension(0))
    output_dir = Path("/scratch/zzhang30/cs420/figure")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"{output_name}.png"
    plt.savefig(output_path)
    print(f"Saved persistence diagram to {output_path}")
    plt.show()
