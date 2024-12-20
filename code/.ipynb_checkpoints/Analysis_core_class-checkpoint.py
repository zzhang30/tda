import gudhi
import numpy as np
import matplotlib.pyplot as plt
from helper_function import *

def main():
    # Define parameters
    paths = ["ivyClass/core"]
    max_edge = 50 # Maximum edge length for the Rips complex
    max_dim = 2     # Maximum dimension for Betti numbers
    interval = 50 * np.linspace(0, 1, 100)  # Interval for Betti curves

    # Step 1: Extract embeddings
    embeddings = time_execution(extract_embeddings, paths=paths)

    # Step 2: Build the Rips complex
    simplex_tree = time_execution(build_rips_complex, embeddings, max_edge, max_dim=max_dim, verbose=True)

    # Step 3: Calculate Betti curves
    betti_curves = time_execution(calculate_betti_curves, simplex_tree, interval, max_dim=max_dim)

    # Step 4: Visualize Betti curves
    visualize_betti_curves(betti_curves, interval, output_name="ivy_core_class_betti_curves")

    # Step 5: Plot and save barcode
    plot_and_save_barcode(simplex_tree, output_name="ivy_core_class_barcode")

    # Step 6: Plot and save persistence diagram
    plot_and_save_persistence_diagram(simplex_tree, output_name="ivy_core_class_persistence_diagram")

if __name__ == "__main__":
    main()