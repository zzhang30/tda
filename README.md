# Topological Data Analysis (TDA) Project

This repository contains code and resources for performing Topological Data Analysis (TDA) using Python. TDA is a set of techniques that study the shape of data, providing insights into its underlying structure.

## Features
- Implements persistent homology to analyze data shapes.
- Visualizes data structures through persistence diagrams.
- Uses state-of-the-art libraries for TDA computations.

## Directory Structure
- `code/`: Contains the Python scripts and Jupyter notebooks for the analysis.
- `figure/`: Stores generated figures and visualizations.
- `tda.yml`: A YAML file specifying the environment setup.

## Environment Setup
You can set up the environment using `conda`:
conda env create -f tda.yml
conda activate tda

##How to use

### 1. Generate Code Embedding
Run the `dataenginerring.py` script to generate embeddings:

### 2. Calculating persistent homology using rips complexes
Run the `AnalyticalEngineering2.ipynb`(Cynthia's code)

### 3. Build Rips Complex and Visualize
Use `Analysis_ant_class.py` and `Analysis_core_class.py` scripts to perform the following:
Build Rips complexes.
Calculate Betti curves.
Plot and save visualizations.


