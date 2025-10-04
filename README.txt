PairNetOps

INTRODUCTION
--------------------------------------------------------------------------------
This package can be used for the following tasks:
1. Run MD simulations using empirical or machine learned potentials.
2. Train or Test machine learned potentials according to the PairNet scheme.
3. Analyse datasets.
4. Modify datasets.

LOCATION
--------------------------------------------------------------------------------
The code for version is located on the CSF here:
    /mnt/iusers01/rb01/mbdx6cw3/bin/PairNetOps_v2/

INSTALLATION AND SETUP
--------------------------------------------------------------------------------

1)  Install Mamba - quicker, easier and more robust than conda:
    Mamba installation instructions: https://github.com/conda-forge/miniforge
    wget https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh
    bash Mambaforge-Linux-x86_64.sh ---> yes to initialise at the end (add executables to the path)

2)  Create a separate Mamba environment for running PairNetOps.
    > conda create -n pair-net-ops python==3.9

2)  Activate the Mamba environment.
    > conda activate pair-net-ops

4)  Install CUDA (GPU install only).
    > conda install -c anaconda cudatoolkit==11.8.0

5)  Install Packages
    > conda install -c conda-forge openmm==8.0.0 openmmtools openmm-plumed matplotlib

6)  Install Tensorflow
   CPU install:
    > pip install tensorflow==2.12.0
   GPU install:
    > pip install --isolated nvidia-cudnn-cu11==8.6.0.163 tensorflow==2.12.0
    > mamba install -c nvidia cuda-nvcc --yes
    > mkdir -p $CONDA_PREFIX/lib/nvvm/libdevice/
    > cp -p $CONDA_PREFIX/lib/libdevice.10.bc $CONDA_PREFIX/lib/nvvm/libdevice/
    > python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"


USING THE CODE
--------------------------------------------------------------------------------

Run PairNetOps.
        Interactive:
        > python3 main.py

        From a job script (e.g.):
        > { echo "1"; echo "1"; } | python3 main.py

Available options.
    [1] - Run a Molecular Dynamics Simulation.
            [1] - Use an Empirical Potential.
            [2] - Use PairNet

    [2] - Train or Test a PairNet Potential.
            [1] - Train a network.
            [2] - Train and test a network.
            [3] - Load and train a network.
            [4] - Load, train and test a network.
            [5] - Load and test a network.
            [6] - Load network and predict.

    [3] - Analyse an Existing Dataset.
            [1] - Analyse Forces and Energies.
            [2] - Assess Stability.
            [3] - Analyse Geometry.
                Analysis to Perform:
                [1] - Get energy vs geometric variable.
                [2] - Get root mean squared deviation of distance matrix.
                [3] - Get 1D probability distribution of geometric variable.
                [4] - Get 2D probability distribution of geometric variable.
                [5] - Get 3D probability distribution of geometric variable.
            [4] - Analyse Charges.
                [1] Calculate mean partial charges.
                [2] Calculate partial charge probability distribution.
                [3] Calculate partial charge vs geometric variable.
                [4] Calculate intramolecular electrostatic potential energy.
            [5] - Compare Datasets.
            [6] - Calculate Multidimensional Free Energy Surface.
            [7] - Make Prediction with a PairNet Potential.

            Type of Dataset to Analyse:
                [1] - ml_data (.txt)
                [2] - md_data (.txt)
                [3] - qm_data (.out)
                [4] - External.
                [5] - pdb_file (.pdb)

    [4] - Generate a New Dataset.
             [1] - ...by Dihedral Rotation.
             [2] - ...by Structure Selection using Index List.
             [3] - ...by Structure Selection using Distance Matrix RMSD (D).
             [4] - ...by Random Structure Selection.
             [5] - ...by Merging Two Existing Datasets.
             [6] - ...using CSD System (Mogul).

            Input format:
            [1] - qm_data (.out)
            [2] - ml_data (.txt)
            [3] - md_data (.txt)

    [5] - Reformat an Existing Dataset.
        Input format:
        [1] - qm_data (.out)
        [2] - ml_data (.txt)
        [3] - md_data (.txt)
        [4] - pdb_file (.pdb)

        Output format:
        [1] - qm_data (.gjf)
        [2] - ml_data (.txt)
        [3] - gro_files (.gro)
        [4] - pdb_files (.pdb)
        [5] - MACE (.xyz)

CITATION:   Stable and Accurate Atomistic Simulations of Flexible Molecules using
            Conformationally Generalisable Machine Learned Potentials, CD Williams,
            J Kalayan, NA Burton and RA Bryce, Chem. Sci., 2024, 15, 12780-12795.
--------------------------------------------------------------------------------