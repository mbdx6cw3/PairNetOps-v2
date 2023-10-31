PairNetOps

INTRODUCTION
--------------------------------------------------------------------------------
This package can be used for the following tasks:
1. Prepare input files for MD simulations.
2. Run MD simulations using and empirical or machine learned potential.
3. Analyse MD simulation trajectories.
4. Prepare input from QM calculations from MD simulation output.
5. Convert and Analyse QM calculation data.
6. Train and Test ANNs
7. Query external datasets.
8. Generate torsional scan QM input.

LOCATION
--------------------------------------------------------------------------------
The code for version is located on the CSF here:
    /mnt/iusers01/rb01/mbdx6cw3/bin/PairNetOps/

SETUP
--------------------------------------------------------------------------------
Before running a job, you first need to correctly set up your environment to
run the code. Setup a new Conda environment using the following commands:

1)  Load Anaconda.
    >   module load apps/binapps/anaconda3/2022.10

2)  Create a separate Conda environment for running PairNetOps.
    >   conda create -n pair-net-ops python==3.9.13

3)  Activate the Conda environment.
    >   source activate pair-net-ops

4)  Install Mamba (faster and less buggy than conda for package management).
    Alternative is to use conda to install everything.
    >   conda install -c conda-forge mamba

5)  Install some required basic Python packages.
    >   mamba install matplotlib numpy

6)  Install OpenMM packages and specific version of Cuda (for running MD simulations)
    >   mamba install openmm openmm-plumed openmmtools plumed cudatoolkit=11.8.0

7)  Install Tensorflow (for training, loading and testing PairFENet MLPs)
    a) CPU only install of tensorflow...
        >   pip install tensorflow==2.12.0

    b) GPU install of tensorflow...
        i)      Install tensorflow
        >   pip install --isolated nvidia-cudnn-cu11==8.6.0.163 tensorflow==2.12.0

        ii)     Extra steps to fix bug in tensorflow 2.11 and 2.12.
        >   conda install -c nvidia cuda-nvcc --yes
        >   mkdir -p $CONDA_PREFIX/lib/nvvm/libdevice/
        >   cp -p $CONDA_PREFIX/lib/libdevice.10.bc $CONDA_PREFIX/lib/nvvm/libdevice/

        iii)    Verify that Tensorflow can find the GPUs:
        >   python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

8)  Install OpenMM-ML (needed to simulate with ANI). By default installing
    openmm-ml may install incompatible versions of pytorch (cpu) and openmm-torch (cuda).
    See discussion here: https://github.com/openmm/openmm-torch/issues/94
    Override and enforce cuda installation of pytorch using:
    >   CONDA_OVERRIDE_CUDA="11.8" mamba install openmm-ml pytorch=*=*cuda*

USING THE CODE
--------------------------------------------------------------------------------
Run PairNetOps.
        Interactively:
        > python3 main.py

        From a job script (e.g.):
        > { echo "1"; echo "1"; } | python3 $HOME/bin/PairNetOps/main.py

Many options require the molecule's nuclear_charges.txt file so it's a good idea
to have this wherever you are running the code from.

Current capabilities and input options:

[1] Run a MD simulation using OpenMM. Recommend to submit to batch.
Requires md_params.txt input file and input.gro and input.top gromacs format
coordinate and topology files.
    [1] Use an empirical potential.
    [2] Use a PairFENet trained machine learned potential.
    [3] Use ANI-2x.
        [OpenCL] - run on GPU
        [CPU] - run on CPU

[2] Analyse MD output. Recommended to use interactively.
    [1] - Calculate force S-curve.
    [2] - Calculate force error distribution.
    [3] - Calculate energy correlation.
    [4] - Calculate dihedral angle probability distributions.
    [5] - Calculate 2D free energy surface.

[3] Convert MD output into QM input. Recommended to use interactively.
You will need a gaussian_spe.txt file if generating input files for single
point energy files and a gaussian_opt.txt file if generating input files for
optimisation.

[4] Analyse QM output. Recommended to use interactively.
    [1] - Calculate force and energy probability distributions.
    [2] - Calculate inter-atomic pairwise force components (q).
    [3] - Calculate energy wrt to geometric variable.
    [4] - Calculate distance matrix RMSD.

[5] Convert MD output into QM input. Recommended to use interactively.
    [1] - Convert to ML input.
    You will need a permutations.txt file if doing permutational shuffling.
    [2] - Convert to MD input (.gro format).

[6] Train or Test an ANN. Recommend to submit to batch.
You will need ann_params.txt input file.
    [1] - Train a network.
    [2] - Train and test a network.
    [3] - Load and train a network.
    [4] - Load, train and test a network.
    [5] - Load and test a network.

[7] Query external dataset. Recommended to use interactively.
This tool is used to query the MD17/rMD17 datasets and calculate pairwise
distance, bend angle and dihedral angle distributions between selected atoms.
    [1] - original MD17
    [2] - revised MD17

You will need to consult the relevant mapping.dat file for connectivity.
Outputs: .png image and output.csv file.

[8] Generate torsional scan QM input. Recommended to use interactively.
You will need a mol_1.out input file with the initial structure.

--------------------------------------------------------------------------------