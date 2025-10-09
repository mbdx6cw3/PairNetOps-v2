Scripts to allow user to train a PairFEQ-Net ML model and then use it
to calculate a free energy surface for a charged molecule via MD 
simulations and the balanced ML/MM force field as described in the paper.
The example provided is solute 1 (phenylethylamine).

Creating a Python environment:
conda create -n pair-net-tests python==3.9
conda activate pair-net-tests
conda install openmm==8.2.0 openmmtools openmm-plumed==2.1
pip install tensorflow==2.12.0

Example usage:

[1] Train a PairFEQ-Net ML force field
python train_pairnet.py 

OR

[2] Calculate free energy surface via MD Simulation using balanced ML/MM force field
python calc_fes.py


