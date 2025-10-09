import numpy as np
import os, shutil
from network import Network
from datetime import datetime

def main():

    print("Training a Neural Network using PairNet...")

    # create instance of molecule and network
    mol = Molecule()
    network = Network(mol)

    # define training and test sets.
    n_train, n_val, n_test = 1900, 100, 0
    size = n_train + n_val + n_test

    # locate and read in dataset
    input_dir = "ml_data"
    isExist = os.path.exists(input_dir)
    if not isExist:
        os.makedirs(input_dir)
    mol.dataset(mol, size, 0, 1, input_dir, "txt")
    mol.orig_energies = np.copy(mol.energies)

    # set job flags
    if n_train == 0 or n_val == 0:
        print("ERROR - can't train without a dataset.")

    mol.trainval = [*range(0, n_train + n_val, 1)]
    trainval_forces = np.take(mol.forces, mol.trainval, axis=0)
    trainval_energies = np.take(mol.energies, mol.trainval, axis=0)
    prescale = prescale_e(mol, trainval_energies, trainval_forces)

    # open new directory to save newly trained model
    output_dir = "trained_model"
    isExist = os.path.exists(output_dir)
    if not isExist:
        os.makedirs(output_dir)
    np.savetxt(f"./{output_dir}/nuclear_charges.txt",
        (np.array(mol.atoms)).reshape(-1, 1).astype(int), fmt='%i')

    # pairwise decomposition of energies
    print("Calculating pairwise energies...")
    get_eij(mol, size)

    # get prescaling factors
    prescale = prescale_eij(mol, prescale)
    np.savetxt(f"./{output_dir}/prescale.txt", (np.array(prescale)).reshape(-1, 1))
    print("Building model...")
    model = network.build(mol, prescale)

    # separate training and validation sets and train network
    mol.train = [*range(0, n_train, 1)]
    mol.val = [*range(n_train, n_train + n_val, 1)]
    print("Training model...")
    startTime = datetime.now()
    network.train(model, mol, output_dir)
    print(datetime.now() - startTime)

    print("Saving model...")
    model.save_weights(f"./{output_dir}/best_ever_model")


def prescale_e(mol, energies, forces):
    min_e, max_e = np.min(energies), np.max(energies)
    min_f, max_f = np.min(forces), np.max(forces)
    min_f = np.min(np.abs(forces))
    prescale = [min_e, max_e, min_f, max_f, 0, 0]
    mol.energies = ((max_f-min_f)*(mol.orig_energies-min_e)/(max_e-min_e)+min_f)
    return prescale


def prescale_eij(mol, prescale):
    n_atoms = len(mol.atoms)
    n_pairs = int(n_atoms * (n_atoms - 1) / 2)
    input_NRF = mol.mat_NRF.reshape(-1, n_pairs)
    trainval_input_NRF = np.take(input_NRF, mol.trainval, axis=0)
    trainval_output_eij = np.take(mol.output_eij, mol.trainval, axis=0)
    prescale[4] = np.max(np.abs(trainval_input_NRF))
    prescale[5] = np.max(np.abs(trainval_output_eij))
    return prescale


def get_eij(mol, set_size):
    '''Get decomposed energies and forces from the same simultaneous equation'''

    n_atoms = len(mol.atoms)
    _NC2 = int(n_atoms * (n_atoms - 1) / 2)

    # assign arrays
    mol.mat_NRF = np.zeros((set_size, _NC2))
    mol.mat_r = np.zeros((set_size, _NC2))
    bias = np.zeros((set_size, _NC2))
    mol.output_eij = np.zeros((set_size, _NC2))
    mol.mat_i = np.zeros(_NC2)
    mol.mat_j = np.zeros(_NC2)

    # loop over all structures
    for s in range(set_size):
        _N = -1
        # calculate the distance matrix, r_ij
        for i in range(n_atoms):
            zi = mol.atoms[i]
            for j in range(i):
                _N += 1
                zj = mol.atoms[j]

                if s == 0:
                    mol.mat_i[_N] = i
                    mol.mat_j[_N] = j

                # calculate interatomic distances, save to distance matrix
                r_ij = np.linalg.norm(mol.coords[s][i] - mol.coords[s][j])
                mol.mat_r[s, _N] = r_ij

                # calculate interatomic nuclear repulsion force (input features)
                mol.mat_NRF[s, _N] = get_NRF(zi, zj, r_ij)
                bias[s, _N] = 1 / r_ij

        # calculation normalisation factor, N
        norm_recip_r = 1 / (np.sum(bias[s] ** 2) ** 0.5)

        # normalise  pair energy biases to give dimensionless quantities
        e_ij = bias[s].reshape((1, _NC2)) * norm_recip_r

        # reference energy biases, will be predicted by the trained potential
        mol.energies = np.vstack(mol.energies)
        mol.output_eij[s] = np.matmul(np.linalg.pinv(e_ij), mol.energies[s])

    return None


def get_NRF(zA, zB, r):
    _NRF = r and (zA * zB * np.float64(627.5095 * 0.529177) / (r ** 2))
    return _NRF


class Molecule(object):
    '''
    Base class for coords, forces and energies of an array of molecule
    structures.
    '''

    def get_ZCFE(self, other):
        # For a given set of atom numbers, coords, forces, energies and charges
        # read in, populate the Molecule object which is used to calculate
        # NRFs, decompFE, etc.
        self.atoms = other.atoms
        self.n_atom = other.n_atom
        self.atom_names = other.atom_names
        self.energies = other.energies

        if len(other.coords) > 0:
            self.coords = np.reshape(np.vstack(other.coords),
                                     (-1, len(self.atoms), 3))
        if len(other.forces) > 0:
            self.forces = np.reshape(np.vstack(other.forces),
                                     (-1, len(self.atoms), 3))
        if len(other.charges) > 0:
            self.charges = np.reshape(np.vstack(other.charges),
                                     (-1, len(self.atoms)))


    def dataset(self, mol, tot_size, init, space, input_dir, format):
        print("Reading dataset...")
        #if format == "txt" or format == "gau":
        element = {1: "H", 6: "C", 7: "N", 8: "O"} # add more elements
        self.atoms = []
        self.atom_names = []
        input_ = open(f"./nuclear_charges.txt", "r")
        for atom in input_:
            self.atoms.append(int(atom))
            self.atom_names.append(element[self.atoms[-1]])
        self.n_atom = len(self.atoms)

        if format == "txt":
            size = tot_size - init
            self.energies = np.reshape(np.loadtxt(f"./{input_dir}/energies.txt",
                max_rows=size, skiprows=init), (size))
            if len(self.energies) < size:
                print("ERROR - requested set size exceeds the dataset size")
                exit()
            length_check = np.loadtxt(f"./{input_dir}/coords.txt")
            if (length_check.shape[0]%self.n_atom) != 0:
                print("ERROR - mismatch between molecule size and dataset size.")
                print("Check the nuclear_charges.txt file.")
                exit()
            self.coords = np.reshape(np.loadtxt(f"./{input_dir}/coords.txt",
                max_rows=size*self.n_atom, skiprows=init*self.n_atom),
                (size, self.n_atom, 3))
            self.forces = np.reshape(np.loadtxt(f"./{input_dir}/forces.txt",
                max_rows=size*self.n_atom, skiprows=init*self.n_atom),
                (size, self.n_atom, 3))
            self.charges = np.reshape(np.loadtxt(f"./{input_dir}/charges.txt",
                max_rows=size*self.n_atom, skiprows=init*self.n_atom),
                (size, self.n_atom))

            mol.get_ZCFE(self)  # populate molecule class

        return None


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

