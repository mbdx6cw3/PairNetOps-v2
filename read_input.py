#!/usr/bin/env python
import numpy as np
from itertools import islice
import analysis

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
        self.elec_energies = other.elec_energies

        if len(other.coords) > 0:
            self.coords = np.reshape(np.vstack(other.coords),
                                     (-1, len(self.atoms), 3))
        if len(other.forces) > 0:
            self.forces = np.reshape(np.vstack(other.forces),
                                     (-1, len(self.atoms), 3))
        if len(other.charges) > 0:
            self.charges = np.reshape(np.vstack(other.charges),
                                     (-1, len(self.atoms)))


class Dataset():
    def __init__(self, mol, tot_size, init, space, input_dir, format):
        print("Reading dataset...")
        if format == "txt" or format == "gau":
            element = {1: "H", 6: "C", 7: "N", 8: "O"} # add more elements
            self.atoms = []
            self.atom_names = []
            input_ = open(f"./nuclear_charges.txt", "r")
            for atom in input_:
                self.atoms.append(int(atom))
                self.atom_names.append(element[self.atoms[-1]])
            self.n_atom = len(self.atoms)
        size = tot_size - init
        if format == "txt":
            self.energies = np.reshape(np.loadtxt(f"./{input_dir}/energies.txt",
                max_rows=size, skiprows=init), (size))
            if len(self.energies) < size:
                print("ERROR - requested set size exceeds the dataset size")
                exit()
            length_check = np.loadtxt(f"./{input_dir}/coords.txt")
            #TODO: this can work but needs to be more robust...
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

        elif format == "gau":

            self.coords, self.energies, self.forces, self.charges, error = \
                gau(size, space, input_dir, self.n_atom)
            if error:
                print("WARNING - some Gaussian jobs did not terminate correctly.")
                exit()

        elif format == "ext":
            try:
                inp_vsn = input("""Enter the dataset version:
                    [1] - original MD17
                    [2] - revised MD17
                    > """)
            except ValueError:
                print("Invalid Value")
                exit()
            if int(inp_vsn) == 1:
                source = "md17"
            elif int(inp_vsn) == 2:
                source = "rmd17"
            else:
                print("Invalid Value")
                exit()

            molecule = int(input("""Enter the molecule name:
                (aspirin, azobenzene, benzene, ethanol, malonaldehyde, 
                 naphthalene, paracetamol, salicylic, toluene, uracial)
                 > """))
            if molecule.strip() == "azobenzene" or molecule == "paracetamol":
                print("Error - molecule not in MD17 dataset")
                exit()

            dataset = np.load(f"{input_dir}/{source}/{source}_{molecule}.npz")

            # slight differences in formatting of md17/rmd17
            if source == "md17":
                self.coords = dataset["R"]
                #self.energies =
                #self.forces =
            elif source == "rmd17":
                self.atoms = dataset["nuclear_charges"]
                self.n_atom = len(self.atoms)
                self.coords = dataset["coords"]
                self.energies = dataset["energies"]
                self.forces = dataset["forces"]
            self.charges = 0.0

        self.elec_energies = analysis.electrostatic_energy(self.charges, self.coords)

        mol.get_ZCFE(self)  # populate molecule class

        return None


def ann(input_file):
    try:
        param_file = open(input_file, "r")
    except FileNotFoundError:
        print("ERROR - no input file in the current working directory")
        exit()
    params = {}
    for line in param_file:
        if line.startswith("#"):
            continue
        line = line.strip()
        key_word = line.split(" = ")
        if len(key_word) == 1:
            continue
        key_word_list = key_word[1].split(", ")
        if len(key_word_list) == 1:
            params[key_word[0].strip()] = key_word_list[0]
        if len(key_word_list) > 1:
            params[key_word[0].strip()] = key_word_list
    param_file.close()

    # check that all input is valid and convert types
    params["activations"] = str(params["activations"])
    accepted_strings = ["silu", "linear"]
    if params["activations"] not in accepted_strings:
        print("***ERROR: activation function type not accepted")
        exit()
    try:
        params["epochs"] = int(params["epochs"])
    except ValueError:
        print("***ERROR: Invalid number of epochs")
        exit()
    try:
        params["n_layers"] = int(params["n_layers"])
    except ValueError:
        print("***ERROR: Invalid number of layers")
        exit()
    params["nodes"] = str(params["n_nodes"])
    accepted_strings = ["auto"]
    if params["n_nodes"] not in accepted_strings:
        try:
            if params["n_layers"] == 1:
                params["n_nodes"] = [int(params["n_nodes"])]
            elif params["n_layers"] > 1:
                params["n_nodes"] = [eval(i) for i in params["n_nodes"]]
        except ValueError:
            print("***ERROR: Invalid number of nodes")
            exit()
    try:
        params["n_data"] = [eval(i) for i in params["n_data"]]
    except ValueError:
        print("***ERROR: Invalid training set size")
    try:
        params["loss_weights"] = [eval(i) for i in params["loss_weights"]]
    except ValueError:
        print("***ERROR: Invalid weights")
        exit()
    try:
        params["init_lr"] = float(params["init_lr"])
    except ValueError:
        print("***ERROR: Invalid initial learning rate")
        exit()
    try:
        params["min_lr"] = float(params["min_lr"])
    except ValueError:
        print("***ERROR: Invalid minimum learning rate")
        exit()
    try:
        params["lr_patience"] = int(params["lr_patience"])
    except ValueError:
        print("***ERROR: Invalid learning rate patience")
        exit()
    try:
        params["lr_factor"] = float(params["lr_factor"])
    except ValueError:
        print("***ERROR: Invalid learning rate factor")
        exit()
    try:
        params["charge_scheme"] = int(params["charge_scheme"])
    except ValueError:
        print("***ERROR: Invalid charge_scheme")
        exit()
    try:
        params["batch_size"] = int(params["batch_size"])
    except ValueError:
        print("***ERROR: Invalid batch size")
        exit()
    params["norm_scheme"] = str(params["norm_scheme"])
    accepted_strings = ["z-score", "force", "none"]
    if params["norm_scheme"] not in accepted_strings:
        print("***ERROR: normalisation scheme not accepted")
        exit()

    return params


def md(input_file):
    try:
        param_file = open(input_file, "r")
    except FileNotFoundError:
        print("***ERROR: no input file in the current working directory")
        exit()
    params = {}
    for line in param_file:
        if line.startswith("#"):
            continue
        line = line.strip()
        key_word = line.split(" = ")
        if len(key_word) == 1:
            continue
        key_word_list = key_word[1].split(", ")
        if len(key_word_list) == 1:
            params[key_word[0].strip()] = key_word_list[0]
        if len(key_word_list) > 1:
            params[key_word[0].strip()] = key_word_list
    param_file.close()
    try:
        params["temp"] = int(params["temp"])
    except ValueError:
        print("***ERROR: Invalid temperature")
        exit()
    params["ensemble"] = str(params["ensemble"])
    accepted_strings = ["nve", "nvt"]
    if params["ensemble"] not in accepted_strings:
        print("***ERROR: ensemble type not accepted")
        exit()
    if params["ensemble"] == "nvt":
        params["thermostat"] = str(params["thermostat"])
        accepted_strings = ["langevin", "nose_hoover"]
        if params["thermostat"] not in accepted_strings:
            print("***ERROR: thermostat type not accepted")
            exit()
        try:
            params["coll_freq"] = float(params["coll_freq"])
        except ValueError:
            print("***ERROR: Invalid temperature")
            exit()
    try:
        params["ts"] = float(params["ts"])
    except ValueError:
        print("***ERROR: Invalid temperature")
        exit()
    try:
        params["n_steps"] = int(params["n_steps"])
    except ValueError:
        print("***ERROR: Invalid number of steps")
        exit()
    try:
        params["print_trj"] = int(params["print_trj"])
    except ValueError:
        print("***ERROR: Invalid printing frequency")
        exit()
    try:
        params["print_summary"] = int(params["print_summary"])
    except ValueError:
        print("***ERROR: Invalid printing frequency")
        exit()
    try:
        params["print_data"] = int(params["print_data"])
    except ValueError:
        print("***ERROR: Invalid printing frequency")
        exit()

    params["partial_charge"] = str(params["partial_charge"]).strip()
    accepted_strings = ["fixed", "predicted"]
    if params["partial_charge"] not in accepted_strings:
        print("***ERROR: charge scheme not accepted.")
        exit()

    if params["minim"].strip() == "False":
        params["minim"] = False
    elif params["minim"].strip() == "True":
        params["minim"] = True

    if params["bias"].strip() == "False":
        params["bias"] = False
    elif params["bias"].strip() == "True":
        params["bias"] = True

    return params


def gau(set_size, set_space, input_dir, n_atom):
    energies = np.empty(shape=[set_size])
    coords = np.empty(shape=[set_size, n_atom, 3])
    forces = np.empty(shape=[set_size, n_atom, 3])
    charges = np.empty(shape=[set_size, n_atom])
    normal_term = np.empty(shape=[set_size], dtype=bool)
    error = False

    # loop over all Gaussian files, extract energies, forces and coordinates
    for i_file in range(set_size):
        if ((i_file) % set_space) == 0:
            normal_term[i_file] = False
            qm_file = open(f"./{input_dir}/mol_{i_file+1}.out", "r")
            for line in qm_file:
                # extract atomic coordinates
                if "Input orientation:" in line:
                    coord_block = list(islice(qm_file, 4+n_atom))[-n_atom:]
                # extract energies, convert to kcal/mol
                if "SCF Done:" in line:
                    energies[i_file] = (float(line.split()[4]))*627.509608
                # extract forces
                if "Axes restored to original set" in line:
                    force_block = list(islice(qm_file, 4+n_atom))[-n_atom:]
                # extract charges
                if "ESP charges:" in line:
                    charge_block = list(islice(qm_file, 1+n_atom))[-n_atom:]
                # assess termination state
                if "Normal termination of Gaussian 09" in line:
                    normal_term[i_file] = True
                    break

            # read atomic coordinates
            for i_atom, atom in enumerate(coord_block):
                coords[i_file, i_atom] = atom.strip('\n').split()[-3:]

            # read atomic forces, convert to kcal/mol/A
            for i_atom, atom in enumerate(force_block):
                forces[i_file, i_atom] = atom.strip('\n').split()[-3:]
                forces[i_file, i_atom] = forces[i_file, i_atom]*627.509608/0.529177

            # read partial charges
            for i_atom, atom, in enumerate(charge_block):
                charges[i_file, i_atom] = atom.strip('\n').split()[-1]

            if not normal_term[i_file]:
                error = True
                print(error)

    return coords, energies, forces, charges, error

def perm(mol):

    with open(f"./permutations.txt", "r") as perm_file:
        max_atm = 10
        max_symm_atm = 10
        n_perm_grp = int(perm_file.readline())
        n_symm = np.zeros(shape=[n_perm_grp], dtype=int)
        n_symm_atm = np.zeros(shape=[n_perm_grp], dtype=int)
        perm_atm = np.zeros(shape=[n_perm_grp, max_symm_atm, max_atm], dtype=int)
        for i_perm in range(0,n_perm_grp):
            n_symm[i_perm] = int(perm_file.readline())
            for i_symm in range(0,n_symm[i_perm]):
                indices = [eval(i) for i in perm_file.readline().split()]
                if i_symm == 0:
                    n_symm_atm[i_perm] = len(indices)
                for i_atm in range(n_symm_atm[i_perm]):
                    perm_atm[i_perm][i_symm][i_atm] = indices[i_atm]
                    if perm_atm[i_perm][i_symm][i_atm] > mol.n_atom:
                        print("Error - permutation atom out of range")
                        exit()
        perm_file.close()

    return n_perm_grp, perm_atm, n_symm, n_symm_atm