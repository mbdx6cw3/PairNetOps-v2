#!/usr/bin/env python
import numpy as np
from itertools import islice
import analysis
import re

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

        elif format == "gau":
            size = tot_size - init
            self.coords, self.energies, self.forces, self.charges, error = \
                gau(size, space, input_dir, self.n_atom)
            if error:
                print("WARNING - some Gaussian jobs did not terminate correctly.")
                #exit()

        elif format == "ext":
            size = tot_size - init
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

            molecule = str(input("""Enter the molecule name:
                (aspirin, azobenzene, benzene, ethanol, malonaldehyde, 
                 naphthalene, paracetamol, salicylic, toluene, uracial)
                 > """))

            dataset = np.load(f"{input_dir}/{source}/{source}_{molecule}.npz")

            # slight differences in formatting of md17/rmd17
            if source == "md17":
                if molecule.strip() == "azobenzene" or molecule == "paracetamol":
                    print("Error - molecule not in MD17 dataset")
                    exit()
                self.coords = dataset["R"]
                #self.energies =
                #self.forces =
            elif source == "rmd17":
                self.atoms = dataset["nuclear_charges"]
                self.n_atom = len(self.atoms)
                self.coords = dataset["coords"]
                self.energies = dataset["energies"]
                self.forces = dataset["forces"]
                self.charges = np.zeros((size, self.n_atom))

        elif format == "pdb":
            self.coords = pdb(input_dir, self.n_atom)
            size = self.coords.shape[0]
            self.energies = np.zeros(size)
            self.forces = np.zeros((size, self.n_atom, 3))
            self.charges = np.zeros((size, self.n_atom))

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
    # TODO: not sure why this is here?
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

    if str(params["loss_weights"]).strip() != "auto":
        try:
            params["loss_weights"] = [eval(i) for i in params["loss_weights"]]
        except ValueError:
            print("***ERROR: weights incorrectly specified")

        if len(params["loss_weights"]) != 3:
            print("***WARNING - loss weights incorrectly specified. Using defaults.")

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
    accepted_strings = ["nve", "nvt", "npt"]
    if params["ensemble"] not in accepted_strings:
        print("***ERROR: ensemble type not accepted")
        exit()
    if params["ensemble"] == "nvt" or params["ensemble"] == "npt":
        params["thermostat"] = str(params["thermostat"])
        accepted_strings = ["langevin", "nose_hoover", "langevin-drude"]
        if params["thermostat"] not in accepted_strings:
            print("***ERROR: thermostat type not accepted")
            exit()
        try:
            params["coll_freq"] = float(params["coll_freq"])
        except ValueError:
            print("***ERROR: Invalid temperature")
            exit()
    if params["ensemble"] == "npt":
        try:
            params["pressure"] = int(params["pressure"])
        except ValueError:
            print("***ERROR: Invalid pressure")
            exit()
    try:
        params["ts"] = float(params["ts"])
    except ValueError:
        print("***ERROR: Invalid temperature")
        exit()
    try:
        params["max_steps"] = int(params["max_steps"])
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
    accepted_strings = ["fixed", "predicted", "predicted-sep"]
    if params["partial_charge"] not in accepted_strings:
        print("***ERROR: charge scheme not accepted.")
        exit()

    if params["partial_charge"] != "fixed":
        try:
            params["charge_scaling"] = float(params["charge_scaling"])
        except ValueError:
            print("***ERROR: Invalid printing frequency")
            exit()

    params["net_charge"] = float(params["net_charge"])

    if params["minim"].strip() == "False":
        params["minim"] = False
    elif params["minim"].strip() == "True":
        params["minim"] = True

    if params["bias"].strip() == "False":
        params["bias"] = False
    elif params["bias"].strip() == "True":
        params["bias"] = True

    if params["force_capping"].strip() == "False":
        params["force_capping"] = False
    elif params["force_capping"].strip() == "True":
        params["force_capping"] = True

    if params["force_capping"]:
        try:
            params["force_cap"] = float(params["force_cap"])
        except ValueError:
            print("***ERROR: Invalid force cap")
            exit()

    if params["adaptive_sampling"].strip() == "False":
        params["adaptive_sampling"] = False
    elif params["adaptive_sampling"].strip() == "True":
        params["adaptive_sampling"] = True
    if params["adaptive_sampling"]:
        try:
            params["rmsd_cut"] = float(params["rmsd_cut"])
            params["n_val"] = int(params["n_val"])
        except ValueError:
            print("***ERROR: Invalid value.")
            exit()

        if params["dynamic_cutoff"].strip() == "False":
            params["dynamic_cutoff"] = False
        elif params["dynamic_cutoff"].strip() == "True":
            params["dynamic_cutoff"] = True

    if params["shuffle_perm"].strip() == "False":
        params["shuffle_perm"] = False
    elif params["shuffle_perm"].strip() == "True":
        params["shuffle_perm"] = True

    if params["cover_conv"].strip() == "False":
        params["cover_conv"] = False
    elif params["cover_conv"].strip() == "True":
        params["cover_conv"] = True

    '''
    if params["background_charges"].strip() == "False":
        params["background_charges"] = False
    elif params["background_charges"].strip() == "True":
        params["background_charges"] = True
    '''

    if params["cover_conv"]:
        try:
            params["n_bin"] = int(params["n_bin"])
            params["conv_time"] = int(params["conv_time"])
            params["cover_surf"] = params["cover_surf"]
            params["cover_surf"] = params["cover_surf"] if type(params["cover_surf"]) \
                is list else [params["cover_surf"]]
        except ValueError:
            print("***ERROR: Invalid value.")
            exit()

    return params

def gau(set_size, set_space, input_dir, n_atom):
    energies = np.empty(shape=[set_size])
    coords = np.empty(shape=[set_size, n_atom, 3])
    forces = np.empty(shape=[set_size, n_atom, 3])
    charges = np.empty(shape=[set_size, n_atom])
    error_term = np.empty(shape=[set_size], dtype=bool)
    error = False
    b3lyp = True
    mp2 = False
    ccsd_t = False

    # loop over all Gaussian files, extract energies, forces and coordinates
    for i_file in range(set_size):
        # energies_found switch is important - allows user to specify
        # whether the last energy is used (optimisation) or first energy (single point)
        energies_found = False
        if ((i_file) % set_space) == 0:
            error_term[i_file] = False
            qm_file = open(f"./{input_dir}/mol_{i_file+1}.out", "r")
            for line in qm_file:
                # extract atomic coordinates
                if "Input orientation:" in line:
                    coord_block = list(islice(qm_file, 4+n_atom))[-n_atom:]
                if not energies_found:
                    # extract energies, convert to kcal/mol
                    if b3lyp:
                        if "SCF Done:  E(RB3LYP) =" in line:
                            energies[i_file] = (float(line.split()[4]))*627.509608
                            energies_found = True
                    elif mp2:
                        if "EUMP2 =" in line:
                            items = line.split()[5]
                            energies[i_file] = float(items.replace("D", "E"))*627.509608
                            energies_found = True
                    elif ccsd_t:
                        if "CCSD(T)= " in line:
                            items = line.split()[1]
                            energies[i_file] = float(
                                items.replace("D", "E")) * 627.509608
                            energies_found = True
                # extract forces
                if "Axes restored to original set" in line:
                    force_block = list(islice(qm_file, 4+n_atom))[-n_atom:]
                # extract ESP charges
                if "ESP charges:" in line or "Hirshfeld charges," in line:
                    charge_block = list(islice(qm_file, 1+n_atom))[-n_atom:]
                # assess termination state
                if "Error termination" in line:
                    error_term[i_file] = True
                    break

            # if no forces it's either an optimisation or an error
            if "force_block" not in locals():
                zero_forces = True
            else:
                zero_forces = False

            # read atomic coordinates
            for i_atom, atom in enumerate(coord_block):
                coords[i_file, i_atom] = atom.strip('\n').split()[-3:]

            # read atomic forces, convert to kcal/mol/A
            # if no forces found set them to zero
            if not zero_forces:
                for i_atom, atom in enumerate(force_block):
                    forces[i_file, i_atom] = atom.strip('\n').split()[-3:]
                    forces[i_file, i_atom] = forces[i_file, i_atom]*627.509608/0.529177
            else:
                forces[i_file] = np.zeros((n_atom, 3))

            # read partial charges
            for i_atom, atom, in enumerate(charge_block):
                charges[i_file, i_atom] = atom.strip('\n').split()[-1]

            if error_term[i_file]:
                error = True
                print(f"Error with file {i_file}")

    return coords, energies, forces, charges, error

def perm(file_name):
    with open(f"./{file_name}", "r") as perm_file:
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
        perm_file.close()

    return n_perm_grp, perm_atm, n_symm, n_symm_atm

def bias():
    plumed_file = open(f"md_input/plumed.dat", "r")
    n_CV = 0
    for line in plumed_file:
        if "TORSION ATOMS=" in line:
            plumed_text = re.split(r",|\n|=", line)
            indices = [eval(i) - 1 for i in plumed_text[-5:-1]]
            n_CV += 1
            if n_CV == 1:
                CV_list = np.empty(shape=[n_CV, 4], dtype=int)
                CV_list[n_CV - 1] = np.array(indices)
            else:
                CV_list = np.append(CV_list, np.reshape(np.array(
                    indices), (1, 4)), axis=0)
            n_CV += 1
    plumed_file.close()
    return CV_list


def fes(input_dir):
    x_count = 0
    y_count = 0
    with open(f"{input_dir}/fes.dat", "r") as input:
        for line in input:
            if line.strip():
                if line.startswith("#"):
                    if "nbins_phi" in line:
                        n_bins = int(line.strip('\n').split()[-1])
                        FE = np.zeros(shape=(n_bins, n_bins))
                    continue
                FE[x_count, y_count] = float(
                    line.strip('\n').split()[2]) / 4.184
                y_count += 1
                if y_count == n_bins:
                    x_count += 1
                    y_count = 0
                if x_count == n_bins:
                    break
    input.close()
    return FE, n_bins


def pdb(input_dir, n_atom):
    pdb_file = open(f"./{input_dir}/conformers.pdb", "r")
    size = 0
    max_size = 100000
    coords = np.zeros((max_size, n_atom, 3))
    # re-order atoms, required because PairNet imposes arbitrary atom ordering
    reorder_atoms = True
    if reorder_atoms:
        atom_mapping = np.loadtxt(f"./{input_dir}/pdb_mapping.dat", dtype=int)
    for line in pdb_file:
        # count number of structures
        if "HEADER" in line:
            coord_block = list(islice(pdb_file, 4 + n_atom))[-n_atom:]
            for i_atom, atom in enumerate(coord_block):
                index = atom_mapping[i_atom]
                coords[size, index] = atom.strip('\n').split()[-6:-3]
            size = size + 1

    return coords[:size]
