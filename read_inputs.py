#!/usr/bin/env python
import numpy as np
import output

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
        if len(other.coords) > 0:
            self.coords = np.reshape(np.vstack(other.coords),
                                     (-1, len(self.atoms), 3))
        if len(other.forces) > 0:
            self.forces = np.reshape(np.vstack(other.forces),
                                     (-1, len(self.atoms), 3))
        if len(other.charges) > 0:
            self.charges = np.reshape(np.vstack(other.charges),
                                     (-1, len(self.atoms)))
        if len(other.energies) > 0:
            self.energies = np.vstack(other.energies)

class dataset():
    def __init__(self, mol, input_dir, set_size, read_charge):
        file_list = ["./nuclear_charges.txt", f"./{input_dir}/coords.txt",
            f"./{input_dir}/forces.txt", f"./{input_dir}/energies.txt",
            f"./{input_dir}/charges.txt"]
        element = {1: "H", 6: "C", 7: "N", 8: "O"}
        self.atoms = []
        self.atom_names = []
        input_ = open(file_list[0], 'r')
        for atom in input_:
            self.atoms.append(int(atom))
            self.atom_names.append(element[self.atoms[-1]])
        self.n_atom = len(self.atoms)
        self.charges = np.zeros((set_size, self.n_atom))
        self.coords = np.reshape(np.loadtxt(file_list[1], max_rows=set_size
            * self.n_atom), (set_size, self.n_atom, 3))
        self.energies = np.reshape(np.loadtxt(file_list[3], max_rows=set_size),
            (set_size))
        self.forces = np.reshape(np.loadtxt(file_list[2], max_rows=set_size
            * self.n_atom), (set_size, self.n_atom, 3))
        if read_charge:
            self.charges = np.reshape(np.loadtxt(file_list[4], max_rows=set_size
                * self.n_atom), (set_size, self.n_atom))
        mol.get_ZCFE(self)  # populate molecule class

        return None


def ann(input_file):
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
    if(params["minim"]) == "False":
        params["minim"] = False
    elif(params["minim"]) == "True":
        params["minim"] = True
    if(params["bias"]) == "False":
        params["bias"] = False
    elif(params["bias"]) == "True":
        params["bias"] = True

    return params

