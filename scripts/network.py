#!/usr/bin/env python
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # this is to stop the placeholder tensor bug in TF 2.12 - remove later
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Layer
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, Callback
from tensorflow.keras.optimizers import Adam, RMSprop

# suppress printing of general information messages
tf.get_logger().setLevel("ERROR")
print("Tensorflow version:", tf.__version__)

class NuclearChargePairs(Layer):
    def __init__(self, _NC2, n_atoms, **kwargs):
        super(NuclearChargePairs, self).__init__()
        self._NC2 = _NC2
        self.n_atoms = n_atoms

    def call(self, atom_nc):
        a = tf.expand_dims(atom_nc, 2)
        b = tf.expand_dims(atom_nc, 1)
        c = a * b
        tri1 = tf.linalg.band_part(c, -1, 0) #lower
        tri2 = tf.linalg.band_part(tri1, 0, 0) #lower
        tri = tri1 - tri2
        nonzero_indices = tf.where(tf.not_equal(tri, tf.zeros_like(tri)))
        nonzero_values = tf.gather_nd(tri, nonzero_indices)
        nc_flat = tf.reshape(nonzero_values,
                shape=(tf.shape(atom_nc)[0], self._NC2)) #reshape to _NC2
        return nc_flat


class CoordsToNRF(Layer):
    def __init__(self, max_NRF, _NC2, n_atoms, **kwargs):
        super(CoordsToNRF, self).__init__()
        self.max_NRF = max_NRF
        self._NC2 = _NC2
        self.n_atoms = n_atoms
        self.au2kcalmola = 627.5095 * 0.529177


    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        n_atoms = input_shape[1]
        return (batch_size, n_atoms, self._NC2)


    def call(self, coords_nc):
        coords, atom_nc = coords_nc
        a = tf.expand_dims(coords, 2)
        b = tf.expand_dims(coords, 1)
        diff = a - b
        diff2 = tf.reduce_sum(diff**2, axis=-1) #get sqrd diff
        tri = tf.linalg.band_part(diff2, -1, 0) #lower
        nonzero_indices = tf.where(tf.not_equal(tri, tf.zeros_like(tri)))
        nonzero_values = tf.gather_nd(tri, nonzero_indices)
        diff_flat = tf.reshape(nonzero_values, shape=(tf.shape(tri)[0], -1))
        r = diff_flat ** 0.5
        recip_r2 = 1 / r ** 2
        _NRF = (((atom_nc * self.au2kcalmola) * recip_r2) / self.max_NRF) #scaled
        _NRF = tf.reshape(_NRF, shape=(tf.shape(coords)[0], self._NC2))
        return _NRF


class E(Layer):
    def __init__(self, prescale, norm_scheme, **kwargs):
        super(E, self).__init__()
        self.prescale = prescale
        self.norm_scheme = norm_scheme

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        return (batch_size, 1)

    def call(self, E_scaled):
        if self.norm_scheme == "z-score":
            E = E_scaled * self.prescale[1] + self.prescale[0]
        elif self.norm_scheme == "force":
            E = ((E_scaled - self.prescale[2]) /
                (self.prescale[3] - self.prescale[2]) *
                (self.prescale[1] - self.prescale[0]) + self.prescale[0])
        elif self.norm_scheme == "none":
            E = E_scaled
        return E


class Eij(Layer):
    def __init__(self, _NC2, max_Eij, **kwargs):
        super(Eij, self).__init__()
        self._NC2 = _NC2
        self.max_Eij = max_Eij

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        return (batch_size, self._NC2)

    def call(self, decomp_scaled):
        decomp_scaled = tf.reshape(decomp_scaled, shape=(tf.shape(decomp_scaled)[0], -1))
        decomp = (decomp_scaled - 0.5) * (2 * self.max_Eij)
        decomp = tf.reshape(decomp,
                shape=(tf.shape(decomp_scaled)[0], self._NC2)) #reshape to _NC2
        return decomp


class ERecomposition(Layer):
    def __init__(self, n_atoms, _NC2, **kwargs):
        super(ERecomposition, self).__init__()
        self.n_atoms = n_atoms
        self._NC2 = _NC2

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        return (batch_size, 1)

    def call(self, coords_decompFE):
        coords, decompFE = coords_decompFE
        decompFE = tf.reshape(decompFE, shape=(tf.shape(decompFE)[0], -1))
        a = tf.expand_dims(coords, 2)
        b = tf.expand_dims(coords, 1)
        diff = a - b
        diff2 = tf.reduce_sum(diff**2, axis=-1) # get sqrd diff
        tri = tf.linalg.band_part(diff2, -1, 0) #lower
        nonzero_indices = tf.where(tf.not_equal(tri, tf.zeros_like(tri)))
        nonzero_values = tf.gather_nd(tri, nonzero_indices)
        diff_flat = tf.reshape(nonzero_values, shape=(tf.shape(tri)[0], -1))
        r_flat = diff_flat**0.5
        recip_r_flat = 1 / r_flat
        norm_recip_r = tf.reduce_sum(recip_r_flat ** 2, axis=1,keepdims=True)**0.5
        eij_E = recip_r_flat / norm_recip_r
        recompE = tf.einsum('bi, bi -> b', eij_E, decompFE)
        recompE = tf.reshape(recompE, shape=(tf.shape(coords)[0], 1))
        return recompE


class F(Layer):
    def __init__(self, n_atoms, _NC2, **kwargs):
        super(F, self).__init__()
        self.n_atoms = n_atoms
        self._NC2 = _NC2

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        return (batch_size, self.n_atoms, 3)

    def call(self, E_coords):
        E, coords = E_coords
        gradients = tf.gradients(E, coords, unconnected_gradients='zero')
        return gradients[0] * -1


class Q(Layer):
    def __init__(self, n_atoms, n_pairs, charge_scheme, **kwargs):
        super(Q, self).__init__()
        self.n_atoms = n_atoms
        self.n_pairs = n_pairs
        self.charge_scheme = charge_scheme

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        return (batch_size, self.n_atoms)

    def call(self, old_q):
        # calculate corrected charges by subtracting net charge from all predicted charges
        if self.charge_scheme == 1: # training on uncorrected charges
            new_q = old_q
        return new_q


class Network(object):
    def __init__(self, molecule):
        self.model = None

    def load(self, mol, input_dir):
        ann_params = self.ann(f"{input_dir}/ann_params.txt")
        prescale = np.loadtxt(f"./{input_dir}/prescale.txt", dtype=np.float64).reshape(-1)
        model = Network.build(self, mol, ann_params, prescale)
        model.summary()
        model.load_weights(f"./{input_dir}/best_ever_model").expect_partial()
        return model


    def train(self, model, mol, ann_params, output_dir1, output_dir2):

        # prepare training and validation data
        atoms = np.array([float(i) for i in mol.atoms], dtype='float32')
        train_coords = np.take(mol.coords, mol.train, axis=0)
        val_coords = np.take(mol.coords, mol.val, axis=0)
        train_energies = np.take(mol.orig_energies, mol.train, axis=0)
        val_energies = np.take(mol.orig_energies, mol.val, axis=0)
        train_forces = np.take(mol.forces, mol.train, axis=0)
        val_forces = np.take(mol.forces, mol.val, axis=0)
        train_charges = np.take(mol.charges, mol.train, axis=0)
        val_charges = np.take(mol.charges, mol.val, axis=0)

        # create arrays of nuclear charges for different sets
        train_atoms = np.tile(atoms, (len(train_coords), 1))
        val_atoms = np.tile(atoms, (len(val_coords), 1))

        # ann parameters
        # TODO: change to max epochs?
        epochs = ann_params["epochs"]
        init_lr = ann_params["init_lr"]
        min_lr = ann_params["min_lr"]
        lr_patience = ann_params["lr_patience"]
        lr_factor = ann_params["lr_factor"]
        batch_size = ann_params["batch_size"]
        file_name=f"./{output_dir2}/best_model" # name of model
        monitor_loss="val_loss" # monitor validation loss during training

        loss_weights = ann_params["loss_weights"]
        if loss_weights == "auto":
            tot = 4 * mol.n_atom + 1
            loss_weight = [3 * mol.n_atom / tot, mol.n_atom / tot, 1 / tot]
        else:
            loss_weight = loss_weights

        # keras training variables and callbacks
        mc = ModelCheckpoint(file_name, monitor=monitor_loss, mode='min',
                save_best_only=True, save_weights_only=True)
        rlrop = ReduceLROnPlateau(monitor=monitor_loss, factor=lr_factor,
                patience=lr_patience, min_lr=min_lr)
        optimizer = Adam(learning_rate=init_lr,
                beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=False)

        # define loss function
        model.compile(loss={'f': 'mse', 'e': 'mse', 'q': 'mse'},
            loss_weights={'f': loss_weight[0], 'e': loss_weight[1], 'q':
                          loss_weight[2]},optimizer=optimizer)

        # print out the model here
        model.summary()
        print("Initial learning rate:", init_lr)
        # check and print out availability of GPUs
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

        # train the network
        result = model.fit([train_coords, train_atoms],
            [train_forces, train_energies, train_charges],
            validation_data=([val_coords, val_atoms], [val_forces,
            val_energies,val_charges]), epochs=epochs,
            verbose=2, batch_size=batch_size,callbacks=[mc,rlrop])

        # plot loss curves
        model_loss = result.history['loss']
        model_val_loss = result.history['val_loss']
        np.savetxt(f"./{output_dir1}/loss.dat", np.column_stack((np.arange
            (epochs), result.history['f_loss'], result.history['e_loss'],
            result.history['q_loss'], result.history['loss'],
            result.history['val_f_loss'], result.history['val_e_loss'],
            result.history['val_q_loss'], result.history['val_loss'] )),
            delimiter=" ", fmt="%.6f")

        return model


    def build(self, mol, ann_params, prescale):
        '''Input coordinates and z_types into model to get NRFS which then
        are used to predict decompFE, which are then recomposed to give
        Cart Fs and molecular E, both of which could be used in the loss
        function, could weight the E or Fs as required.
        '''
        n_atoms = mol.n_atom

        # set variables
        n_pairs = int(n_atoms * (n_atoms - 1) / 2)
        activations = ann_params["activations"]
        n_layers = ann_params["n_layers"]
        n_nodes = ann_params["n_nodes"]
        charge_scheme = ann_params["charge_scheme"]
        norm_scheme = ann_params["norm_scheme"]
        if ann_params["n_nodes"] == "auto":
            n_nodes = [n_atoms * 30] * n_layers

        # set prescaling factors
        max_NRF = tf.constant(prescale[4], dtype=tf.float32)
        max_matFE = tf.constant(prescale[5], dtype=tf.float32)
        prescale = tf.constant(prescale, dtype=tf.float32)

        # create input layer tensors
        coords_layer = Input(shape=(n_atoms, 3), name='coords_layer')
        nuclear_charge_layer = Input(shape=(n_atoms), name='nuclear_charge_layer')
        nc_pairs_layer = NuclearChargePairs(n_pairs, n_atoms)(nuclear_charge_layer)

        # calculate scaled NRFs from coordinates and nuclear charges
        NRF_layer = CoordsToNRF(max_NRF, n_pairs, n_atoms, name='NRF_layer')\
                ([coords_layer, nc_pairs_layer])

        # define input layer as the NRFs
        connected_layer = NRF_layer

        # loop over hidden layers
        for l in range(n_layers):
            net_layer = Dense(units=n_nodes[l], activation=activations,
                name='net_layerA{}'.format(l))(connected_layer)
            connected_layer = net_layer

        # output layer for interatomic pairwise energy components
        output_layer1 = Dense(units=n_pairs, activation="linear",
            name='net_layer_n_pair')(connected_layer)

        # output layer for uncorrected predicted charges
        output_layer2 = Dense(units=n_atoms, activation="linear",
            name='net_layer_n_atm')(connected_layer)

        # calculated unscaled interatomic energies
        unscale_E_layer = Eij(n_pairs, max_matFE, name='unscale_E_layer')\
            (output_layer1)

        # calculate the scaled energy from the coordinates and unscaled qFE
        E_layer = ERecomposition(n_atoms, n_pairs, name='E_layer')\
            ([coords_layer, unscale_E_layer])

        # calculate the unscaled energy
        energy = E(prescale, norm_scheme, name='energy')(E_layer)

        # obtain the forces by taking the gradient of the energy
        force = F(n_atoms, n_pairs, name='force')([energy, coords_layer])

        # predict partial charges
        if charge_scheme == 1:
            charge = Q(n_atoms, n_pairs, charge_scheme, name='charge')\
                (output_layer2)
        elif charge_scheme == 2:
            charge = Q(n_atoms, n_pairs, charge_scheme, name ='charge')\
                (output_layer2)

        # define the input layers and output layers used in the loss function
        model = Model(
                inputs=[coords_layer, nuclear_charge_layer],
                outputs=[force, energy, charge],
                )

        return model


    def predict(self, model, mol, indices):
        # define test set
        atoms = np.array([float(i) for i in mol.atoms], dtype='float32')
        coords = np.take(mol.coords, indices, axis=0)
        atoms = np.tile(atoms, (len(coords), 1))
        prediction = model.predict([coords, atoms])

        # TODO: currently for zero total charge
        corr = np.zeros([len(prediction[0])])
        # calculate predicted net charge of each test structure
        for s in range(len(prediction[0])):
            corr[s] = (sum(prediction[2][s]) - 0) / mol.n_atom
            for atm in range(mol.n_atom):
                prediction[2][s][atm] = prediction[2][s][atm] - corr[s]

        return prediction


    def ann(self, input_file):
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

        if str(params["loss_weights"]).strip() != "auto":
            try:
                params["loss_weights"] = [eval(i) for i in params["loss_weights"]]
            except ValueError:
                print("***ERROR: weights incorrectly specified")

            if len(params["loss_weights"]) != 3:
                print(
                    "***WARNING - loss weights incorrectly specified. Using defaults.")

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


    def md(self, file_name):
        try:
            param_file = open(file_name, "r")
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

        if params["cover_conv"]:
            try:
                params["n_bin"] = int(params["n_bin"])
                params["conv_time"] = int(params["conv_time"])
                params["cover_surf"] = params["cover_surf"]
                params["cover_surf"] = params["cover_surf"] if type(
                    params["cover_surf"]) is list else [params["cover_surf"]]
            except ValueError:
                print("***ERROR: Invalid value.")
                exit()

        return params

