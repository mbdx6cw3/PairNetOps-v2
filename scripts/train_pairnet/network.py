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
        tri1 = tf.linalg.band_part(c, -1, 0)
        tri2 = tf.linalg.band_part(tri1, 0, 0)
        tri = tri1 - tri2
        nonzero_indices = tf.where(tf.not_equal(tri, tf.zeros_like(tri)))
        nonzero_values = tf.gather_nd(tri, nonzero_indices)
        nc_flat = tf.reshape(nonzero_values,
                shape=(tf.shape(atom_nc)[0], self._NC2))
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
        diff2 = tf.reduce_sum(diff**2, axis=-1)
        tri = tf.linalg.band_part(diff2, -1, 0)
        nonzero_indices = tf.where(tf.not_equal(tri, tf.zeros_like(tri)))
        nonzero_values = tf.gather_nd(tri, nonzero_indices)
        diff_flat = tf.reshape(nonzero_values, shape=(tf.shape(tri)[0], -1))
        r = diff_flat ** 0.5
        recip_r2 = 1 / r ** 2
        _NRF = (((atom_nc * self.au2kcalmola) * recip_r2) / self.max_NRF) #scaled
        _NRF = tf.reshape(_NRF, shape=(tf.shape(coords)[0], self._NC2))
        return _NRF


class E(Layer):
    def __init__(self, prescale,  **kwargs):
        super(E, self).__init__()
        self.prescale = prescale

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        return (batch_size, 1)

    def call(self, E_scaled):
        E = ((E_scaled - self.prescale[2]) /
            (self.prescale[3] - self.prescale[2]) *
            (self.prescale[1] - self.prescale[0]) + self.prescale[0])
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
                shape=(tf.shape(decomp_scaled)[0], self._NC2))
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
        diff2 = tf.reduce_sum(diff**2, axis=-1)
        tri = tf.linalg.band_part(diff2, -1, 0)
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
        if self.charge_scheme == 1: # training on uncorrected charges
            new_q = old_q
        return new_q


class Network(object):
    def __init__(self, molecule):
        self.model = None


    def load(self, mol, input_dir):
        prescale = np.loadtxt(f"./{input_dir}/prescale.txt", dtype=np.float64).reshape(-1)
        model = Network.build(self, mol, prescale)
        model.summary()
        model.load_weights(f"./{input_dir}/best_ever_model").expect_partial()
        return model


    def train(self, model, mol, output_dir):
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
        file_name=f"./{output_dir}/best_model" # name of model
        monitor_loss="val_loss" # monitor validation loss during training

        lr_factor = 0.5
        lr_patience = 2000
        min_lr = 1e-6
        init_lr = 5e-4
        batch_size = 32
        epochs = 80000
        loss_weight = [0.74, 0.01, 0.25]

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
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

        # train the network
        result = model.fit([train_coords, train_atoms],
            [train_forces, train_energies, train_charges],
            validation_data=([val_coords, val_atoms], [val_forces,
            val_energies,val_charges]), epochs=epochs,
            verbose=2, batch_size=batch_size,callbacks=[mc,rlrop])

        # plot loss curves
        np.savetxt(f"loss.dat", np.column_stack((np.arange
            (epochs), result.history['f_loss'], result.history['e_loss'],
            result.history['q_loss'], result.history['loss'],
            result.history['val_f_loss'], result.history['val_e_loss'],
            result.history['val_q_loss'], result.history['val_loss'] )),
            delimiter=" ", fmt="%.6f")

        return model


    def build(self, mol, prescale):
        '''Input coordinates and z_types into model to get NRFS which then
        are used to predict decompFE, which are then recomposed to give
        Cart Fs and molecular E, both of which could be used in the loss
        function, could weight the E or Fs as required.
        '''
        n_atoms = mol.n_atom

        # set variables
        n_pairs = int(n_atoms * (n_atoms - 1) / 2)
        activations = "silu"
        n_layers = 3
        n_nodes = [360, 360, 360]

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
        energy = E(prescale, name='energy')(E_layer)

        # obtain the forces by taking the gradient of the energy
        force = F(n_atoms, n_pairs, name='force')([energy, coords_layer])

        # predict partial charges
        charge = Q(n_atoms, n_pairs, 1, name='charge')\
            (output_layer2)

        # define the input layers and output layers used in the loss function
        model = Model(
                inputs=[coords_layer, nuclear_charge_layer],
                outputs=[force, energy, charge],
                )

        return model

