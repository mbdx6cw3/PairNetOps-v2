#!/usr/bin/env python

'''
This module is for running a NN with a training set of data.
'''
from __future__ import print_function #for tf printing
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Layer, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
import tensorflow as tf
import sys, time, os
import output, analyseQM

start_time = time.time()

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
        # TODO: this can be removed
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
        #flatten diff2 so that _NC2 values are left
        tri = tf.linalg.band_part(diff2, -1, 0) #lower
        nonzero_indices = tf.where(tf.not_equal(tri, tf.zeros_like(tri)))
        nonzero_values = tf.gather_nd(tri, nonzero_indices)
        diff_flat = tf.reshape(nonzero_values,
                shape=(tf.shape(tri)[0], -1)) #reshape to _NC2
        r = diff_flat ** 0.5
        recip_r2 = 1 / r ** 2
        # TODO: this can be removed and simplified - make consistent with other
        _NRF = (((atom_nc * self.au2kcalmola) * recip_r2) /
                self.max_NRF) #scaled
        _NRF = tf.reshape(_NRF,
                shape=(tf.shape(coords)[0], self._NC2)) #reshape to _NC2
        return _NRF


class Energy(Layer):
    def __init__(self, prescale, **kwargs):
        super(Energy, self).__init__()
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
        decomp_scaled = tf.reshape(decomp_scaled,
                shape=(tf.shape(decomp_scaled)[0], -1))
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
        # flatten diff2 so that _NC2 values are left
        tri = tf.linalg.band_part(diff2, -1, 0) #lower
        nonzero_indices = tf.where(tf.not_equal(tri, tf.zeros_like(tri)))
        nonzero_values = tf.gather_nd(tri, nonzero_indices)
        diff_flat = tf.reshape(nonzero_values,
                shape=(tf.shape(tri)[0], -1)) # reshape to _NC2
        r_flat = diff_flat**0.5
        recip_r_flat = 1 / r_flat
        norm_recip_r = tf.reduce_sum(recip_r_flat ** 2, axis=1,
                keepdims=True) ** 0.5
        eij_E = recip_r_flat / norm_recip_r
        recompE = tf.einsum('bi, bi -> b', eij_E, decompFE)
        recompE = tf.reshape(recompE, shape=(tf.shape(coords)[0], 1))
        return recompE


class Force(Layer):
    def __init__(self, n_atoms, _NC2, **kwargs):
        super(Force, self).__init__()
        self.n_atoms = n_atoms
        self._NC2 = _NC2

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        return (batch_size, self.n_atoms, 3)

    def call(self, E_coords):
        E, coords = E_coords
        gradients = tf.gradients(E, coords, unconnected_gradients='zero')
        return gradients[0] * -1


class Network(object):
    '''
    '''
    def __init__(self, molecule):
        self.model = None

    def train(self, model, mol, ann_params, output_dir1, output_dir2):

        atoms = np.array([float(i) for i in mol.atoms], dtype='float32')

        # prepare training and validation data
        train_coords = np.take(mol.coords, mol.train, axis=0)
        val_coords = np.take(mol.coords, mol.val, axis=0)
        train_energies = np.take(mol.orig_energies, mol.train, axis=0)
        val_energies = np.take(mol.orig_energies, mol.val, axis=0)
        train_forces = np.take(mol.forces, mol.train, axis=0)
        val_forces = np.take(mol.forces, mol.val, axis=0)
        train_output_eij = np.take(mol.output_eij, mol.train, axis=0)
        val_output_eij = np.take(mol.output_eij, mol.val, axis=0)
        #trains_charges = np.take(mol.charges, mol.train, axis=0)
        #val_charges = np.take(mol.charges, mol.val, axis=0)

        # create arrays of nuclear charges for different sets
        train_atoms = np.tile(atoms, (len(train_coords), 1))
        val_atoms = np.tile(atoms, (len(val_coords), 1))

        epochs = ann_params["epochs"]
        loss_weights = ann_params["loss_weights"]
        init_lr = ann_params["init_lr"]
        min_lr = ann_params["min_lr"]
        lr_patience = ann_params["lr_patience"]
        lr_factor = ann_params["lr_factor"]
        file_name=f"./{output_dir2}/best_model" # name of model
        monitor_loss='val_loss' # monitor validation loss during training
        batch_size = 32 # number of structures that are cycled through

        # keras training variables
        mc = ModelCheckpoint(file_name, monitor=monitor_loss, mode='min',
                save_best_only=True, save_weights_only=True)
        rlrop = ReduceLROnPlateau(monitor=monitor_loss, factor=lr_factor,
                patience=lr_patience, min_lr=min_lr)
        optimizer = keras.optimizers.Adam(learning_rate=init_lr,
                beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=False)

        # define loss function, TODO: add charges - replace eij weight
        model.compile(loss={'force': 'mse', 'eij': 'mse', 'energy': 'mse'},
            loss_weights={'force': loss_weights[0], 'eij': loss_weights[1],
            'energy': loss_weights[2]}, optimizer=optimizer)

        # print out the model here
        model.summary()
        print('initial learning rate:', K.eval(model.optimizer.lr))

        # train the network
        result = model.fit([train_coords, train_atoms],
            [train_forces, train_output_eij, train_energies],
            validation_data=([val_coords, val_atoms], [val_forces,
            val_output_eij, val_energies,]), epochs=epochs,
            verbose=2, batch_size=batch_size,callbacks=[mc,rlrop])

        # plot loss curves
        model_loss = result.history['loss']
        model_val_loss = result.history['val_loss']
        np.savetxt(f"./{output_dir1}/loss.dat", np.column_stack((np.arange
            (epochs), result.history['force_loss'], result.history['eij_loss'],
            result.history['energy_loss'], result.history['loss'],
            result.history['val_force_loss'], result.history['val_eij_loss'],
            result.history['val_energy_loss'],result.history['val_loss'] )),
            delimiter=" ", fmt="%.6f")
        output.twolineplot(np.arange(epochs),np.arange(epochs),model_loss,
            model_val_loss, "training loss", "validation loss", "linear",
            "epoch", "loss", "loss_curve", output_dir1)

        return model

    def test(self, model, mol, output_dir):
        '''test previously trained ANN'''

        # define test set
        atoms = np.array([float(i) for i in mol.atoms], dtype='float32')
        test_coords = np.take(mol.coords, mol.test, axis=0)
        test_atoms = np.tile(atoms, (len(test_coords), 1))
        test_output_eij = np.take(mol.output_eij, mol.test, axis=0)
        test_prediction = model.predict([test_coords, test_atoms])

        print(f"\nErrors over {len(mol.test)} test structures")
        print(f"                MAE            RMS            MSD          MSE")

        # q test output
        mae, rms, msd = Network.summary(test_output_eij.flatten(),
                test_prediction[1].flatten())
        print(f"eij: {mae}    {rms}    {msd}    {rms ** 2}")
        output.scurve(test_output_eij.flatten(), test_prediction[1].flatten(),
            output_dir, "eij_scurve")
        np.savetxt(f"./{output_dir}/eij_test.dat", np.column_stack((
            test_output_eij.flatten(), test_prediction[1].flatten())),
            delimiter=" ", fmt="%.6f")

        # force test output
        test_output_F = np.take(mol.forces, mol.test, axis=0)
        mae, rms, msd = Network.summary(test_output_F.flatten(),
                test_prediction[0].flatten())
        print(f"Force:    {mae}    {rms}    {msd}    {rms**2}")
        output.scurve(test_output_F.flatten(), test_prediction[0].flatten(),
            output_dir, "f_scurve")
        np.savetxt(f"./{output_dir}/f_test.dat", np.column_stack((
            test_output_F.flatten(), test_prediction[0].flatten())),
            delimiter=" ", fmt="%.6f")

        # energy test output
        test_output_E = np.take(mol.orig_energies, mol.test, axis=0)
        mae, rms, msd = Network.summary(test_output_E.flatten(),
                test_prediction[2].flatten())
        print(f"Energy:   {mae}    {rms}    {msd}    {rms ** 2}")
        output.scurve(test_output_E.flatten(), test_prediction[2].flatten(),
            output_dir, "e_scurve")
        np.savetxt(f"./{output_dir}/e_test.dat", np.column_stack((
            test_output_E.flatten(), test_prediction[2].flatten())),
            delimiter=", ", fmt="%.6f")

        # charge test output


        return None


    def build(self, n_atoms, ann_params, prescale):
        '''Input coordinates and z_types into model to get NRFS which then
        are used to predict decompFE, which are then recomposed to give
        Cart Fs and molecular E, both of which could be used in the loss
        function, could weight the E or Fs as required.
        '''

        # set variables
        n_pairs = int(n_atoms * (n_atoms - 1) / 2)
        activations = ann_params["activations"]
        n_layers = ann_params["n_layers"]
        n_nodes = ann_params["n_nodes"]
        if ann_params["n_nodes"] == "auto":
            n_nodes = [n_atoms * 30] * n_layers

        # set prescaling factors
        max_NRF = tf.constant(prescale[4], dtype=tf.float32)
        max_matFE = tf.constant(prescale[5], dtype=tf.float32)
        prescale = tf.constant(prescale, dtype=tf.float32)

        coords_layer = Input(shape=(n_atoms, 3), name='coords_layer')
        nuclear_charge_layer = Input(shape=(n_atoms),
            name='nuclear_charge_layer')
        nc_pairs_layer = NuclearChargePairs(n_pairs, n_atoms)(
            nuclear_charge_layer)

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

        # output layer
        connected_layer = Dense(units=n_pairs, activation="linear",
            name='net_layerQ')(connected_layer)

        # calculated unscaled q's
        unscale_qFE_layer = Eij(n_pairs, max_matFE, name='unscale_qF_layer')\
            (connected_layer)

        # calculate the scaled energy from the coordinates and unscaled qFE
        E_layer = ERecomposition(n_atoms, n_pairs, name='qFE_layer')\
            ([coords_layer, unscale_qFE_layer])

        # calculate the unscaled energy
        unscaleE_layer = Energy(prescale, name='unscaleE_layer')(E_layer)

        # obtain the forces by taking the gradient of the energy
        dE_dx = Force(n_atoms, n_pairs, name='dE_dx')\
            ([unscaleE_layer, coords_layer])

        # another layer here for charges summation / removal of excess charge

        # define the input layers and output layers used in the loss function
        model = Model(
                inputs=[coords_layer, nuclear_charge_layer],
                outputs=[dE_dx, unscale_qFE_layer, unscaleE_layer],
                )

        return model


    def summary(all_actual, all_prediction):
        '''Get total errors for array values.'''
        _N = np.size(all_actual)
        mae = 0
        rms = 0
        msd = 0  # mean signed deviation
        for actual, prediction in zip(all_actual, all_prediction):
            diff = prediction - actual
            mae += np.sum(abs(diff))
            rms += np.sum(diff ** 2)
            msd += np.sum(diff)
        mae = mae / _N
        rms = (rms / _N) ** 0.5
        msd = msd / _N
        return mae, rms, msd

