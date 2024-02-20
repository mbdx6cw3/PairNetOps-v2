#!/usr/bin/env python
import numpy as np
import write_output, os, analysis
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# suppress printing of information messages
tf.get_logger().setLevel("ERROR")

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
        self.au2kcalmola = 627.5095 * 0.529177 #TODO: remove?


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
        # TODO: this can be removed and simplified - make consistent with other
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
        '''
        elif self.charge_scheme == 2: # training on corrected charges
            sum_q = tf.reduce_sum(old_q) / self.n_atoms
            new_q = old_q - sum_q
        elif self.charge_scheme == 3: # training on "interatomic charges"
            # recompose charges here (see analysis.getrecomposedcharges)
            pass
        elif self.charge_scheme == 4: # training on electrostatic energy (WHY?)
            # sum all charge pairs to get electrostatic energy
            # might need to z-normalise electrostatic energies prior to training
            pass
        #K.print_tensor(old_q[0], message="pred_q = ")
        #K.print_tensor(new_q[0], message="corr_q = ")
        '''
        return new_q


class Network(object):
    def __init__(self, molecule):
        self.model = None

    def load(self, mol, ann_params):
        input_dir = "trained_model"
        isExist = os.path.exists(input_dir)
        if not isExist:
            print("ERROR - previously trained model could not be located.")
            exit()
        prescale = np.loadtxt(f"./{input_dir}/prescale.txt", dtype=np.float64).reshape(-1)
        model = Network.build(self, mol, ann_params, prescale)
        model.summary()
        model.load_weights(f"./{input_dir}/best_ever_model").expect_partial()
        return model


    def train(self, model, mol, ann_params, output_dir1, output_dir2):
        # ensures that tensorflow does not use more cores than requested
        NUMCORES = int(os.getenv("NSLOTS", 1))
        sess = tf.compat.v1.Session(
            config=tf.compat.v1.ConfigProto(
                inter_op_parallelism_threads=NUMCORES,
                allow_soft_placement=True, device_count={'CPU': NUMCORES}))
        tf.compat.v1.keras.backend.set_session(sess)

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

        # keras training variables
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
        write_output.twolineplot(np.arange(epochs),np.arange(epochs),model_loss,
            model_val_loss, "training loss", "validation loss", "linear",
            "epoch", "loss", "loss_curve", output_dir1)

        return model

    def test(self, model, mol, output_dir, ann_params, conf_test):
        '''test previously trained ANN'''

        charge_scheme = ann_params["charge_scheme"]

        # define test set
        atoms = np.array([float(i) for i in mol.atoms], dtype='float32')
        test_coords = np.take(mol.coords, mol.test, axis=0)
        test_atoms = np.tile(atoms, (len(test_coords), 1))
        test_prediction = model.predict([test_coords, test_atoms])
        print(f"\nError summary using {len(mol.test)} test structures...")
        print(f"                       MeanAE  |   MaxAE  | L1/L0.01 (%)")
        print(f"--------------------------------------------------")

        # force test output
        test_output_F = np.take(mol.forces, mol.test, axis=0)
        mean_ae, max_ae, L = Network.summary(self, test_output_F.flatten(),
            test_prediction[0].flatten(), output_dir, "f", 1.0)
        print(f"F (kcal mol^-1 A^-1): {mean_ae:7.4f}  | {max_ae:7.4f}  | {L:6.1f} ")
        np.savetxt(f"./{output_dir}/f_test.dat", np.column_stack((
            test_output_F.flatten(), test_prediction[0].flatten())),
            delimiter=" ", fmt="%.6f")

        # test conformational distribution of force errors
        if conf_test:
            CV_list = analysis.getCVs(2)
            analysis.error2D(test_coords, CV_list, output_dir,
                test_output_F, test_prediction[0])

        # energy test output
        test_output_E = np.take(mol.orig_energies, mol.test, axis=0)
        mean_ae, max_ae, L = Network.summary(self, test_output_E.flatten(),
            test_prediction[1].flatten(), output_dir, "e", 1.0)
        print(f"E (kcal mol^-1)     : {mean_ae:7.4f}  | {max_ae:7.4f}  | {L:6.1f} ")
        np.savetxt(f"./{output_dir}/e_test.dat", np.column_stack((
            test_output_E.flatten(), test_prediction[1].flatten())),
            delimiter=", ", fmt="%.6f")

        # charge test output
        test_output_q = np.take(mol.charges, mol.test, axis=0)
        if charge_scheme == 1:
            # get net charge from first structure
            net_charge = np.sum(test_output_q, axis=1)
            corr_prediction = np.zeros((len(test_output_E),mol.n_atom),dtype=float)
            corr = np.zeros([len(test_output_E)])
            for s in range(len(test_output_E)):
                corr[s] = (sum(test_prediction[2][s]) - net_charge[0]) / mol.n_atom
                for atm in range(mol.n_atom):
                    corr_prediction[s][atm] = test_prediction[2][s][atm] - corr[s]
        elif charge_scheme == 2:
            corr_prediction = test_prediction[2][:][:]

        # charge test output
        mean_ae, max_ae, L = Network.summary(self, test_output_q.flatten(),
            corr_prediction.flatten(), output_dir, "q", 0.01)
        print(f"Q (e)               : {mean_ae:7.4f}  | {max_ae:7.4f}  | {L:6.1f} ")
        np.savetxt(f"./{output_dir}/q_test.dat", np.column_stack((
            test_output_q.flatten(), corr_prediction.flatten(),
            test_prediction[2].flatten())), delimiter=" ", fmt="%.6f")

        # electrostatic energy test output
        elec_prediction = analysis.electrostatic_energy(corr_prediction, test_coords)
        test_output_elec = np.take(mol.elec_energies, mol.test, axis=0)
        mean_ae, max_ae, L = Network.summary(self, test_output_elec.flatten(),
            elec_prediction.flatten(), output_dir, "E_elec", 1.0)
        print(f"E_elec (kcal mol^-1): {mean_ae:7.4f}  | {max_ae:7.4f}  | {L:6.1f} ")
        np.savetxt(f"./{output_dir}/elec_test.dat", np.column_stack((
            test_output_elec.flatten(), elec_prediction.flatten())),
            delimiter=" ", fmt="%.6f")

        # calculate electrostatic energy and compare to reference

        return None


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
        # TODO: move charge_scheme options into function
        if charge_scheme == 0 or charge_scheme == 1:
            charge = Q(n_atoms, n_pairs, charge_scheme, name='charge')\
                (output_layer2)
        elif charge_scheme == 2:
            charge = Q(n_atoms, n_pairs, charge_scheme, name ='charge')\
                (output_layer1)
            pass
            # TODO: charge pairs will have to take output_layer1 as input.

        # define the input layers and output layers used in the loss function
        model = Model(
                inputs=[coords_layer, nuclear_charge_layer],
                outputs=[force, energy, charge],
                )

        return model


    def summary(self, all_actual, all_prediction, output_dir, label, val):
        '''Get total errors for array values.'''
        _N = np.size(all_actual)
        mean_ae = 0
        max_ae = 0
        for actual, prediction in zip(all_actual, all_prediction):
            diff = prediction - actual
            mean_ae += np.sum(abs(diff))
            if abs(diff) > max_ae:
                max_ae = abs(diff)
        mean_ae = mean_ae / _N
        L = write_output.scurve(all_actual.flatten(), all_prediction.flatten(),
                      output_dir, f"{label}_scurve", val)
        return mean_ae, max_ae, L

