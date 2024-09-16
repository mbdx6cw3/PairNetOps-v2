from openmm.app import *
from openmm import *
from openmm.unit import *
import numpy as np
import read_input, analysis, os,
from network import Network
import tensorflow as tf
import random

def setup():

    input_dir = "md_input"
    md_params = read_input.md(f"{input_dir}/md_params.txt")

    output_dir = "md_data"
    os.makedirs(output_dir)

    temp = md_params["temp"]
    ts = md_params["ts"]
    ensemble = md_params["ensemble"]
    coll_freq = md_params["coll_freq"]
    gro = GromacsGroFile(f"{input_dir}/input.gro")
    top = GromacsTopFile(f"{input_dir}/input.top",
        periodicBoxVectors=gro.getPeriodicBoxVectors())

    # for rigid water to be found the water residue name must be "HOH"
    system = top.createSystem(nonbondedMethod=PME, nonbondedCutoff=1*nanometer,
        constraints=None, removeCMMotion=True,rigidWater=True, switchDistance=None)

    residues = list(top.topology.residues())
    ligand_n_atom = len(list(residues[0].atoms()))

    nb = [f for f in system.getForces() if isinstance(f, NonbondedForce)][0]

    # set exceptions for all ligand atoms
    for i in range(21):
        for j in range(i):
            nb.addException(i, j, 0, 1, 0)

    # create custom force for PairNet predictions
    ml_force = CustomExternalForce("-fx*x-fy*y-fz*z")
    system.addForce(ml_force)
    ml_force.addPerParticleParameter("fx")
    ml_force.addPerParticleParameter("fy")
    ml_force.addPerParticleParameter("fz")
    for j in range(ligand_n_atom):
        ml_force.addParticle(j, (0, 0, 0))

    integrator = LangevinMiddleIntegrator(temp*kelvin,
        coll_freq / picosecond, ts*picoseconds)

    # set up simulation
    simulation = Simulation(top.topology, system, integrator)
    simulation.context.setPositions(gro.positions)

    # select initial velocities from MB distribution
    if ensemble == "nvt":
        simulation.context.setVelocitiesToTemperature(temp*kelvin)

    simulation.reporters.append(StateDataReporter(f"./{output_dir}/openmm.csv",
        reportInterval=1000,step=True, time=True, potentialEnergy=True,
        kineticEnergy=True, temperature=True, separator=" "))

    return simulation, system, md_params, gro, top, ml_force, output_dir

def simulate(simulation, md_params, top, ml_force, output_dir):

    # this is necessary to tell tensorflow to use CPU and GPU when building/predicting
    tf.config.set_visible_devices([], 'GPU')

    mol = read_input.Molecule()
    network = Network(mol)
    input_dir = "trained_model"
    ligand_atoms = np.loadtxt(f"{input_dir}/nuclear_charges.txt", dtype=np.float32).reshape(-1)
    mol.n_atom = len(ligand_atoms)
    model = network.load(mol, input_dir)

    # this prevents tensorflow printing warnings or other information
    tf.get_logger().setLevel('ERROR')

    f6 = open(f"./{output_dir}/ML_forces.txt", 'w')
    f7 = open(f"./{output_dir}/MM_forces.txt", 'w')

    for i in range(1000):

        coords = simulation.context.getState(getPositions=True). \
            getPositions(asNumpy=True).in_units_of(angstrom)

        # clears session to avoid running out of memory
        if (i % 1000) == 0:
            tf.keras.backend.clear_session()

        # predict ML forces: predict_on_batch faster for  single structure
        prediction = model.predict_on_batch([np.reshape(coords
            [:mol.n_atom]/angstrom, (1, -1, 3)),
            np.reshape(ligand_atoms,(1, -1))])
        ML_forces = prediction[0]

        # convert to OpenMM internal units
        ML_forces = np.reshape(ML_forces*kilocalories_per_mole/angstrom, (-1, 3))

        # assign predicted forces to ML atoms
        for j in range(mol.n_atom):
            ml_force.setParticleParameters(j, j, ML_forces[j])
        ml_force.updateParametersInContext(simulation.context)

        # advance trajectory one timestep
        simulation.step(1)

    f6.close()
    f7.close()
    return None


def rmsd_sample(mat_d, rmsd_cut):
    # get distance matrix for this structure and add to dataset
    print_coords = True

    # calculate distance matrix RMSD for last vs all others
    for j in range(mat_d.shape[0] - 1):
        D = analysis.D_rmsd(-1, j, mat_d)
        if D < rmsd_cut:
            print_coords = False
            break

    return print_coords


def permute(ligand_coords, n_perm_grp, perm_atm, n_symm, n_symm_atm):
    # loop through symmetry groups
    for i_perm in range(n_perm_grp):
        # perform 10 swap moves for this symmetry group
        for i_swap in range(10):
            # for this permutation randomly select a symmetry group
            old_perm = perm_atm[i_perm][random.randint(0,n_symm[i_perm]-1)][:]
            new_perm = perm_atm[i_perm][random.randint(0,n_symm[i_perm]-1)][:]
            # swap and save coordinates for these groups
            for i_atm in range(n_symm_atm[i_perm]):
                temp_coord = np.copy(ligand_coords[0, old_perm[i_atm] - 1])
                ligand_coords[0, old_perm[i_atm] - 1] = \
                    ligand_coords[0, new_perm[i_atm] - 1]
                ligand_coords[0, new_perm[i_atm] - 1] = temp_coord

    return ligand_coords


def get_coverage(CV_list, ligand_coords, n_bins, pop):
    dih = np.empty(shape=[CV_list.shape[0]], dtype=int)
    bin_width = 360 / n_bins
    for i_dih in range(CV_list.shape[0]):
        p = np.zeros([CV_list.shape[1], 3])
        p[0:] = ligand_coords[0][CV_list[i_dih][:]]
        dih[i_dih] = int((analysis.dihedral(p) + 180) / bin_width)
        if dih[i_dih] == n_bins:  # this deals with 360 degree angles
            dih[i_dih] = 0
    # populate coverage counter TODO: tidy this up!
    if CV_list.shape[0] == 1:
        pop[dih[0]] += 1
    elif CV_list.shape[0] == 2:
        pop[dih[1]][dih[0]] += 1
    elif CV_list.shape[0] == 3:
        pop[dih[2]][dih[1]][dih[0]] += 1
    elif CV_list.shape[0] == 4:
        pop[dih[3]][dih[2]][dih[1]][dih[0]] += 1
    elif CV_list.shape[0] == 5:
        pop[dih[4]][dih[3]][dih[2]][dih[1]][dih[0]] += 1
    return pop


def predict_charges(md_params, prediction, charge_model, coords, n_atom,
                    ligand_atoms, net_charge):

    # get charge prediction from same network as forces / energies
    if md_params["partial_charge"] == "predicted":
        ligand_charges = prediction[2].T

    # get charge prediction from separate network
    elif md_params["partial_charge"] == "predicted-sep":
        charge_prediction = charge_model.predict_on_batch(
            [np.reshape(coords[:n_atom] / angstrom, (1, -1, 3)),
             np.reshape(ligand_atoms, (1, -1))])
        ligand_charges = charge_prediction[2].T

    # correct predicted partial charges so that ligand has correct net charge
    corr = (sum(ligand_charges) - net_charge) / n_atom
    for atm in range(n_atom):
        ligand_charges[atm] = ligand_charges[atm] - corr

    return ligand_charges


def check_conv(i, conf_cover, conv_time):
    n_surf = conf_cover.shape[0]
    converged = False
    if np.all(conf_cover[:, i] == 100.0):
        converged = True
    else:
        if i > conv_time:
            for i_surf in range(n_surf):
                if (conf_cover[i_surf, i] - conf_cover
                [i_surf, i - conv_time]) < 1.0:
                    converged = True
                else:
                    converged = False
                    break
    return converged

