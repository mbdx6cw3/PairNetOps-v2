from openmm.app import *
from openmm import *
from openmm.unit import *
from openmmplumed import PlumedForce
from openmmtools import integrators
import numpy as np
import write_output, read_input, analysis, os, shutil
from network import Network
import tensorflow as tf
import random, math

def setup(force_field):

    input_dir = "md_input"
    isExist = os.path.exists(input_dir)
    if not isExist:
        print("Error - no input files detected")
        exit()
    md_params = read_input.md(f"{input_dir}/md_params.txt")

    output_dir = "md_data"
    isExist = os.path.exists(output_dir)
    if isExist:
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    temp = md_params["temp"]
    ts = md_params["ts"]
    ensemble = md_params["ensemble"]
    thermostat = md_params["thermostat"]
    minim = md_params["minim"]
    coll_freq = md_params["coll_freq"]
    gro = GromacsGroFile(f"{input_dir}/input.gro")
    top = GromacsTopFile(f"{input_dir}/input.top",
        periodicBoxVectors=gro.getPeriodicBoxVectors())

    # for rigid water to be found the water residue name must be "HOH"
    system = top.createSystem(nonbondedMethod=PME, nonbondedCutoff=1*nanometer,
        constraints=None, removeCMMotion=True,rigidWater=True, switchDistance=None)

    #ewaldErrorTolerance=0.0005,

    print("Checking simulation setup...")
    print()
    print("Total number of atoms :", top.topology.getNumAtoms())
    print("Total number of residues :", top.topology.getNumResidues())
    print("Total number of bonds :", top.topology.getNumBonds())
    print("Total number of constraints :", system.getNumConstraints())
    print()
    residues = list(top.topology.residues())
    print("Ligand residue name: ", residues[0].name)
    ligand_n_atom = len(list(residues[0].atoms()))
    print("Number of atoms in ligand: ", ligand_n_atom)
    print("Number of bonds in ligand: ", len(list(residues[0].bonds())))

    nb = [f for f in system.getForces() if isinstance(f, NonbondedForce)][0]
    ewald_tol = nb.getEwaldErrorTolerance()
    [pme_alpha, pme_nx, pme_ny, pme_nz] = nb.getPMEParameters() # TODO: why does this return 0s?
    nb_cut = nb.getCutoffDistance()
    alpha_ewald = (1.0 / nb_cut) * np.sqrt(-np.log(2.0 * ewald_tol))
    print("Periodic boundary conditions? ", system.usesPeriodicBoundaryConditions())
    print("Box dimensions: ", gro.getUnitCellDimensions())
    print("Non-bonded method: ", nb.getNonbondedMethod())
    print("Non-bonded cut-off distance: ", nb_cut)
    print("Ewald sum error tolerance (units?) :", ewald_tol)
    print("PME separation parameter: ", pme_alpha)
    print("Number of PME grid points along each axis: ", pme_nx, pme_ny, pme_nz)
    print("Ewald Gaussian width: ", alpha_ewald)

    if force_field == "empirical":
        ml_force = None

    if force_field == "pair_net":
        # set exceptions for all ligand atoms
        solv_except = nb.getNumExceptions()
        print("Number of solvent exceptions: ", solv_except)
        for i in range(ligand_n_atom):
            for j in range(i):
                nb.addException(i, j, 0, 1, 0)
        print("Number of ligand exceptions: ", nb.getNumExceptions()-solv_except)
        print("Total number of exceptions: ", nb.getNumExceptions())

        # create custom force for PairNet predictions
        ml_force = CustomExternalForce("-fx*x-fy*y-fz*z")
        system.addForce(ml_force)
        ml_force.addPerParticleParameter("fx")
        ml_force.addPerParticleParameter("fy")
        ml_force.addPerParticleParameter("fz")
        for j in range(ligand_n_atom):
            ml_force.addParticle(j, (0, 0, 0))

    # define ensemble, thermostat and integrator
    if ensemble == "nve":
        integrator = VerletIntegrator(ts*picoseconds)
    elif ensemble == "nvt":
        if thermostat == "nose_hoover":
            integrator = integrators.NoseHooverChainVelocityVerletIntegrator\
                (system, temp*kelvin, coll_freq / picosecond, ts*picoseconds, 10, 5, 5)
            if force_field == "pair_net":
                print("WARNING - are you sure you want to use Nose Hoover with pair-net?")
        elif thermostat == "langevin":
            integrator = LangevinMiddleIntegrator(temp*kelvin,
                coll_freq / picosecond, ts*picoseconds)
            # TODO: what is the difference between picoseconds and picosecond?

    # define biasing potentials
    if md_params["bias"]:
        plumed_file = open(f"{input_dir}/plumed.dat", "r")
        plumed_script = plumed_file.read()
        system.addForce(PlumedForce(plumed_script))
        plumed_file.close()

    # set up simulation
    simulation = Simulation(top.topology, system, integrator)
    simulation.context.setPositions(gro.positions)

    # minimise initial configuration
    if minim:
        simulation.minimizeEnergy()

    # select initial velocities from MB distribution
    if ensemble == "nvt":
        simulation.context.setVelocitiesToTemperature(temp*kelvin)

    return simulation, system, md_params, gro, top, ml_force, output_dir

def simulate(simulation, system, force_field, md_params, gro, top, ml_force, output_dir):

    max_steps = md_params["max_steps"]
    print_trj = md_params["print_trj"]
    print_data = md_params["print_data"]
    print_summary = md_params["print_summary"]
    tot_n_atom = len(gro.getPositions())

    charges = np.zeros(tot_n_atom)
    nb = [f for f in system.getForces() if isinstance(f, NonbondedForce)][0]
    for i in range(system.getNumParticles()):
        charge, sigma, epsilon = nb.getParticleParameters(i)
        charges[i] = charge.value_in_unit(elementary_charge)

    # need to get ligand_n_atom from residue instead
    residues = list(top.topology.residues())
    ligand_n_atom = len(list(residues[0].atoms()))

    if force_field == "pair_net":

        # this is necessary to tell tensorflow to use CPU and GPU when building/predicting
        tf.config.set_visible_devices([], 'GPU')

        mol = read_input.Molecule()
        network = Network(mol)
        print("Loading a trained model...")
        input_dir = "trained_model"
        isExist = os.path.exists(input_dir)
        if not isExist:
            print("ERROR - previously trained model could not be located.")
            exit()
        ligand_atoms = np.loadtxt(f"{input_dir}/nuclear_charges.txt", dtype=np.float32).reshape(-1)
        if len(ligand_atoms) != ligand_n_atom:
            print("ERROR - number of atoms in trained network is incompatible with number of atoms in topology")
            exit()
        mol.n_atom = len(ligand_atoms)
        model = network.load(mol, input_dir)

        # if charges are being predicted enforce ligand net charge
        if md_params["partial_charge"] != "fixed":
            net_charge = md_params["net_charge"]
            charge_model = False

        # load separate network for charge prediction
        if md_params["partial_charge"] == "predicted-sep":
            print("Loading a trained charge model...")
            input_dir = "trained_charge_model"
            isExist = os.path.exists(input_dir)
            if not isExist:
                print("ERROR - previously trained model could not be located.")
                exit()
            charge_model = network.load(mol, input_dir)

    simulation.reporters.append(StateDataReporter(f"./{output_dir}/openmm.csv",
        reportInterval=print_summary,step=True, time=True, potentialEnergy=True,
        kineticEnergy=True, temperature=True, separator=" "))

    # this prevents tensorflow printing warnings or other information
    tf.get_logger().setLevel('ERROR')

    f1 = open(f"./{output_dir}/coords.txt", 'w')
    f2 = open(f"./{output_dir}/forces.txt", 'w')
    f3 = open(f"./{output_dir}/velocities.txt", 'w')
    f4 = open(f"./{output_dir}/energies.txt", 'w')
    f5 = open(f"./{output_dir}/charges.txt", 'w')
    if force_field == "pair_net":
        f6 = open(f"./{output_dir}/ML_forces.txt", 'w')
        f7 = open(f"./{output_dir}/MM_forces.txt", 'w')

        # force capping
        if md_params["force_capping"]:
            force_cap = md_params["force_cap"]
            print(f"Capping forces to {force_cap} kcal/mol/A")

    # run MD simulation for requested number of timesteps
    print("Performing MD simulation...")
    state = simulation.context.getState(getEnergy=True)
    PE = state.getPotentialEnergy() / kilocalories_per_mole
    print("Initial Potential Energy: ", PE, "kcal/mol")

    # sampling using the distance matrix RMSD
    if md_params["adaptive_sampling"]:
        print("Dynamic sampling based on distance matrix RMSD cut-off.")
        f8 = open(f"./{output_dir}/dataset_size.txt", "w")
        f8.write("time (ps) | n_train | rmsd_cut | accept_ratio | conf_cover\n")
        rmsd_cut = md_params["rmsd_cut"]
        n_val = md_params["n_val"]
        n_train = np.zeros((max_steps), dtype=int)
        if md_params["dynamic_cutoff"]:
            rmsd_step = print_data*100

    if md_params["shuffle_perm"]:
        print("Shuffle permutationally equivalent atoms.")
        n_perm_grp, perm_atm, n_symm, n_symm_atm = \
            read_input.perm("md_input/permutations.txt")

    # get torsion surfaces from md_params.txt
    if md_params["cover_conv"]:
        print("Simulation will end when torsional surface populations have converged.")
        converged = False
        conv_time = int(md_params["conv_time"] / md_params["ts"])
        surf_indices = md_params["cover_surf"]
        n_surf = len(surf_indices)
        n_bin_dih = md_params["n_bin"]
        dim = np.zeros((n_surf), dtype=int)
        n_bin = np.zeros((n_surf), dtype=int)
        CV_list = []
        pop = []

        # create list of arrays defining torsion surfaces
        # necessary due to varying number of dimensions in different surfaces
        for i_surf in range(n_surf):
            surf = [eval(i) - 1 for i in surf_indices[i_surf].split()]
            dim[i_surf] = len(surf) / 4
            CV_list.append(np.reshape(np.array(surf), (dim[i_surf], 4)))
            pop.append(np.zeros((n_bin_dih,) * dim[i_surf], dtype=int))
            n_bin[i_surf] = n_bin_dih ** dim[i_surf]

        conf_cover = np.zeros((n_surf,max_steps),dtype=float)
        print()

    for i in range(max_steps):

        coords = simulation.context.getState(getPositions=True). \
            getPositions(asNumpy=True).in_units_of(angstrom)

        if force_field == "pair_net":

            # clears session to avoid running out of memory
            if (i % 1000) == 0:
                tf.keras.backend.clear_session()

            # predict ML forces: predict_on_batch faster for  single structure
            prediction = model.predict_on_batch([np.reshape(coords
                [:ligand_n_atom]/angstrom, (1, -1, 3)),
                np.reshape(ligand_atoms,(1, -1))])
            ML_forces = prediction[0]

            # force capping
            if md_params["force_capping"]:
                ML_forces[ML_forces > force_cap] = force_cap
                ML_forces[ML_forces < -1.0*force_cap] = -1.0*force_cap

            # convert to OpenMM internal units
            ML_forces = np.reshape(ML_forces*kilocalories_per_mole/angstrom, (-1, 3))

            # TODO: we surely don't need to do this on every step?
            if md_params["partial_charge"] != "fixed":

                # predict charges
                ligand_charges = predict_charges(md_params, prediction, charge_model,
                    coords, ligand_n_atom, ligand_atoms, net_charge)

                # assign predicted charges to ML atoms
                nbforce = [f for f in system.getForces() if isinstance(f, NonbondedForce)][0]
                for j in range(ligand_n_atom):
                    [old_charge, sigma,epsilon] = nbforce.getParticleParameters(j)
                    nbforce.setParticleParameters(j, ligand_charges[j], sigma, epsilon)
                    charges[j] = ligand_charges[j]  # TODO: units???
                nbforce.updateParametersInContext(simulation.context)

            # assign predicted forces to ML atoms
            for j in range(ligand_n_atom):
                ml_force.setParticleParameters(j, j, ML_forces[j])
            ml_force.updateParametersInContext(simulation.context)

        # get total forces
        forces = simulation.context.getState(getForces=True). \
            getForces(asNumpy=True).in_units_of(kilocalories_per_mole / angstrom)

        # check MM contribution to forces (should be 0 for pure ML simulation)
        if force_field == "pair_net":
            MM_forces = forces[:ligand_n_atom] - ML_forces

        # advance trajectory one timestep
        simulation.step(1)

        if (i % print_data) == 0:

            state = simulation.context.getState(getEnergy=True)
            vels = simulation.context.getState(getVelocities=True).\
                getVelocities(asNumpy=True).value_in_unit(nanometer / picoseconds)
            forces = simulation.context.getState(getForces=True). \
                getForces(asNumpy=True).in_units_of(kilocalories_per_mole / angstrom)
            ligand_coords = np.reshape(coords[:ligand_n_atom] / angstrom, (1, -1, 3))

            # predicts energies in kcal/mol
            if force_field == "pair_net":
                PE = prediction[1][0][0]
            else:
                PE = state.getPotentialEnergy() / kilocalories_per_mole

            # shuffle permutations
            if md_params["shuffle_perm"]:
                ligand_coords = permute(ligand_coords, n_perm_grp, perm_atm, n_symm, n_symm_atm)

            # add structure to dataset if not doing distance matrix sampling
            if not md_params["adaptive_sampling"]:
                np.savetxt(f1, ligand_coords[0][:ligand_n_atom])
                np.savetxt(f2, forces[:ligand_n_atom])
                np.savetxt(f3, vels[:ligand_n_atom])
                f4.write(f"{PE}\n")
                np.savetxt(f5, charges[:ligand_n_atom])
                if force_field == "pair_net":
                    np.savetxt(f6, ML_forces[:ligand_n_atom])
                    np.savetxt(f7, MM_forces[:ligand_n_atom])

            # adaptive sampling
            else:

                # get the distance matrix for the first structure and add to dataset array
                if i == 0:

                    n_train[i] = 1  # counter for number of training structures
                    n_reject = 0 # counter for number of rejected structures
                    mat_d = analysis.get_rij(ligand_coords, ligand_n_atom, 1)
                    accept = True
                    accept_fract = 1.0

                else:

                    mat_r = analysis.get_rij(ligand_coords, ligand_n_atom, 1)
                    test_mat_d = np.append(mat_d, mat_r, axis=0)
                    accept = rmsd_sample(test_mat_d, rmsd_cut)

                    # get distance matrix for this structure, compare to othr structure
                    n_train[i] = n_train[i-print_data]

                    # check acceptance fraction and dynamically adjust RMSD cut-off accordingly
                    # TODO: make accept_fract and rmsd_cut arrays so they can be monitored during simulation and averages calculated
                    accept_fract = n_train[i] / ((i / print_data))
                    if md_params["dynamic_cutoff"]:
                        if i >= rmsd_step:
                            accept_fract = (n_train[i]-n_train[i-(rmsd_step)])/100
                            if (i % rmsd_step) == 0:
                                rmsd_factor = 1+(accept_fract-0.5)/100
                                rmsd_cut = rmsd_cut * rmsd_factor

                    # save structure to training dataset, populate coverage counter
                    if accept:
                        mat_d = np.append(mat_d, mat_r, axis=0)

                if accept:
                    n_train[i] = mat_d.shape[0]

                    for i_surf in range(n_surf):
                        pop[i_surf] = get_coverage(CV_list[i_surf], ligand_coords, n_bin_dih, pop[i_surf])
                        conf_cover[i_surf][i] = 100.0 * np.count_nonzero(pop[i_surf]) / n_bin[i_surf]

                    time = i * md_params["ts"]
                    np.savetxt(f1, ligand_coords[0][:ligand_n_atom])
                    np.savetxt(f2, forces[:ligand_n_atom])
                    f4.write(f"{PE}\n")
                    np.savetxt(f5, charges[:ligand_n_atom])

                    print_cover = [round(i, 1) for i in conf_cover[:,i].tolist()]
                    f8.write(f"{time:.2f} {n_train[i]:8d} {rmsd_cut:.3f} {accept_fract:.4f} "
                        f"{' '.join(str(j) for j in print_cover)}\n")
                    print(f"{time:.2f} {n_train[i]:8d} {rmsd_cut:.3f} {accept_fract:.4f} "
                        f"{' '.join(str(j) for j in print_cover)}")
                    sys.stdout.flush()

                    # check convergence each time new structure is generated
                    if md_params["cover_conv"]:
                        if time >= 500:
                            converged = check_conv(i, conf_cover, conv_time)

                else:
                    n_reject += 1
                    if n_reject == 1:
                        val_coords = ligand_coords
                        val_forces = np.reshape(forces[:ligand_n_atom], (1, -1, 3))
                        val_charges = np.reshape(charges[:ligand_n_atom], (1, -1))
                        val_energies = np.full(1, PE)
                    else:
                        val_coords = np.append(val_coords, ligand_coords, axis=0)
                        val_forces = np.append(val_forces, np.reshape(
                            forces[:ligand_n_atom],(1, -1, 3)), axis=0)
                        val_energies = np.append(val_energies, np.full(1, PE), axis=0)
                        val_charges = np.append(val_charges, np.reshape(
                            charges[:ligand_n_atom], (1, -1)), axis=0)

                # conformational convergence checks
                if converged:
                    if n_reject < n_val:
                        print("ERROR - not enough rejected structures to form validation set")
                        print(f"Validation set will have {n_reject} structures")
                        indices = list(range(n_reject))
                        n_val = n_reject
                    else:
                        indices = []
                        for i in range(n_val):
                            while True:
                                test_index = random.randint(0, len(val_energies) - 1)
                                if test_index not in indices:
                                    indices.append(test_index)
                                    break

                    # append validation set to end of training set
                    val_energies = np.take(val_energies, indices)
                    val_coords = np.take(val_coords, indices, axis=0)
                    val_forces = np.take(val_forces, indices, axis=0)
                    val_charges = np.take(val_charges, indices, axis=0)
                    np.savetxt(f1,val_coords.reshape(n_val * ligand_n_atom,3))
                    np.savetxt(f2,val_forces.reshape(n_val * ligand_n_atom, 3))
                    np.savetxt(f4,val_energies)
                    np.savetxt(f5,val_charges.flatten())

                    n_train[i] = mat_d.shape[0]
                    time = i * md_params["ts"]
                    f6.write(f"{time:.2f} {n_train[i]:8d} {rmsd_cut:.3f} {accept_fract:.4f} "
                        f"{' '.join(str(j) for j in print_cover)}\n")
                    print("Surface coverage has converged. Ending MD simulation.")
                    print("Number of steps = ", i)
                    print("Fraction of total structures accepted = ",accept_fract)
                    print("Number of rejected structures = ", n_reject)
                    print("Number of training structures = ", n_train[i])
                    print("Number of validation structures = ", n_val)
                    for i_surf in range(conf_cover.shape[0]):
                        conf_cover[i_surf][i] = 100.0 * np.count_nonzero(
                            pop[i_surf]) / n_bin[i_surf]
                        print(f"Conformational coverage for surface {i_surf} =",
                            conf_cover[i_surf][i])
                    sys.stdout.flush()
                    break

        if (i % print_trj) == 0:
            time = simulation.context.getState().getTime().in_units_of(picoseconds)
            vels = simulation.context.getState(getVelocities=True).\
                getVelocities(asNumpy=True).value_in_unit(nanometer / picoseconds)
            coords = coords / nanometer
            vectors = gro.getUnitCellDimensions().value_in_unit(nanometer)
            write_output.grotrj(tot_n_atom, residues, vectors, time,
                coords, vels, gro.atomNames, output_dir, "output")

    f1.close()
    f2.close()
    f3.close()
    f4.close()
    f5.close()
    if force_field == "pair_net":
        f6.close()
        f7.close()
    if md_params["adaptive_sampling"]:
        f8.close()
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

