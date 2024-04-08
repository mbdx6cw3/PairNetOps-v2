from openmm.app import *
from openmm import *
from openmm.unit import *
from openmmplumed import PlumedForce
from openmmtools import integrators
import numpy as np
import write_output, read_input, analysis, os, shutil
from network import Network
import tensorflow as tf

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
    shutil.copy2(f"./{input_dir}/md_params.txt", f"./{output_dir}/")

    temp = md_params["temp"]
    ts = md_params["ts"]
    bias = md_params["bias"]
    ensemble = md_params["ensemble"]
    thermostat = md_params["thermostat"]
    minim = md_params["minim"]
    coll_freq = md_params["coll_freq"]
    gro = GromacsGroFile(f"{input_dir}/input.gro")
    top = GromacsTopFile(f"{input_dir}/input.top",
        periodicBoxVectors=gro.getPeriodicBoxVectors())

    # for rigid water to be found the water residue name must be "HOH"
    system = top.createSystem(nonbondedMethod=PME, nonbondedCutoff=1*nanometer,
        ewaldErrorTolerance=0.0005, constraints=None, removeCMMotion=True,
        rigidWater=True, switchDistance=None)

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
    [pme_alpha, pme_nx, pme_ny, pme_nz] = nb.getPMEParameters()
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
        for i in range(ligand_n_atom):
            for j in range(i):
                nb.addException(i, j, 0, 1, 0)
        print("Number of exceptions: ", nb.getNumExceptions())

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
    if bias:
        plumed_file = open(f"{input_dir}/plumed.dat", "r")
        plumed_script = plumed_file.read()
        system.addForce(PlumedForce(plumed_script))

    # set up simulation
    simulation = Simulation(top.topology, system, integrator)
    simulation.context.setPositions(gro.positions)

    # minimise initial configuration
    if minim:
        simulation.minimizeEnergy()

    # select initial velocities from MB distribution
    if ensemble == "nvt":
        simulation.context.setVelocitiesToTemperature(temp*kelvin)

    return simulation, system, output_dir, md_params, gro, top, ml_force

def simulate(simulation, system, force_field, output_dir, md_params, gro, top, ml_force):

    n_steps = md_params["n_steps"]
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

    # run MD simulation for requested number of timesteps
    print("Performing MD simulation...")
    state = simulation.context.getState(getEnergy=True)
    PE = state.getPotentialEnergy() / kilocalories_per_mole
    print("Initial Potential Energy: ", PE, "kcal/mol")

    if md_params["D_sample"]:
        print("Dynamic sampling based on distance matrix cut-off.")
        D_start = md_params["D_start"]
        D_conv = md_params["D_conv"]
        D_cut = md_params["D_cut"]
        f6 = open(f"./{output_dir}/dataset_size.txt", "w")
        n_structure = np.zeros((n_steps))

    for i in range(n_steps):
        print_coords = False
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

            # convert to OpenMM internal units
            ligand_forces = np.reshape(prediction[0]
                *kilocalories_per_mole/angstrom, (-1, 3))

            # assign predicted forces to ML atoms
            for j in range(ligand_n_atom):
                ml_force.setParticleParameters(j, j, ligand_forces[j])
            ml_force.updateParametersInContext(simulation.context)

            # assign predicted charges to ML atoms
            # TODO: we surely don't need to do this on every step?
            if md_params["partial_charge"] != "fixed":

                # get charge prediction from same network as forces / energies
                if md_params["partial_charge"] == "predicted":
                    ligand_charges = prediction[2].T
                # get charge prediction from separate network
                elif md_params["partial_charge"] == "predicted-sep":
                    charge_prediction = charge_model.predict_on_batch([np.reshape(coords[
                        :ligand_n_atom] / angstrom, (1, -1, 3)), np.reshape(ligand_atoms,(1, -1))])
                    ligand_charges = charge_prediction[2].T

                # correct predicted partial charges so that ligand has correct net charge
                corr = (sum(ligand_charges) - net_charge) / mol.n_atom
                for atm in range(mol.n_atom):
                    ligand_charges[atm] = ligand_charges[atm] - corr

                nbforce = [f for f in system.getForces() if isinstance(f, NonbondedForce)][0]
                for j in range(ligand_n_atom):
                    [old_charge, sigma, epsilon] = nbforce.getParticleParameters(j)
                    nbforce.setParticleParameters(j, ligand_charges[j], sigma, epsilon)
                    charges[j] = ligand_charges[j] # TODO: units???
                nbforce.updateParametersInContext(simulation.context)

        # advance trajectory one timestep
        simulation.step(1)

        coords = simulation.context.getState(getPositions=True). \
            getPositions(asNumpy=True).in_units_of(angstrom)

        # if doing distance matrix sampling decide whether to save structure
        if md_params["D_sample"]:
            if i == 0:
                mat_r = analysis.get_rij(coords, 1)
                mat_d = mat_r
                print_coords = True
                size = 1

            if i > D_start:
                if (i % print_data) == 0:
                    mat_r = analysis.get_rij(coords, 1)
                    test_mat_d = np.append(mat_d, mat_r, axis=0)
                    print_coords = True

                    for j in range(mat_d.shape[0]):
                        D = analysis.D_rmsd(-1, j, test_mat_d)
                        if D < D_cut:
                            print_coords = False
                            break

                    if print_coords:
                        mat_d = np.append(mat_d, mat_r, axis=0)
                        size += 1
                        #print(i, size)

        # print as normal if not using distance matrix or before D_start
        elif (i % print_data) == 0 or i == 0:
            print_coords = True

        if print_coords:
            state = simulation.context.getState(getEnergy=True)
            vels = simulation.context.getState(getVelocities=True).\
                getVelocities(asNumpy=True).value_in_unit(nanometer / picoseconds)
            forces = simulation.context.getState(getForces=True). \
                getForces(asNumpy=True).in_units_of(kilocalories_per_mole / angstrom)

            # predicts energies in kcal/mol
            if force_field == "pair_net":
                PE = prediction[1][0][0]
            else:
                PE = state.getPotentialEnergy() / kilocalories_per_mole

            np.savetxt(f1, coords[:ligand_n_atom])
            np.savetxt(f2, forces[:ligand_n_atom])
            np.savetxt(f3, vels[:ligand_n_atom])
            f4.write(f"{PE}\n")
            np.savetxt(f5, charges[:ligand_n_atom])
            if md_params["D_sample"]:
                f6.write(f"{i} {size}\n")

        if (i % print_trj) == 0:
            time = simulation.context.getState().getTime().in_units_of(picoseconds)
            vels = simulation.context.getState(getVelocities=True).\
                getVelocities(asNumpy=True).value_in_unit(nanometer / picoseconds)
            coords = coords / nanometer
            vectors = gro.getUnitCellDimensions().value_in_unit(nanometer)
            write_output.grotrj(tot_n_atom, residues, vectors, time,
                coords, vels, gro.atomNames, output_dir, "output")

        # convergence check wrt to number of structures
        if md_params["D_sample"]:
            n_structure[i] = size
            if n_structure[i] == n_structure[i-D_conv]:
                print("Dataset size has converged. Ending MD simulation.")
                break

    f1.close()
    f2.close()
    f3.close()
    f4.close()
    f5.close()
    f6.close()
    return None

